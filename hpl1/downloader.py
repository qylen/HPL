"""Robust historical archive downloader.

Features: resumable (.part + HTTP Range), retries with exponential backoff and
jitter, sha256 checksum validation against official .CHECKSUM files, duplicate
avoidance, structured manifests, size guard requiring explicit acknowledgement,
configurable concurrency. Never silently skips a failed date: failures are
collected and reported.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

from .binance_archive import ArchiveKey, archive_key, iter_dates

USER_AGENT = "HPL-1-research/1.0 (+public archive use; no credentials)"


@dataclass
class DownloadResult:
    key: str
    url: str
    path: str
    status: str            # ok | skipped_existing | checksum_missing | failed
    bytes_downloaded: int = 0
    size: int = 0
    checksum_verified: bool = False
    attempts: int = 0
    error: str | None = None


@dataclass
class DownloadManifest:
    market: str
    symbol: str
    dtype: str
    start: str
    end: str
    started_utc: str
    finished_utc: str | None = None
    results: list[DownloadResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            **{k: v for k, v in asdict(self).items() if k != "results"},
            "results": [asdict(r) for r in self.results],
        }


def raw_path_for(raw_dir: str | Path, market: str, symbol: str, dtype: str) -> Path:
    return Path(raw_dir) / market / symbol.upper() / dtype


def sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def fetch_checksum(key: ArchiveKey, session: requests.Session, timeout: float = 30.0) -> str | None:
    """Official .CHECKSUM files contain 'sha256  filename' (some old ones md5)."""
    try:
        resp = session.get(key.checksum_url, timeout=timeout)
        if resp.status_code != 200:
            return None
        text = resp.text.strip()
        if not text:
            return None
        first = text.splitlines()[0].split()
        digest = first[0].lower()
        if len(digest) in (32, 64):  # md5 or sha256 hex
            return digest
        return None
    except requests.RequestException:
        return None


def download_one(
    key: ArchiveKey,
    dest_dir: Path,
    session: requests.Session,
    max_retries: int = 5,
    timeout: float = (30.0, 300.0),
) -> DownloadResult:
    """Download a single archive with resume, retry, checksum validation."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(key.key).name
    part = dest.with_suffix(dest.suffix + ".part")
    result = DownloadResult(key=key.key, url=key.url, path=str(dest), status="failed")
    expected = fetch_checksum(key, session)

    if dest.exists() and dest.stat().st_size > 0:
        if expected and sha256_of_file(dest) == expected:
            return DownloadResult(key=key.key, url=key.url, path=str(dest),
                                  status="skipped_existing", size=dest.stat().st_size,
                                  checksum_verified=True, attempts=0)
        if not expected:
            return DownloadResult(key=key.key, url=key.url, path=str(dest),
                                  status="skipped_existing", size=dest.stat().st_size,
                                  checksum_verified=False, attempts=0,
                                  error="existing file kept; no official checksum available")
        # checksum mismatch on existing file -> re-download
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        result.attempts = attempt
        try:
            headers = {"User-Agent": USER_AGENT}
            resume_from = part.stat().st_size if part.exists() else 0
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"
            with session.get(key.url, headers=headers, timeout=timeout, stream=True) as resp:
                if resp.status_code == 416 and resume_from:
                    # range not satisfiable: part file already complete or stale
                    part.unlink(missing_ok=True)
                    continue
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", 0) or 0)
                mode = "ab" if resume_from > 0 and resp.status_code == 206 else "wb"
                if resp.status_code == 200:
                    part.unlink(missing_ok=True)
                    mode = "wb"
                with part.open(mode) as fh:
                    for block in resp.iter_content(chunk_size=1 << 20):
                        fh.write(block)
            got = part.stat().st_size
            if total and got != total and mode == "wb":
                raise OSError(f"truncated download: {got}/{total} bytes")
            if expected:
                digest = sha256_of_file(part)
                if digest != expected:
                    part.unlink(missing_ok=True)
                    raise OSError(f"checksum mismatch (sha256 {digest[:16]}... != expected {expected[:16]}...)")
                verified = True
            else:
                verified = False
            part.replace(dest)
            result.status = "ok" if verified else "checksum_missing"
            result.checksum_verified = verified
            result.size = dest.stat().st_size
            result.bytes_downloaded = got - (resume_from if mode == "ab" else 0)
            return result
        except Exception as exc:  # noqa: BLE001 - report and retry
            result.error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                sleep = backoff * (0.5 + random.random())
                time.sleep(min(sleep, 30.0))
                backoff = min(backoff * 2, 60.0)
    result.error = result.error or "unknown failure"
    return result


def download_range(
    market: str,
    symbol: str,
    dtype: str,
    start: str,
    end: str,
    raw_dir: str | Path,
    session: requests.Session | None = None,
    max_retries: int = 5,
    max_concurrency: int = 4,
    acknowledged: bool = False,
    large_download_allow_mb: float = 500.0,
) -> tuple[list[DownloadResult], dict]:
    """Download all daily archives for a range. Returns (results, manifest dict)."""
    from concurrent.futures import ThreadPoolExecutor

    dates = list(iter_dates(start, end))
    dest_dir = raw_path_for(raw_dir, market, symbol, dtype)
    sess = session or requests.Session()

    # Pre-check sizes via HEAD so the guard sees the real total.
    total_expected = 0
    sized: list[ArchiveKey] = []
    for d in dates:
        k = archive_key(market, symbol, dtype, d)
        try:
            with sess.head(k.url, timeout=15.0, allow_redirects=True) as h:
                sz = int(h.headers.get("Content-Length", 0) or 0)
            sized.append(k.__class__(**{**k.__dict__, "size": sz}))
            total_expected += sz
        except requests.RequestException:
            sized.append(k)
    keys = sized
    approx_mb = total_expected / 1e6
    if approx_mb > large_download_allow_mb and not acknowledged:
        raise PermissionError(
            f"estimated download {approx_mb:.0f} MB exceeds the configured guard "
            f"({large_download_allow_mb:.0f} MB). Re-run with --yes to acknowledge."
        )

    manifest = DownloadManifest(
        market=market, symbol=symbol.upper(), dtype=dtype, start=start, end=end,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    results: list[DownloadResult] = []
    worker = max(1, min(max_concurrency, len(keys) or 1))
    if len(keys) == 1:
        results = [download_one(k, dest_dir, sess, max_retries=max_retries) for k in keys]
    else:
        with ThreadPoolExecutor(max_workers=worker) as pool:
            futures = {pool.submit(download_one, k, dest_dir, sess, max_retries): k for k in keys}
            for fut, k in futures.items():
                results.append(fut.result())

    results.sort(key=lambda r: r.key)
    manifest.results = results
    manifest.finished_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary = {
        "ok": sum(1 for r in results if r.status == "ok"),
        "skipped_existing": sum(1 for r in results if r.status == "skipped_existing"),
        "checksum_missing": sum(1 for r in results if r.status == "checksum_missing"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "approx_total_mb": round(approx_mb, 1),
        "guard": {"limit_mb": large_download_allow_mb, "acknowledged": acknowledged},
    }
    return results, {"manifest": manifest.to_dict(), "summary": summary}


def save_manifest(manifest: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def print_results(results: list[DownloadResult], label: str) -> None:
    print(f"[download] {label}: {len(results)} file(s)")
    for r in results:
        mark = {"ok": "OK   ", "skipped_existing": "SKIP ", "checksum_missing": "NOCRC", "failed": "FAIL "}.get(r.status, "?????")
        size = f"{r.size / 1e6:8.1f}MB" if r.size else "        -"
        err = f"  {r.error}" if r.error and r.status == "failed" else ""
        print(f"  {mark} {Path(r.key).name:48s} {size} attempts={r.attempts}{err}")
    failed = [r for r in results if r.status == "failed"]
    if failed:
        print(f"[download] WARNING: {len(failed)} failed file(s) — NOT silently skipped; inspect errors above.")
