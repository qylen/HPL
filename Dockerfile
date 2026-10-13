# HPL-1 research and paper-trading image (NO live-order execution client).
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY hpl1/ hpl1/
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

# Run as non-root; paper trading and research never need more.
RUN useradd -m hpl && mkdir -p /app/data /app/artifacts /app/logs /app/run \
    && chown -R hpl:hpl /app
USER hpl

# Default: the offline test suite. Override for other commands, e.g.
#   docker run hpl1 python -m hpl1 research --market futures ...
CMD ["python", "-m", "pytest", "-q"]
