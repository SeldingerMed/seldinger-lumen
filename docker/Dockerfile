FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=disable \
    LUMEN_BACKEND_LOG_LEVEL=warning

WORKDIR /opt/lumen

RUN apt-get update \
    && apt-get install --no-install-recommends --yes ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY lumen ./lumen
COPY benchmarks ./benchmarks
COPY tools ./tools

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir ".[solver]"

RUN useradd --create-home --uid 10001 lumen
USER lumen

ENTRYPOINT ["lumen"]
CMD ["doctor", "--json"]
