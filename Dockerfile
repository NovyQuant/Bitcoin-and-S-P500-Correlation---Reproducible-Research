FROM python:3.11

WORKDIR /app

# system deps + Quarto CLI (arch-aware: amd64 or arm64)
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates make; \
    ARCH=$(dpkg --print-architecture); \
    curl -fLO "https://github.com/quarto-dev/quarto-cli/releases/download/v1.5.57/quarto-1.5.57-linux-${ARCH}.deb"; \
    apt-get install -y "./quarto-1.5.57-linux-${ARCH}.deb"; \
    rm "quarto-1.5.57-linux-${ARCH}.deb"; \
    quarto --version; \
    rm -rf /var/lib/apt/lists/*

# Copy entire project (including CSV data files)
COPY . .

# Python dependencies
RUN pip install --upgrade pip && pip install -e '.[dev]'

RUN mkdir -p outputs

# When professor runs the container -> generate the report
CMD ["make", "report"]
