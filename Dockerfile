FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.4 support
RUN pip3 install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

WORKDIR /workspace
COPY . .

RUN pip3 install --no-cache-dir -e ".[dev]"

CMD ["pytest", "smoke_test.py", "-v", "-m", "gpu"]
