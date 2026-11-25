FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-tk libgl1 libglib2.0-0 libusb-1.0-0 \
    xterm xauth xvfb ca-certificates bash && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/teleop
ENV PATH="/opt/teleop/bin:$PATH"
WORKDIR /workspace/dobot-teleop
COPY requirements-teleop-linux.txt ./
RUN python -m pip install --no-cache-dir -r requirements-teleop-linux.txt pytest==8.3.5
COPY . .
RUN mkdir -p /workspace/datasets /workspace/.local/runtime .local
# Default is hardware-free regression/data flow. Real operation is explicit.
CMD ["python", "tools/container_test.py"]
