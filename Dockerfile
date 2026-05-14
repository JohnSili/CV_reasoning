FROM python:3.10-slim

# System deps for PyBullet + headless rendering
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    xvfb \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/

RUN mkdir -p data output

ENV DISPLAY=:99
ENV PYBULLET_EGL=1

# Start Xvfb and run generation
CMD ["bash", "-c", "Xvfb :99 -screen 0 1024x768x24 &> /dev/null & sleep 1 && python scripts/generate_dataset.py"]