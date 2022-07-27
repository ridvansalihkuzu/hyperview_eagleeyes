FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    python3-pyqt5 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip


# Install Project Specific Dependencies
ADD requirements.txt .
RUN pip install -r requirements.txt


# Set the default command to python3
# CMD ["python3"]
