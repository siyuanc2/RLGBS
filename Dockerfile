# Base image
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel

# Make RUN commands use bash login shell
SHELL ["/bin/bash", "--login", "-c"]

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    linux-headers-generic \
    libopenblas-dev \
    liblapack3 \
    liblapack-dev \
    wget \
    lsb-release \
    curl \
    gpg \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install Armadillo
WORKDIR /opt
RUN wget https://sourceforge.net/projects/arma/files/armadillo-14.0.3.tar.xz \
    && tar -xf armadillo-14.0.3.tar.xz \
    && cd armadillo-14.0.3 \
    && cmake . \
    && make install \
    && cd .. \
    && rm -rf armadillo-14.0.3*

# Install Redis
RUN curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list \
    && apt-get update \
    && apt-get install -y redis \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install \
    stable-baselines3==2.2.1 \
    redis[hiredis] \
    pybind11 \
    scipy

# Create directory for source code
WORKDIR /opt/RLGBS-source

# Create startup script
COPY <<'EOF' /usr/local/bin/startup.sh
#!/bin/bash
# Start redis server in background
tmux new-session -d redis-server

# Wait for Redis to be ready
echo "Waiting for Redis to start..."
until redis-cli ping &>/dev/null
do
    sleep 1
done
echo "Redis server is ready!"

# Configure redis as a LRU cache
redis-cli CONFIG SET maxmemory 20gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli CONFIG SET save ""

# Execute command if provided, otherwise start bash
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec /bin/bash
fi
EOF

RUN chmod +x /usr/local/bin/startup.sh

# Set startup script as entrypoint
ENTRYPOINT ["/usr/local/bin/startup.sh"]