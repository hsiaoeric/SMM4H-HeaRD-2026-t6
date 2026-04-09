FROM nvcr.io/nvidia/pytorch:26.03-py3

WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy Docker-specific pyproject.toml (excludes torch/vllm already in NGC)
COPY docker/pyproject.toml ./

# Install remaining dependencies
RUN uv sync --no-install-project

# Copy project source
COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/

# Default: run training
ENTRYPOINT ["uv", "run", "python", "src/train.py"]
