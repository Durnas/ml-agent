# =============================================================
# Universal ML Training Image
# - PyTorch + CUDA runtime pre-installed (no local code baked in)
# - Clones the repo given via GITHUB_URL env var at runtime,
#   installs its requirements.txt (if present) and runs train.py
# =============================================================
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# git is required for runtime cloning; keep the layer slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# The entrypoint is infrastructure glue, NOT user/training code
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
