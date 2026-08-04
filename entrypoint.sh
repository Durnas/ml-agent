#!/bin/bash
# ==============================================================
# Runtime entrypoint for the Universal ML Training Image.
# Expects: GITHUB_URL  - full URL of a repo containing train.py
# ==============================================================
set -euo pipefail

if [ -z "${GITHUB_URL:-}" ]; then
    echo "[entrypoint] ERROR: GITHUB_URL environment variable is not set." >&2
    exit 1
fi

echo "[entrypoint] Cloning repository: ${GITHUB_URL}"
git clone --depth 1 "${GITHUB_URL}" /workspace/repo
cd /workspace/repo

# Install repo-specific dependencies only if the file exists
if [ -f "requirements.txt" ]; then
    echo "[entrypoint] Installing requirements.txt ..."
    pip install --no-cache-dir -r requirements.txt
else
    echo "[entrypoint] No requirements.txt found - skipping dependency install."
fi

if [ ! -f "train.py" ]; then
    echo "[entrypoint] ERROR: train.py not found in repository root." >&2
    exit 1
fi

echo "[entrypoint] Starting training (python train.py) ..."
# -u = unbuffered stdout so logs stream live to kubectl / the agent
exec python -u train.py
