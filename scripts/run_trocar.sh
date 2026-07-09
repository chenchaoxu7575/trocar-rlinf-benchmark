#!/usr/bin/env bash
# Run the assemble_trocar RLinf PPO+GR00T sample inside the trocar image.
# Usage: run_trocar.sh [train|play] [extra isaaclab.sh args...]
set -euo pipefail

DK=/home/chenchaox/project/rlinf_pub/env/bin/docker
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="trocar-rlinf:latest"
MODE="${1:-train}"; shift || true

MODELS="${HERE}/models"
OUTPUT="${HERE}/output"
CACHE="${HERE}/cache/isaac-sim"
mkdir -p "${OUTPUT}" "${CACHE}"/{kit,ov,pip,glcache,computecache,logs}

chmod -R 777 "${OUTPUT}" 2>/dev/null || true
CNAME="trocar_${MODE}"
$DK rm -f "${CNAME}" 2>/dev/null || true
$DK run --name "${CNAME}" --gpus "${GPUS:-all}" --network host \
  --entrypoint bash \
  --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
  -e OMNI_KIT_ACCEPT_EULA=yes -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ALLOW_ROOT=1 \
  -e HF_HUB_OFFLINE=1 \
  -v "${MODELS}/Assemble_Trocar:/models/Assemble_Trocar:ro" \
  -v "${OUTPUT}:/workspace/isaaclab/output" \
  -v "${CACHE}/kit:/isaac-sim/kit/cache" \
  -v "${CACHE}/ov:/root/.cache/ov" \
  -v "${CACHE}/pip:/root/.cache/pip" \
  -v "${CACHE}/glcache:/root/.cache/nvidia/GLCache" \
  -v "${CACHE}/computecache:/root/.nv/ComputeCache" \
  -v "${CACHE}/logs:/root/.nvidia-omniverse/logs" \
  "${IMAGE}" \
  -lc "cd /workspace/isaaclab && ./isaaclab.sh ${MODE} --rl_library rlinf \
      --config_name isaaclab_ppo_gr00t_assemble_trocar \
      --model_path /models/Assemble_Trocar $* \
      2>&1 | tee output/${MODE}_\$(date +%Y%m%d_%H%M%S).log"
