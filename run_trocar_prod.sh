#!/usr/bin/env bash
# Run the PRODUCTION-scale assemble_trocar RLinf PPO+GR00T benchmark.
# Mounts host-edited IsaacLab files over the image's baked copies so config /
# camera / cache-isolation changes take effect without a rebuild.
#
# Usage:
#   GPUS='"device=0,1,2,3,4,5,6,7"' bash run_trocar_prod.sh train --num_envs 64 --max_epochs 3
#   GPUS='"device=0,1,2,3,4,5,6,7"' bash run_trocar_prod.sh train --num_envs 8 --max_epochs 1   # fast startup test
set -euo pipefail

DK=/home/chenchaox/project/rlinf_pub/env/bin/docker
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="trocar-rlinf:latest"
MODE="${1:-train}"; shift || true
CONFIG_NAME="${CONFIG_NAME:-isaaclab_ppo_gr00t_assemble_trocar_prod}"

MODELS="${HERE}/models"
OUTPUT="${HERE}/output"
CACHE="${HERE}/cache/isaac-sim"
# Persistent local mirror of the S3 healthcare/Isaac assets. IsaacLab's
# retrieve_file_path() downloads to tempfile.gettempdir()==/tmp/Assets with an
# unlocked check-then-use, so 8 concurrent env workers race and read partial
# .usd files ("Failed reading N bytes at offset 0"). Mounting a persistent,
# PRE-WARMED /tmp/Assets makes every file complete before the workers start,
# so they skip the download and only read -> no race. Warm once with
# WARMUP=1 (single process) before the multi-GPU run.
ASSETS="${HERE}/cache/isaac-assets"
mkdir -p "${OUTPUT}" "${CACHE}"/{kit,ov,pip,glcache,computecache,logs} "${ASSETS}"
chmod -R 777 "${OUTPUT}" 2>/dev/null || true

# Host-edited IsaacLab files (on branch chenchaox/trocar-8gpu-prod-benchmark).
IL="${HERE}/IsaacLab"
CFG_HOST="${IL}/source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config"
CFG_CTR="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config"
EXT_HOST="${IL}/source/isaaclab_contrib/isaaclab_contrib/rl/rlinf/extension.py"
EXT_CTR="/workspace/isaaclab/source/isaaclab_contrib/isaaclab_contrib/rl/rlinf/extension.py"
# Atomic-download fix for the concurrent USD asset race (assets.py).
ASSETS_PY_HOST="${IL}/source/isaaclab/isaaclab/utils/assets.py"
ASSETS_PY_CTR="/workspace/isaaclab/source/isaaclab/isaaclab/utils/assets.py"

CNAME="trocar_prod_${MODE}"
$DK rm -f "${CNAME}" 2>/dev/null || true
$DK run --name "${CNAME}" --gpus "${GPUS:-all}" --network host \
  --entrypoint bash \
  --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
  -e OMNI_KIT_ACCEPT_EULA=yes -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ALLOW_ROOT=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TROCAR_PER_RANK_CACHE="${TROCAR_PER_RANK_CACHE:-1}" \
  -v "${MODELS}/Assemble_Trocar:/models/Assemble_Trocar:ro" \
  -v "${ASSETS}:/tmp/Assets" \
  -v "${OUTPUT}:/workspace/isaaclab/output" \
  -v "${CFG_HOST}/isaaclab_ppo_gr00t_assemble_trocar_prod.yaml:${CFG_CTR}/isaaclab_ppo_gr00t_assemble_trocar_prod.yaml:ro" \
  -v "${CFG_HOST}/camera_config.py:${CFG_CTR}/camera_config.py:ro" \
  -v "${CFG_HOST}/gr00t_config.py:${CFG_CTR}/gr00t_config.py:ro" \
  -v "${EXT_HOST}:${EXT_CTR}:ro" \
  -v "${ASSETS_PY_HOST}:${ASSETS_PY_CTR}:ro" \
  -v "${CACHE}/kit:/isaac-sim/kit/cache" \
  -v "${CACHE}/ov:/root/.cache/ov" \
  -v "${CACHE}/pip:/root/.cache/pip" \
  -v "${CACHE}/glcache:/root/.cache/nvidia/GLCache" \
  -v "${CACHE}/computecache:/root/.nv/ComputeCache" \
  -v "${CACHE}/logs:/root/.nvidia-omniverse/logs" \
  "${IMAGE}" \
  -lc "cd /workspace/isaaclab && ./isaaclab.sh ${MODE} --rl_library rlinf \
      --config_name ${CONFIG_NAME} \
      --model_path /models/Assemble_Trocar $* \
      2>&1 | tee output/prod_${MODE}_\$(date +%Y%m%d_%H%M%S).log"
