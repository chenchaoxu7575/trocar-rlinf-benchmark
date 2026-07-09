#!/usr/bin/env bash
# Native RLinf per-worker Nsight profiling: run normal training with
# cluster.nsight enabled (config's ..._prof yaml, steps:[3,4]) so RLinf wraps
# EACH worker with its own `nsys profile`, producing per-worker
# rlinf_nsight_{Worker}_{pid}.nsys-rep. Uses the v0.2-based RLinf checkout in
# trocar-image/RLinf (mounted over the wheel) + the latest-nsys image.
#
# Usage:
#   GPUS='"device=0,1,2,3,4,5,6,7"' bash scripts/run_nsys_native.sh
set -euo pipefail

DK=/home/chenchaox/project/rlinf_pub/env/bin/docker
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="trocar-rlinf-prof-nsys:latest"          # latest nsys (2026.1.3) on PATH
CONFIG_NAME="isaaclab_ppo_gr00t_assemble_trocar_prof"
MAX_EPOCHS="${MAX_EPOCHS:-5}"                    # need >=5 to reach RL steps [3,4]

MODELS="${HERE}/models"; OUTPUT="${HERE}/output"
CACHE="${HERE}/cache/isaac-sim"; ASSETS="${HERE}/cache/isaac-assets"
LOGS="${OUTPUT}/rlinf_logs"                      # nsights land here (mounted out)
mkdir -p "${OUTPUT}" "${CACHE}"/{kit,ov,glcache,computecache} "${ASSETS}" "${LOGS}"
chmod -R 777 "${OUTPUT}" 2>/dev/null || true

IL="${HERE}/IsaacLab"
CFG_HOST="${IL}/source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config"
CFG_CTR="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config"
ASSETS_PY_HOST="${IL}/source/isaaclab/isaaclab/utils/assets.py"
ASSETS_PY_CTR="/workspace/isaaclab/source/isaaclab/isaaclab/utils/assets.py"
# our v0.2 + nsys RLinf source, mounted over the pip wheel
RLINF_HOST="${HERE}/RLinf/rlinf"
RLINF_CTR="/isaac-sim/kit/python/lib/python3.12/site-packages/rlinf"
# RLinf writes per-worker nsys reports under <log_path>/nsights; log_path is set
# by train.py to scripts/reinforcement_learning/rlinf/logs/... -> mount it out.
RLLOG_CTR="/workspace/isaaclab/scripts/reinforcement_learning/rlinf/logs"

CNAME="trocar_nsys_native"
$DK rm -f "${CNAME}" 2>/dev/null || true
$DK run -d --name "${CNAME}" --gpus "${GPUS:-all}" --network host \
  --entrypoint bash --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add=SYS_ADMIN \
  -e OMNI_KIT_ACCEPT_EULA=yes -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ALLOW_ROOT=1 \
  -e HF_HUB_OFFLINE=1 \
  -e RAY_enable_worker_prestart=0 \
  -e RAY_worker_lease_timeout_milliseconds=120000 \
  -v "${RLINF_HOST}:${RLINF_CTR}:ro" \
  -v "${MODELS}/Assemble_Trocar:/models/Assemble_Trocar:ro" \
  -v "${ASSETS}:/tmp/Assets" \
  -v "${OUTPUT}:/workspace/isaaclab/output" \
  -v "${LOGS}:${RLLOG_CTR}" \
  -v "${CFG_HOST}/${CONFIG_NAME}.yaml:${CFG_CTR}/${CONFIG_NAME}.yaml:ro" \
  -v "${ASSETS_PY_HOST}:${ASSETS_PY_CTR}:ro" \
  -v "${CACHE}/kit:/isaac-sim/kit/cache" \
  -v "${CACHE}/ov:/root/.cache/ov" \
  -v "${CACHE}/computecache:/root/.nv/ComputeCache" \
  -v "${CACHE}/glcache:/root/.cache/nvidia/GLCache" \
  "${IMAGE}" -lc "
    set -e
    echo '=== native nsight run (steps [3,4], per-worker) ==='; nsys --version | head -1; date
    cd /workspace/isaaclab && ./isaaclab.sh train --rl_library rlinf \
      --config_name ${CONFIG_NAME} \
      --model_path /models/Assemble_Trocar --num_envs 64 --max_epochs ${MAX_EPOCHS}
    echo '=== RUN DONE ==='; date
    echo '=== per-worker nsys reports produced: ==='; find /workspace/isaaclab/scripts/reinforcement_learning/rlinf/logs -name 'rlinf_nsight_*.nsys-rep' | head -40
  "
echo "launched ${CNAME}: $?"
