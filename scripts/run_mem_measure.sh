#!/usr/bin/env bash
# Measure per-PROCESS GPU memory over a trocar RL run to quantify the colocated
# limitation: Isaac Sim is non-offloadable, so its VRAM is dead residual during
# the train phase. Logs nvidia-smi per-PID memory every 2s + ps snapshots (to map
# pid -> Isaac-child / Actor / Rollout / EnvWorker). nsys OFF (clean memory).
# Usage: GPUS='"device=0,1,2,3,4,5,6,7"' bash scripts/run_mem_measure.sh
set -euo pipefail

DK=/home/chenchaox/project/rlinf_pub/env/bin/docker
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="trocar-rlinf-prof-nsys:latest"
CONFIG_NAME="${CONFIG_NAME:-isaaclab_ppo_gr00t_assemble_trocar_memtest}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"
NUM_ENVS="${NUM_ENVS:-64}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

MODELS="${HERE}/models"; OUTPUT="${HERE}/output"
CACHE="${HERE}/cache/isaac-sim"; ASSETS="${HERE}/cache/isaac-assets"
LOGS="${OUTPUT}/rlinf_logs"
MEM_OUT="${OUTPUT}/mem_measure/${RUN_TAG}"
mkdir -p "${OUTPUT}" "${CACHE}"/{kit,ov,glcache,computecache} "${ASSETS}" "${LOGS}" "${MEM_OUT}"
chmod -R 777 "${OUTPUT}" 2>/dev/null || true

IL="${HERE}/IsaacLab"
CFG_HOST="${IL}/source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config"
CFG_CTR="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config"
ASSETS_PY_HOST="${IL}/source/isaaclab/isaaclab/utils/assets.py"
ASSETS_PY_CTR="/workspace/isaaclab/source/isaaclab/isaaclab/utils/assets.py"
RLINF_HOST="${HERE}/RLinf/rlinf"
RLINF_CTR="/isaac-sim/kit/python/lib/python3.12/site-packages/rlinf"
RLLOG_CTR="/workspace/isaaclab/scripts/reinforcement_learning/rlinf/logs"
MEM_OUT_CTR="/workspace/isaaclab/output/mem_measure/${RUN_TAG}"

CNAME="trocar_mem_measure"
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
    OUT=${MEM_OUT_CTR}; mkdir -p \$OUT
    echo '=== mem-measure run (nsys OFF) ==='; date
    # per-PID GPU memory every 2s (needs --pid host to attribute to host PIDs)
    nvidia-smi --query-compute-apps=timestamp,gpu_bus_id,pid,used_memory --format=csv -lms 2000 > \$OUT/mem.csv 2>/dev/null &
    SMI=\$!
    # total per-GPU memory every 2s (always works; fallback for phase analysis)
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv -lms 2000 > \$OUT/gpu_total.csv 2>/dev/null &
    SMI2=\$!
    # ps snapshots every 3s to map pid->role (Isaac child = python3 with ppid=EnvWorker)
    ( while true; do echo \"T=\$(date +%s)\"; ps -eo pid,ppid,rss,comm,args --sort=-rss | grep -iE 'python3|kit' | grep -v grep | head -30; echo '==='; sleep 3; done ) > \$OUT/ps.log 2>/dev/null &
    PS=\$!
    cd /workspace/isaaclab && ./isaaclab.sh train --rl_library rlinf \
      --config_name ${CONFIG_NAME} \
      --model_path /models/Assemble_Trocar --num_envs ${NUM_ENVS} --max_epochs ${MAX_EPOCHS} || true
    kill \$SMI \$SMI2 \$PS 2>/dev/null || true
    echo '=== MEM MEASURE DONE ==='; date; wc -l \$OUT/mem.csv \$OUT/gpu_total.csv \$OUT/ps.log
  "
echo "launched ${CNAME}: $?  -> ${MEM_OUT}"
