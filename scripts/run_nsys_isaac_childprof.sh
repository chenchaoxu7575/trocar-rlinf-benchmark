#!/usr/bin/env bash
# Design B: capture the SPAWNED Isaac Sim subprocess's CUDA by giving it its OWN
# nsys session (venv.py -> mp.set_executable wrapper), step-gated by the child's
# own torch.cuda.profiler.start/stop (steps [3,4], warmup skipped). The parent
# EnvWorker is NOT nsys-wrapped (EnvGroup dropped from worker_groups) so the
# child's nsys does not nest. Actor/Rollout still profiled the normal way.
#
# Produces: rlinf_nsight_IsaacSim_<pid>.nsys-rep under output/isaac_nsys/
# Usage:  GPUS='"device=0,1,2,3,4,5,6,7"' bash scripts/run_nsys_isaac_childprof.sh
set -euo pipefail

DK=/home/chenchaox/project/rlinf_pub/env/bin/docker
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="trocar-rlinf-prof-nsys:latest"          # nsys 2026.1.3 on PATH
CONFIG_NAME="${CONFIG_NAME:-isaaclab_ppo_gr00t_assemble_trocar_prof_bchild}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"                    # >=5 to reach RL steps [3,4]

MODELS="${HERE}/models"; OUTPUT="${HERE}/output"
CACHE="${HERE}/cache/isaac-sim"; ASSETS="${HERE}/cache/isaac-assets"
LOGS="${OUTPUT}/rlinf_logs"
# Each run gets its OWN Isaac-report dir so per-experiment .nsys-rep files don't
# pile up together. Override RUN_TAG to name it; defaults to a timestamp.
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ISAAC_NSYS_OUT_HOST="${OUTPUT}/isaac_nsys/${RUN_TAG}"
mkdir -p "${OUTPUT}" "${CACHE}"/{kit,ov,glcache,computecache} "${ASSETS}" "${LOGS}" "${ISAAC_NSYS_OUT_HOST}"
chmod -R 777 "${OUTPUT}" 2>/dev/null || true

IL="${HERE}/IsaacLab"
CFG_HOST="${IL}/source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config"
CFG_CTR="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config"
ASSETS_PY_HOST="${IL}/source/isaaclab/isaaclab/utils/assets.py"
ASSETS_PY_CTR="/workspace/isaaclab/source/isaaclab/isaaclab/utils/assets.py"
RLINF_HOST="${HERE}/RLinf/rlinf"
RLINF_CTR="/isaac-sim/kit/python/lib/python3.12/site-packages/rlinf"
RLLOG_CTR="/workspace/isaaclab/scripts/reinforcement_learning/rlinf/logs"
ISAAC_NSYS_OUT_CTR="/workspace/isaaclab/output/isaac_nsys/${RUN_TAG}"

CNAME="trocar_nsys_isaac_childprof"
$DK rm -f "${CNAME}" 2>/dev/null || true
$DK run -d --name "${CNAME}" --gpus "${GPUS:-all}" --network host \
  --entrypoint bash --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add=SYS_ADMIN \
  -e OMNI_KIT_ACCEPT_EULA=yes -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ALLOW_ROOT=1 \
  -e HF_HUB_OFFLINE=1 \
  -e RAY_enable_worker_prestart=0 \
  -e RAY_worker_lease_timeout_milliseconds=120000 \
  -e RLINF_ISAAC_NSYS=1 \
  -e RLINF_ISAAC_NSYS_RANKS=0,1 \
  -e RLINF_ISAAC_NSYS_OUT="${ISAAC_NSYS_OUT_CTR}" \
  -e RLINF_ISAAC_NSYS_TRACE="${ISAAC_NSYS_TRACE:-cuda,cudnn,cublas,nvtx,vulkan}" \
  -e RLINF_ISAAC_NSYS_CAPTURE="${ISAAC_NSYS_CAPTURE:-cudaProfilerApi}" \
  -e RLINF_ISAAC_NSYS_DURATION="${ISAAC_NSYS_DURATION:-}" \
  -e RLINF_ISAAC_KIT_NVTX="${ISAAC_KIT_NVTX:-0}" \
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
    echo '=== Design B: Isaac-child self-nsys (steps [3,4]) ==='; nsys --version | head -1; date
    cd /workspace/isaaclab && ./isaaclab.sh train --rl_library rlinf \
      --config_name ${CONFIG_NAME} \
      --model_path /models/Assemble_Trocar --num_envs ${NUM_ENVS:-64} --max_epochs ${MAX_EPOCHS}
    echo '=== RUN DONE ==='; date
    echo '=== gathering ALL worker reports into this run dir ==='
    LATEST=\$(ls -dt ${RLLOG_CTR}/rlinf/*/ 2>/dev/null | head -1)
    cp \"\${LATEST}\"trocar_prof_bench/nsights/rlinf_nsight_*.nsys-rep ${ISAAC_NSYS_OUT_CTR}/ 2>/dev/null || true
    echo '=== all nsys reports for this run: ==='; ls -la ${ISAAC_NSYS_OUT_CTR}/*.nsys-rep 2>&1
  "
echo "launched ${CNAME}: $?"
