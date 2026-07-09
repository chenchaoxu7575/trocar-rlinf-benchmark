#!/usr/bin/env bash
# Nsight Systems profile of the PRODUCTION-shaped config (op shapes preserved:
# envs=64, micro_batch=128; rollout shortened via the _prof yaml so one RL step
# is short and the report stays manageable). Uses the trocar-rlinf-prof-gate2
# image (NVTX + RL-step cudaProfilerStart/Stop gate baked in) and mounts the
# host prod-benchmark files (prof yaml, atomic-download assets.py) over the baked
# copies. Captures PROF_GATE_NSTEPS steady-state step(s) starting at PROF_GATE_START.
#
# Usage:
#   GPUS='"device=0,1,2,3,4,5,6,7"' bash scripts/run_nsys_prod.sh
set -euo pipefail

DK=/home/chenchaox/project/rlinf_pub/env/bin/docker
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="trocar-rlinf-prof-gate2:latest"
CONFIG_NAME="isaaclab_ppo_gr00t_assemble_trocar_prof"

MODELS="${HERE}/models"; OUTPUT="${HERE}/output"
CACHE="${HERE}/cache/isaac-sim"; ASSETS="${HERE}/cache/isaac-assets"
mkdir -p "${OUTPUT}" "${CACHE}"/{kit,ov,pip,glcache,computecache,logs} "${ASSETS}"
chmod -R 777 "${OUTPUT}" 2>/dev/null || true

IL="${HERE}/IsaacLab"
CFG_HOST="${IL}/source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config"
CFG_CTR="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config"
ASSETS_PY_HOST="${IL}/source/isaaclab/isaaclab/utils/assets.py"
ASSETS_PY_CTR="/workspace/isaaclab/source/isaaclab/isaaclab/utils/assets.py"

TRACE="${TRACE:-cuda,nvtx,osrt,cudnn,cublas}"
CNAME="trocar_nsys_prod"
$DK rm -f "${CNAME}" 2>/dev/null || true
$DK run -d --name "${CNAME}" --gpus "${GPUS:-all}" --network host \
  --entrypoint bash --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add=SYS_ADMIN \
  -e OMNI_KIT_ACCEPT_EULA=yes -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ALLOW_ROOT=1 \
  -e HF_HUB_OFFLINE=1 -e PROF_GATE_START="${PROF_GATE_START:-2}" -e PROF_GATE_NSTEPS="${PROF_GATE_NSTEPS:-1}" \
  -v "${MODELS}/Assemble_Trocar:/models/Assemble_Trocar:ro" \
  -v "${ASSETS}:/tmp/Assets" \
  -v "${OUTPUT}:/workspace/isaaclab/output" \
  -v "${CFG_HOST}/${CONFIG_NAME}.yaml:${CFG_CTR}/${CONFIG_NAME}.yaml:ro" \
  -v "${ASSETS_PY_HOST}:${ASSETS_PY_CTR}:ro" \
  -v "${CACHE}/kit:/isaac-sim/kit/cache" \
  -v "${CACHE}/ov:/root/.cache/ov" \
  -v "${CACHE}/computecache:/root/.nv/ComputeCache" \
  -v "${CACHE}/glcache:/root/.cache/nvidia/GLCache" \
  "${IMAGE}" -lc "
    set -e
    echo '=== nsys prod profile (trace=${TRACE}, gate step ${PROF_GATE_START:-2} x ${PROF_GATE_NSTEPS:-1}) ==='; date
    cd /workspace/isaaclab
    nsys profile -t ${TRACE} --cuda-graph-trace=node --force-overwrite=true --sample=none \
      --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
      -o /workspace/isaaclab/output/prod_nsys \
      ./isaaclab.sh train --rl_library rlinf \
        --config_name ${CONFIG_NAME} \
        --model_path /models/Assemble_Trocar --num_envs 64 --max_epochs 3
    echo '=== NSYS DONE ==='; date; ls -la /workspace/isaaclab/output/prod_nsys.* 2>/dev/null
  "
echo "launched ${CNAME}: $?"
