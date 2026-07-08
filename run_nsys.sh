#!/usr/bin/env bash
# Profile a short trocar run under Nsight Systems with graphics (Vulkan/RTX) +
# CUDA + NVTX + OS-runtime tracing. Produces output/trocar_nsys.nsys-rep.
#
# NVTX ranges come from the trocar-rlinf-prof image (Worker.timer regions +
# isaaclab.sim_step). Isaac Sim RTX rendering is captured via the `vulkan` trace.
set -euo pipefail
DK=/home/chenchaox/project/rlinf_pub/env/bin/docker
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config/isaaclab_ppo_gr00t_assemble_trocar.yaml
CNAME=trocar_nsys
GPUS="${GPUS:-\"device=0\"}"
# nsys trace set. gpu-metrics is opt-in (needs driver perf-counter access); enable with GPU_METRICS=1
TRACE="cuda,nvtx,osrt,vulkan"
GPU_METRICS_ARG=""
[ "${GPU_METRICS:-0}" = "1" ] && GPU_METRICS_ARG="--gpu-metrics-devices=all"

mkdir -p "$HERE/output"; chmod 777 "$HERE/output"
$DK rm -f "$CNAME" 2>/dev/null || true
$DK run -d --name "$CNAME" --gpus "$GPUS" --network host \
  --entrypoint bash --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add=SYS_ADMIN \
  -e OMNI_KIT_ACCEPT_EULA=yes -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ALLOW_ROOT=1 -e HF_HUB_OFFLINE=1 \
  -v "$HERE/models/Assemble_Trocar:/models/Assemble_Trocar:ro" \
  -v "$HERE/output:/workspace/isaaclab/output" \
  trocar-rlinf-prof:latest -lc "
    set -e
    # keep the profile short: few sim steps + one train update
    sed -i 's/max_steps_per_rollout_epoch: 256/max_steps_per_rollout_epoch: 16/' $CFG
    echo '=== nsys profile (trace=$TRACE ${GPU_METRICS_ARG}) ==='; date
    cd /workspace/isaaclab
    nsys profile -t $TRACE $GPU_METRICS_ARG \
      --cuda-graph-trace=node --force-overwrite=true \
      --sample=none \
      -o /workspace/isaaclab/output/trocar_nsys \
      ./isaaclab.sh train --rl_library rlinf \
        --config_name isaaclab_ppo_gr00t_assemble_trocar \
        --model_path /models/Assemble_Trocar --num_envs 4 --max_epochs 1
    echo '=== NSYS DONE ==='; date; ls -la /workspace/isaaclab/output/*.nsys-rep
  "
echo "launched $CNAME: $?"
