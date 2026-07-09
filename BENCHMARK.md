# Trocar RLinf PPO+GR00T — One-Click Production Throughput Benchmark

Run the 8-GPU `assemble_trocar` RLinf PPO+GR00T production benchmark from a
single prebuilt Docker image. The production config and the concurrent
asset-download fix are **baked in**, so you only need to pull the image, grab the
GR00T checkpoint, and run one command — no IsaacLab clone, no source mounts.

## Prerequisites

- **8× NVIDIA Hopper GPUs** (validated on 8× H20, 97 GB). Fewer GPUs also work if
  the divisibility rules below hold (4 GPUs is the other tested point).
- **Docker** with the NVIDIA container runtime (`--gpus`).
- **~120 GB disk** (image ~34 GB + GR00T checkpoint ~5 GB + asset cache/logs).
- `hf` (HuggingFace CLI) to download the checkpoint: `pip install -U huggingface_hub`.

## 1. Pull the image

```bash
docker pull chenchaox72877/trocar-rlinf-bench:latest
```

## 2. Get the GR00T checkpoint

```bash
mkdir -p ~/trocar-bench && cd ~/trocar-bench
hf download --repo-type model nvidia/Assemble_Trocar --local-dir Assemble_Trocar
```

## 3. Run the benchmark (8 GPU)

```bash
cd ~/trocar-bench
mkdir -p output assets-cache
docker run --rm --name trocar_bench \
  --gpus '"device=0,1,2,3,4,5,6,7"' --network host \
  --entrypoint bash --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
  -e OMNI_KIT_ACCEPT_EULA=yes -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e OMNI_KIT_ALLOW_ROOT=1 -e HF_HUB_OFFLINE=1 \
  -v "$PWD/Assemble_Trocar:/models/Assemble_Trocar:ro" \
  -v "$PWD/output:/workspace/isaaclab/output" \
  -v "$PWD/assets-cache:/tmp/Assets" \
  chenchaox72877/trocar-rlinf-bench:latest -lc "
    cd /workspace/isaaclab && ./isaaclab.sh train --rl_library rlinf \
      --config_name isaaclab_ppo_gr00t_assemble_trocar_prod \
      --model_path /models/Assemble_Trocar --num_envs 64 --max_epochs 3 \
      2>&1 | tee output/bench_\$(date +%Y%m%d_%H%M%S).log"
```

Notes:
- `--entrypoint bash` is required — the image's default entrypoint is Isaac Sim's
  `runheadless.sh`, which would otherwise swallow the command.
- The `-v .../assets-cache:/tmp/Assets` mount is **optional** — it persists the
  S3 healthcare assets across runs so subsequent starts skip the ~150 MB download.
  The baked atomic-download fix makes the first cold run race-safe either way.
- `--max_epochs 3` runs 3 RL steps (enough for a stable throughput reading). The
  first startup takes ~5–8 min (model load + 8 Isaac Sim workers); each RL step is
  ~48 min at production scale.

## Expected results (8× H20, 64 envs)

Per-RL-step timing is very stable (measured variance 0.26% across 3 steps):

| Phase | Mean | Share |
|---|---:|---:|
| **Step Time** | **~2875 s** (~47.9 min) | 100% |
| actor/run_training (FSDP) | ~1731 s | ~60% |
| env/env_interact_step (sim) | ~686 s | ~24% |
| rollout generate (wall) | ~1025 s | ~36% |

Throughput ≈ **22.8 env-steps/s** aggregate (128 × 8 × 64 = 65,536 env-steps/step).
GPU memory ≈ 27 / 97 GB per GPU. **Training (FSDP) is the dominant cost (~60%).**

Look for lines like this in the log (one per RL step):
```
Step Time: 2875.3s    actor/run_training=1731.1    env/env_interact_step=685.9
```

## GPU count / divisibility

RLinf asserts, for `world_size = N` GPUs:
- `total_num_envs % N == 0`
- `global_batch_size % (micro_batch_size × N) == 0`

The baked config is `total_num_envs=64, micro_batch_size=128, global_batch_size=2048`,
so **N ∈ {1, 2, 4, 8}**. Change the GPU set with `--gpus '"device=0,1,2,3"'` and
`--num_envs` (must stay a multiple of N and divide evenly, e.g. 32 on 4 GPUs).

## Notes / deviations

- Cameras render at **224×224** (stock), vs the 640×480→224 of the original
  production pipeline. The GR00T model runs on 224 either way, so this only
  affects the sim-render slice (~24%); the training-dominated headline is
  unaffected. Re-render at 640×480 only if you need exact sim-render fidelity.
- No external throughput reference was available; the table above is our baseline
  measured on this machine.

---
Image source / build recipe: `chenchaoxu7575/trocar-rlinf-benchmark`
(`docker/Dockerfile.trocar-bench` = `trocar-rlinf:latest` + baked config + assets fix).
