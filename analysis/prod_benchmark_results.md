# Trocar RLinf PPO+GR00T — Production-scale Throughput Benchmark

Reproduction of mingxue's production config on **8× H20 (single node)**.
Config: `isaaclab_ppo_gr00t_assemble_trocar_prod.yaml`
(total_num_envs=64, max_steps_per_rollout_epoch=128, rollout_epoch=8,
micro_batch_size=128, global_batch_size=2048; 8 env/GPU, grad-accum=2).
Camera: 224×224 (stock; deviation from prod 640×480 — see notes).
Run: `GPUS='"device=0..7"' bash run_trocar_prod.sh train --num_envs 64 --max_epochs 3`

## Per-RL-step timing (3 steps, very stable — 0.26% variance)

| Phase | Step 1 | Step 2 | Step 3 | Mean | Share |
|---|---:|---:|---:|---:|---:|
| **Step Time** | 2873.2 | 2869.3 | 2883.4 | **2875.3 s** (~47.9 min) | 100% |
| actor/run_training (FSDP) | 1742.9 | 1727.1 | 1723.4 | **1731.1 s** | **60.2%** |
| env/env_interact_step (sim) | 676.8 | 691.3 | 689.7 | **685.9 s** | 23.9% |
| rollout generate (wall) | ~1040 | ~1025 | ~1025 | **~1025 s** | ~35.7% |
| cal_adv_and_returns | 0.026 | 0.017 | 0.022 | ~0.02 s | ~0% |

Phase decomposition of one ~2875 s step:
- **Training (FSDP backward): 1731 s (60%)** — dominant bottleneck.
- Rollout generation: ~1025 s (36%) = sim env-step 686 s (24%) + GR00T policy inference ~340 s (12%).
- Sync / overhead (checkpoint excluded via save_interval=100): ~4%.

## Throughput

- Env-steps per RL step = 128 × 8 × 64 = **65,536**.
- Aggregate ≈ **22.8 env-steps/s**; during rollout ≈ **64 env-steps/s**.
- GPU memory ≈ **27 / 97 GB per GPU** — micro_batch=128 fits comfortably; headroom for larger batch.
- Losses stable/sane across steps (policy_loss ≈ -0.004, grad_norm ≈ 12).

## Findings

1. **train (FSDP) is the bottleneck at production scale** (60%), consistent with the
   earlier single-GPU profiling. Optimization should target FSDP training first.
2. **8-GPU startup no longer crashes** — the blocker was a concurrent USD asset
   download race (not kit/shader cache): IsaacLab `retrieve_file_path()` downloaded
   S3 healthcare assets to a shared `/tmp/Assets` with an unlocked check-then-use, so
   concurrent env workers read partial `.usd` files
   (`pxr.Tf.ErrorException: Failed reading 88 bytes at offset 0`, SurgicalTray001.usd).
   Non-deterministic: sometimes fatal (memory's "8-GPU crash"), sometimes a recoverable
   warning (this run). Fixed by an atomic download (temp file + `os.replace`) in
   `assets.py`, plus a persistent pre-warmable `/tmp/Assets` mount for cache reuse.

## Notes / deviations

- **Camera 224 vs prod 640×480**: kept stock 224. The GR00T model runs on 224 either
  way (rollout + train identical), so camera resolution only affects the sim-render
  slice (~24%); 640×480 also couples to GR00T `VideoToTensor` modality-meta and was
  deemed not worth the risk for a first repro. Quantify the delta with a 640×480 run if
  exact fidelity is needed.
- No colleague reference throughput number was available; this is our baseline.
