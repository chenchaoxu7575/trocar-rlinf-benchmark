# Trocar Production Config — nsys Bottleneck Profile

Kernel- and phase-level bottleneck breakdown of the 8×H20 production config,
from a Nsight Systems capture of one steady-state RL step (step 2, cudaProfiler-
gated). Complements the wall-clock throughput table in
[`prod_benchmark_results.md`](prod_benchmark_results.md).

## Method

- Image `trocar-rlinf-prof-gate2` (NVTX regions + RL-step `cudaProfilerStart/Stop`
  gate baked in); `scripts/run_nsys_prod.sh`.
- Config `isaaclab_ppo_gr00t_assemble_trocar_prof.yaml`: **production op shapes
  preserved** (`total_num_envs=64`, `micro_batch_size=128`) with a **shortened
  rollout** (`rollout_epoch=1`, `max_steps_per_rollout_epoch=16`,
  `global_batch=1024`) so one RL step is ~2 min and the report stays ~97 MB.
- `nsys profile --capture-range=cudaProfilerApi`, gate `PROF_GATE_START=2 NSTEPS=1`.
- Report contains all 8 rank processes (kernel instance counts are multiples of 8).

> **Note on the perf-sol skill.** `rlinf_vla_perf-sol` (SoL reconciliation) is
> scoped to **AsyncEmbodiedRunner** + RLinf's built-in per-worker `nsight_profiler`
> layout and its staleness / weight-sync metrics. This production run is
> **colocate + synchronous** (no weight_syncer), profiled as one nsys-wrapped
> process tree, so the skill finds 0 workers / 0 async symbols. The breakdown
> below is extracted directly from `nsys stats` instead.

## Phase breakdown (NVTX PushPop wall, 8 ranks summed)

| NVTX region | Share | Meaning |
|---|---:|---|
| `:run_training` | 38.1% | FSDP training (fwd+bwd+opt) |
| `:interact` | 21.6% | env interaction (sim rollout) |
| `:generate_one_epoch` | 16.0% | rollout generation (policy inference) |
| `:env_interact_step` | 9.7% | per-step env interaction (⊂ :interact) |
| `:isaaclab.sim_step` | 9.0% | Isaac Sim physics+render (⊂ :interact) |
| `:predict` | 5.6% | policy prediction (⊂ :generate) |

Regions nest, so they do not sum to 100%. Training is the largest phase here even
with the shortened rollout; at full production rollout the throughput run measured
**run_training ≈ 60% of the RL step**.

## GPU kernel breakdown (cuda_gpu_kern_sum, 8 ranks summed)

| Category | %GPU | Time (s) |
|---|---:|---:|
| **GEMM** (nvjet / cuBLASLt) | **60.3%** | 138.3 |
| **NCCL comm** (AllGather / AllReduce / ReduceScatter) | **13.4%** | 30.8 |
| elementwise / copy / memset | 10.7% | 24.4 |
| other | 6.3% | 14.5 |
| PhysX sim (articulation / TGS solver) | 5.7% | 13.0 |
| attention (fmha cutlass, bf16) | 3.6% | 8.4 |

Top individual kernels: `nvjet_tst_192x192...` (14%), `nvjet_tst_128x256...` (7%),
`ncclDevKernel_AllGather_RING_LL` (6.8%), `ncclDevKernel_AllReduce_Sum_f32` (6.1%),
`fmha_cutlassF_bf16` (3.6%), `artiSolveInternalConstraintsTGS1T` (2.8%).

## Optimization levers (ranked)

1. **GEMM-bound (60%)** — the core GR00T matmuls in bf16. Levers, roughly in
   order of expected payoff:
   - **fp8 / lower-precision GEMM** on Hopper (H20 has fp8 tensor cores) for the
     action-model / backbone matmuls — the single biggest lever.
   - **`torch.compile`** on the training forward (fuses GEMM+bias+activation +
     the elementwise/copy 10.7% below). A GR00T DiT `torch.compile` override
     already exists (`patches/gr00t_compile_patch.py`) but only covers the DiT;
     extend to the training path.
   - Larger effective batch — memory is only ~27/97 GB, so there is headroom to
     grow `micro_batch_size` and improve GEMM tile efficiency.
2. **NCCL comm (13%)** — FSDP AllGather(params)+AllReduce/ReduceScatter(grads).
   The config currently has `forward_prefetch: False`, `backward_prefetch: null`,
   `limit_all_gathers: False`. Levers:
   - **Enable forward/backward prefetch** to overlap the param AllGather with
     compute (cheap config change, direct hit on the exposed comm).
   - Hybrid sharding (`fsdp_size` < world) / HSDP to keep AllGather intra-smaller-group.
   - Confirm NVLink is used intra-node (RING_LL suggests it may be falling back).
3. **elementwise/copy/memset (11%)** — dtype casts and tensor copies; largely
   fused away by `torch.compile` (see 1).
4. Sim (PhysX 5.7%) and attention (3.6%) are minor — not worth optimizing before
   the above.

## Artifacts

- `output/prod_nsys.nsys-rep` (97 MB), `output/kern_sum.csv`.
- Regenerate: `GPUS='"device=0,1,2,3,4,5,6,7"' bash scripts/run_nsys_prod.sh`.
