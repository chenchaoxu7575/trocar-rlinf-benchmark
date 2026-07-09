# Trocar RLinf PPO + GR00T on IsaacLab — Build & Run Guide

End-to-end setup for the **assemble_trocar** RLinf VLA post-training task
(upstream `isaac-sim/IsaacLab` develop + RLinf + Isaac-GR00T), packaged as a
portable Docker image so it can run multi-node / on other machines.

Reference doc this is based on:
`isaac-sim/IsaacLab` → `docs/source/experimental-features/rlinf_vla_posttraining.rst`
Task code: `source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/`

> **Why a custom build?** The rst pins `rlinf==0.2.0dev2` (Python ≤3.11) but
> latest IsaacLab develop runs on Isaac Sim 6.0.0 = **Python 3.12** — no PyPI
> flash-attn/rlinf wheel exists for that combo. This guide documents the
> working path (force-install pure-python rlinf, compile flash-attn from source).

---

## 0. Machine prerequisites

- **GPU**: NVIDIA Hopper (H20/H100, `sm_90`) — validated on 8× H20 (97 GB).
  Blackwell (`sm_120`) supported via a separate flash-attn wheel (see §5).
- **Docker** with NVIDIA runtime (`--gpus`). On this box the binary is
  `/home/chenchaox/project/rlinf_pub/env/bin/docker` (aliased `docker`).
- **NGC login** (`nvcr.io`) — needed to pull the Isaac Sim base image.
  `~/.docker/config.json` already has the auth entry.
- **Disk**: ~120 GB (isaac-sim 21 GB + base 32 GB + trocar 34 GB + build 43 GB).
- **compose v2 plugin**: installed at `~/.docker/cli-plugins/docker-compose`
  (v2.32.4). Needed to build the base image; the trocar layer uses the legacy
  builder (`DOCKER_BUILDKIT=0`) because `buildx` is not installed.

Component versions (pinned):

| Component | Version / commit |
|---|---|
| Isaac Sim base | `nvcr.io/nvidia/isaac-sim:6.0.0` (Python 3.12.13, torch 2.10.0+cu128) |
| IsaacLab | develop (built at `378dc59`, 2026-07-02) |
| RLinf | `0.2` PyPI wheel (pure-python `py3-none-any`) |
| transformers | `4.51.3` (force over base's 4.57.6) |
| Isaac-GR00T | `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` (N1.5) |
| flash-attn | `2.8.3` compiled from source (sm_90 + sm_120 wheels) |
| GR00T checkpoint | `hf download nvidia/Assemble_Trocar` |

---

## 1. Existing artifacts on this machine (skip the build if present)

Docker images:

| Image | Purpose |
|---|---|
| `nvcr.io/nvidia/isaac-sim:6.0.0` | NVIDIA base (pulled from NGC) |
| `isaac-lab-base:latest` | IsaacLab installed on the isaac-sim base |
| **`trocar-rlinf:latest`** | **Final runnable image** (rlinf+GR00T+trocar+flash-attn) |
| `trocar-build:latest` | trocar-rlinf + CUDA toolkit 12.8 — only for compiling flash-attn |

Repository layout (build context = repo root `~/project/ai4health/trocar-image/`):

```
├── docker/       # Dockerfiles (build with -f docker/<file> from repo root)
│   ├── Dockerfile.trocar              # the trocar layer -> trocar-rlinf:latest
│   ├── Dockerfile.trocar-prof         # + NVTX (profiling)
│   ├── Dockerfile.trocar-prof-gate{,2}# + step-gated nsys capture
│   └── Dockerfile.trocar-fa{,c}       # + GR00T flash-attn / torch.compile
├── scripts/      # launchers (resolve paths relative to repo root)
│   ├── run_trocar.sh                  # CI-scale smoke run
│   ├── run_trocar_prod.sh            # production 8-GPU benchmark
│   └── run_nsys.sh                    # nsys profiling run
├── patches/      # in-container code patches (COPY'd by the prof/fa/fac images)
│   ├── nvtx_patch.py  prof_gate_patch.py  prof_gate_rlstep_patch.py
│   └── gr00t_flashattn_patch.py  gr00t_compile_patch.py
├── analysis/     # nsys parsers + benchmark results
│   ├── osrt_breakdown.py  probe_processes.py  rlstep_timeline.py
│   └── prod_benchmark_results.md
└── README.md
```

Gitignored large/generated artifacts (regenerate via this guide):
- `IsaacLab/` — fresh develop clone (build context for the base image)
- `models/Assemble_Trocar/` — GR00T checkpoint (5.1 GB)
- `wheels/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl` — **sm_90** (H20)
- `wheels_sm120/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl` — **sm_120** (Blackwell)
  - ⚠️ same filename as sm_90 (arch is in the `.so`, not the name) — keep in
    separate dirs, don't mix.
- `cache/`, `output/` — Isaac asset/kit cache and run logs/profiles.

---

## 2. Build from scratch

### 2.1 Clone IsaacLab develop
```bash
cd ~/project/ai4health/trocar-image
git clone --depth 1 --branch develop https://github.com/isaac-sim/IsaacLab.git
```
Confirm the task + rlinf entrypoint exist:
```bash
ls IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/  # g129_dex3_env_cfg.py config mdp
ls IsaacLab/scripts/reinforcement_learning/rlinf/                          # train_rlinf.py play_rlinf.py
grep ISAACSIM_VERSION IsaacLab/docker/.env.base                            # -> 6.0.0
```

### 2.2 Pull the Isaac Sim base (NGC)
```bash
docker pull nvcr.io/nvidia/isaac-sim:6.0.0
```

### 2.3 Build `isaac-lab-base`
Uses IsaacLab's own docker recipe (handles EULA, apt deps, `isaaclab.sh --install`,
`_isaac_sim` symlink). Requires the compose plugin.
```bash
cd ~/project/ai4health/trocar-image/IsaacLab/docker
DOCKER_BUILDKIT=1 docker compose --env-file .env.base --profile base build isaac-lab-base
```
Produces `isaac-lab-base:latest` (ISAACLAB_PATH=`/workspace/isaaclab`, user home `/root`).

### 2.4 Download the GR00T checkpoint
```bash
cd ~/project/ai4health/trocar-image
hf download --repo-type model nvidia/Assemble_Trocar --local-dir models/Assemble_Trocar
```

### 2.5 Build `trocar-rlinf` (the rst layers)
`Dockerfile.trocar` = FROM isaac-lab-base + rst Steps 1–4, adapted for the
docker base (see §4 for *why* each line is the way it is). Build with the legacy
builder (no buildx):
```bash
cd ~/project/ai4health/trocar-image
DOCKER_BUILDKIT=0 docker build -f docker/Dockerfile.trocar -t trocar-rlinf:latest .
```
`.dockerignore` excludes `IsaacLab/ models/ cache/ output/ wheels_sm120/` so the
context is small and the only thing COPYed is the flash-attn wheel.

Verify:
```bash
docker run --rm --entrypoint bash -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=yes \
  -e OMNI_KIT_ALLOW_ROOT=1 trocar-rlinf:latest -lc \
  '/isaac-sim/kit/python/bin/python3 -c "import torch,flash_attn_2_cuda; \
     from flash_attn import flash_attn_func; print(flash_attn.__version__)"'
```

---

## 3. Run

Use `scripts/run_trocar.sh` (sets `--entrypoint bash`, mounts model + output, picks GPUs):
```bash
# 1-GPU smoke (default config divisibility holds at world_size=1)
GPUS='"device=0"' bash scripts/run_trocar.sh train --max_epochs 2

# eval a checkpoint
GPUS='"device=0"' bash scripts/run_trocar.sh play --max_epochs 2
```

### Multi-GPU (throughput)
On N GPUs RLinf asserts:
- `total_num_envs % N == 0`  → set `--num_envs` to a multiple of N
- `global_batch_size % (micro_batch_size × N) == 0` (micro_batch_size=2)

The config ships CI-scale (`total_num_envs=4`, `global_batch_size=4`). `--num_envs`
is a CLI flag; **`global_batch_size` is not**, so bump it in the config yaml first.
Working 4-GPU invocation (validated):
```bash
CFG=/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config/isaaclab_ppo_gr00t_assemble_trocar.yaml
docker run -d --name trocar_bench --gpus '"device=0,1,2,3"' --network host \
  --entrypoint bash --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
  -e OMNI_KIT_ACCEPT_EULA=yes -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ALLOW_ROOT=1 -e HF_HUB_OFFLINE=1 \
  -v $PWD/models/Assemble_Trocar:/models/Assemble_Trocar:ro \
  -v $PWD/output:/workspace/isaaclab/output \
  trocar-rlinf:latest -lc "
    sed -i 's/global_batch_size: 4/global_batch_size: 8/' $CFG   # N=4 -> need multiple of 8
    cd /workspace/isaaclab && ./isaaclab.sh train --rl_library rlinf \
      --config_name isaaclab_ppo_gr00t_assemble_trocar \
      --model_path /models/Assemble_Trocar --num_envs 4 --max_epochs 2"
# watch: docker logs -f trocar_bench   (headless, no tee — the mount is uid-mismatched)
```
For N=8 use `global_batch_size=16`, `--num_envs 8` — **but see §6 (8-GPU crashes)**.

---

## 4. Why each trocar-layer step is the way it is (the hard-won fixes)

All installs run **as root into the MAIN site-packages** (`OMNI_KIT_ALLOW_ROOT=1`)
— user-site installs (default `isaaclab` user, home `/root` unwritable) silently
don't take effect. `PY=/isaac-sim/kit/python/bin/python3`.

1. **`isaaclab.sh -i 'contrib[rlinf]'`** — rst Step 1 (ray/av/diffusers/timm/peft…).
2. **rlinf**: `pip install --ignore-requires-python --no-deps rlinf==0.2` — the
   wheel is pure-python (`py3-none-any`) so it runs on 3.12; the ≤3.11.14 cap is
   just conservative metadata. `--no-deps` avoids pulling `torch<=2.9` (would
   downgrade Isaac Sim's torch 2.10 and break it).
3. **transformers**: force `transformers==4.51.3 --no-deps --force-reinstall` —
   base ships 4.57.6 where `VideoInput` moved out of `transformers.image_utils`,
   which GR00T's Eagle2.5 processor imports. Also `pip install requests pyyaml`
   (the `--no-deps` pins leave the minimal kit python missing these; 4.51.3's
   import chain hits huggingface_hub → urllib3/idna/yaml).
4. **Isaac-GR00T** pinned `4af2b62`, `pip install -e ".[base]" --no-deps`.
5. **flash-attn**: install the prebuilt wheel we compiled (§5). Replaces the rst's
   `no_flash_attn.patch` fallback so GR00T uses real flash-attn.

Other runtime gotcha: `isaac-lab-base` ENTRYPOINT is `/isaac-sim/runheadless.sh`,
so every `docker run` needs **`--entrypoint bash`** or your command is swallowed
as args and the container hangs.

---

## 5. Compiling flash-attn from source

No PyPI wheel exists for flash-attn on torch 2.10 (checked 2.8.3→2.9.1). The
torch 2.9 wheel is **ABI-incompatible** with torch 2.10 (`undefined symbol:
_ZN3c104cuda29c10_cuda_check_implementation…`). So compile in-image against the
image's own torch (guarantees ABI match).

### 5.1 One-time: image with CUDA toolkit
```bash
# start a container, apt-install cuda-toolkit-12-8, commit it
docker run -d --name fb --entrypoint bash -e OMNI_KIT_ALLOW_ROOT=1 trocar-rlinf:latest -lc '
  apt-get update -qq && apt-get install -y -qq wget
  cd /tmp && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update -qq && apt-get install -y -qq cuda-toolkit-12-8
  /isaac-sim/kit/python/bin/python3 -m pip install ninja packaging psutil
  sleep infinity'
docker commit fb trocar-build:latest && docker rm -f fb
```

### 5.2 THE critical gotcha — ninja must be on PATH
The `ninja` binary is at `/isaac-sim/kit/python/bin/ninja` but **not on PATH**, so
`torch.utils.cpp_extension.is_ninja_available()` is False → torch compiles
**serially** (1 giant Hopper kernel at a time, ~1 obj / 5 min, ETA *hours*).
Put it on PATH → ~378 parallel procs, ~6 obj/min, **wheel in ~13 min**.
> Whenever a CUDA-extension build is mysteriously slow, check `is_ninja_available()`.

### 5.3 Build the wheel
```bash
# sm_90 (H20).  For Blackwell set FLASH_ATTN_CUDA_ARCHS=120 and output to wheels_sm120/
docker run -d --name fbuild --entrypoint bash \
  -e OMNI_KIT_ALLOW_ROOT=1 -e CUDA_HOME=/usr/local/cuda-12.8 \
  -e MAX_JOBS=64 -e NVCC_THREADS=2 -e FLASH_ATTN_CUDA_ARCHS=90 \
  -e FLASH_ATTENTION_DISABLE_FP8=TRUE \
  -v $PWD/wheels:/out trocar-build:latest -lc '
    export PATH=/isaac-sim/kit/python/bin:/usr/local/cuda-12.8/bin:$PATH   # <-- ninja!
    /isaac-sim/kit/python/bin/python3 -m pip wheel flash-attn==2.8.3 \
      --no-build-isolation --no-deps -w /out'
```
- Arch knob is **`FLASH_ATTN_CUDA_ARCHS`** (`"90"`, `"120"`, or default `"80;90;100;120"`).
- Keep `sm_80` — those are the FA2 kernels GR00T's `flash_attn_func` actually uses
  (built with sm_90 gencode too). The `DISABLE_HDIM*/LOCAL/…` flags did NOT cut
  build time here — ninja parallelism was the real lever.
- Verify: `import torch, flash_attn_2_cuda` (torch first!), or
  `cuobjdump <.so> | grep sm_` to confirm the target arch is baked in.

sm_90 → `wheels/`, sm_120 → `wheels_sm120/` (validated: `cuobjdump` shows `sm_120`).

---

## 6. Known issues / gotchas

- **8-GPU startup crashes.** Running 8 Isaac Sim env workers in one container makes
  them race on shared kit/USD caches → most/all crash at startup (breakpad `[Fatal]`,
  USD `TfBaseException`). Non-deterministic (1 rank one run, 7 the next). **4 GPUs
  is stable.** To use 8, isolate per-rank Isaac Sim cache dirs (open item).
- **flash-attn on 3.12 / torch 2.10** — no prebuilt wheel; must compile (§5).
- **Config is CI-scale** — scale `total_num_envs` / `global_batch_size` for real
  throughput; mind the two divisibility asserts.
- **tee log mount** — the container writes as a different uid; `output/` bind mount
  may not receive the tee'd log. Use `docker logs <container>` instead.
- **Isaac Sim py3.12 kit python** is minimal — expect to add small pure-python deps
  (`requests`, `pyyaml`, `urllib3`…) when a `--no-deps` install exposes a gap.

---

## 7. Throughput baseline (4× H20, flash-attn, 4 envs / 8 trajectories, max_epochs=2)

| Metric | 1-GPU SDPA (no flash-attn) | **4-GPU + flash-attn** | speedup |
|---|---|---|---|
| Step Time | 2478 s | 1341 s | 1.85× |
| actor/run_training | 2013 s | 1030 s | 1.95× |
| generate_rollouts | 433 s | 292 s | 1.49× |
| env/interact | 428 s | 279 s | 1.54× |

⚠️ This changes **two** variables (GPU count *and* attention impl), so it is a
combined 4-GPU+flash-attn number, **not** flash-attn's isolated gain. For that,
run a controlled **4-GPU SDPA** comparison (apply `no_flash_attn.patch` or set
`attn_implementation`) against the 4-GPU flash-attn numbers above.

> **Correction (found via profiling, §9):** GR00T defaults to `sdpa`, so this
> "4-GPU + flash-attn" run was actually **still SDPA** — flash-attn was installed
> but not used. See §9 for how to actually enable it and why it barely matters
> for this workload.

---

## 8. Orchestration — how GPUs/workers are placed

Set by `cluster` in the config. trocar ships the simplest form: full colocate.

```yaml
cluster:
  num_nodes: 1
  component_placement:
      actor,env,rollout: all      # all 3 worker groups share ALL visible GPUs
```

Three RLinf worker groups, all colocated on the same GPUs (time-shared in the
synchronous PPO loop):

| Group | Role | key config |
|---|---|---|
| **ActorGroup** | FSDP training (fwd/bwd/optim) | `training_backend: fsdp`, `micro_batch_size: 2`, `global_batch_size: 4` |
| **RolloutGroup** | GR00T action generation | `mode: colocate`, `backend: huggingface`, `enable_offload: True` |
| **EnvGroup** | Isaac Sim env stepping | `total_num_envs: 4` |

- `all` = every visible GPU (`--gpus`/`CUDA_VISIBLE_DEVICES`). N GPUs → N ranks per group (`EnvGroup(rank=0..N-1)` etc.).
- Loop: RolloutGroup generates → EnvGroup steps sim → ActorGroup trains on the batch → weights synced rollout←actor (colocate uses cudaIPC).
- **Divisibility (asserts on N GPUs):** `total_num_envs % N == 0` **and** `global_batch_size % (micro_batch_size × N) == 0`. `--num_envs` is a CLI flag; `global_batch_size` must be edited in the yaml (e.g. N=4 → `8`, N=8 → `16`).

Other RLinf placement modes (usable by editing `component_placement`, not shipped by IsaacLab): per-component GPU ranges for **disaggregated** (`env: 0-3`, `rollout: 4-5`, `actor: 6-7` → rollout/train overlap, no offload, needs more GPUs); `node_groups` for **heterogeneous/multi-node** (e.g. real-robot node + GPU node). IsaacLab develop only ships trocar with the single colocate config.

---

## 9. Profiling & experiments (NVTX + Nsight Systems)

### Image lineage
```
isaac-lab-base → trocar-rlinf (§2, runnable, flash-attn wheel installed)
   └─ trocar-build       (+ CUDA toolkit 12.8 / nsys — used to compile flash-attn)
        └─ trocar-rlinf-prof  (+ NVTX: Worker.timer regions + isaaclab.sim_step; via nvtx_patch.py)
             └─ trocar-rlinf-fa   (+ GR00T LLM attn_implementation sdpa→flash_attention_2; gr00t_flashattn_patch.py)
                  └─ trocar-rlinf-fac  (+ GR00T DiT enable_torch_compile override; gr00t_compile_patch.py)
```

### NVTX instrumentation (self-added; RLinf 0.2 has none)
`nvtx_patch.py` (baked into `trocar-rlinf-prof`+):
- patches RLinf `Worker.timer` decorator → every timed region becomes an NVTX range (`:interact`, `:env_interact_step`, `:run_training`, `:generate_one_epoch`, `:predict`).
- wraps the Isaac Sim per-step call in `isaaclab_env.chunk_step` → `isaaclab.sim_step` NVTX (physics+render). Uses `torch.cuda.nvtx` (no extra pkg).

### nsys run
`scripts/run_nsys.sh` → `nsys profile -t cuda,nvtx,osrt,vulkan --cuda-graph-trace=node` (Vulkan traces Isaac Sim RTX). Ray worker child processes ARE captured automatically. Verify: `nsys stats --report nvtx_sum,cuda_gpu_kern_sum <rep>`.

### The 3 reports (in `output/`, same config: 1×H20 / num_envs=4 / rollout 16 steps / max_epochs=1)
| report | image | what it adds |
|---|---|---|
| `trocar_nsys.nsys-rep` (826 MB) | trocar-rlinf-prof | SDPA baseline; attention = `fmha_cutlassF_...sm80` |
| `trocar_nsys_fa.nsys-rep` (849 MB) | trocar-rlinf-fa | LLM flash-attn; `flash_fwd_*`/`flash_bwd_*` kernels appear (vision stays SDPA, head_dim 72) |
| `trocar_nsys_fac.nsys-rep` (890 MB) | trocar-rlinf-fac | + DiT `torch.compile(default)`; `triton_*_fused_*` kernels appear |

Config used (runtime seds over the yaml): `max_steps_per_rollout_epoch: 256→16`; `_fac` also adds `rollout.enable_torch_compile: True` + `torch_compile_mode: "default"`. CLI: `./isaaclab.sh train --rl_library rlinf --config_name isaaclab_ppo_gr00t_assemble_trocar --model_path /models/Assemble_Trocar --num_envs 4 --max_epochs 1`.

### Findings
1. **flash-attn**: installed but **GR00T defaults to `sdpa`** — must set the Eagle2.5 LLM (Qwen2, head_dim 128) to `flash_attention_2` (vision head_dim 72 has no flash kernel, stays sdpa). Once enabled, `flash_fwd`/`flash_bwd` kernels confirmed. **But no measurable end-to-end gain** — GR00T sequences are short (`max_prompt_length: 30`), where flash-attn ≈ SDPA-memeff, and attention is a small slice of `run_training`.
2. **GR00T torch.compile** (new `enable_torch_compile` override — RLinf's BasePolicy left it NotImplementedError; mirrors cnn_policy/PR#968): compiles the flow-matching DiT forward. Confirmed working (fused `triton_poi_fused_addmm_gelu_view`, `triton_red_fused_..._layer_norm_...` kernels; no errors). **Minimal end-to-end gain** — the DiT denoise is a small fraction of the workload (dominated by the Eagle2.5 VLM backbone + FSDP training); total kernel count not reduced.
3. **Open**: the "denoise = too many small-kernel CPU launches" premise is **not yet quantified** (need to measure CPU-launch gaps in the denoise region). Bigger denoise cost is likely in the **actor training** path (recomputes the denoise chain for logprob), which the rollout-only `enable_torch_compile` flag does not touch. Next: quantify launch-bound fraction, then extend compile to the actor path and/or try `mode=reduce-overhead` (CUDA graphs).

Patch scripts live in `patches/` (`nvtx_patch.py`, `gr00t_flashattn_patch.py`,
`gr00t_compile_patch.py`) and the profiling Dockerfiles in `docker/`
(`Dockerfile.trocar-prof/-fa/-fac`). Build them from the repo root so the
`COPY patches/...` lines resolve, e.g.
`DOCKER_BUILDKIT=0 docker build -f docker/Dockerfile.trocar-prof -t trocar-rlinf-prof:latest .`
