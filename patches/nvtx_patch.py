#!/usr/bin/env python3
"""Add NVTX ranges to RLinf's Worker.timer decorator (broad pipeline coverage:
env_interact_step / interact / generate / train ...) and to the Isaac Sim
per-step call inside isaaclab_env.chunk_step (sim physics+render detail).

Uses torch.cuda.nvtx (bundled with torch) so no extra package is needed.
Idempotent-ish: asserts each expected substring is present exactly so the build
fails loudly if the upstream source drifts.
"""
import sys
from pathlib import Path

SP = Path("/isaac-sim/kit/python/lib/python3.12/site-packages/rlinf")

def patch(path: Path, old: str, new: str, label: str):
    txt = path.read_text()
    if new in txt:
        print(f"[nvtx_patch] {label}: already patched, skip")
        return
    n = txt.count(old)
    if n == 0:
        sys.exit(f"[nvtx_patch] ERROR {label}: anchor not found in {path}")
    path.write_text(txt.replace(old, new))
    print(f"[nvtx_patch] {label}: patched {n} site(s) in {path}")

# 1) Worker.timer -> also emit an NVTX range named after the timer tag.
#    The sync and async wrappers share the identical `with self.worker_timer(...)` line.
worker = SP / "scheduler/worker/worker.py"
patch(
    worker,
    "                with self.worker_timer(tag or func.__name__):",
    "                with torch.cuda.nvtx.range(tag or func.__name__), self.worker_timer(tag or func.__name__):",
    "Worker.timer NVTX",
)

# 2) isaaclab_env.chunk_step -> wrap the per-step Isaac Sim call (physics+render).
env = SP / "envs/isaaclab/isaaclab_env.py"
patch(
    env,
    "            extracted_obs, step_reward, terminations, truncations, infos = self.step(\n                actions, auto_reset=False\n            )",
    "            with torch.cuda.nvtx.range(\"isaaclab.sim_step\"):\n"
    "                extracted_obs, step_reward, terminations, truncations, infos = self.step(\n"
    "                    actions, auto_reset=False\n"
    "                )",
    "isaaclab sim_step NVTX",
)

print("[nvtx_patch] done")
