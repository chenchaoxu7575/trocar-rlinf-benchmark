#!/usr/bin/env python3
"""Step-gate nsys capture at the RL-step granularity in EmbodiedRunner.run():
one colocated RL step = rollout (env.interact + rollout.generate) + train
(actor.run_training). Capture PROF_GATE_NSTEPS consecutive steady-state steps
starting at global_step==PROF_GATE_START (default: steps 2,3 — skips 0,1 warmup).

cudaProfilerStart at the top of step START; cudaProfilerStop right after
step (START+NSTEPS-1)'s training completes. Run nsys with
--capture-range=cudaProfilerApi --capture-range-end=stop-shutdown.
"""
import sys
from pathlib import Path

F = Path("/isaac-sim/kit/python/lib/python3.12/site-packages/rlinf/runners/embodied_runner.py")
txt = F.read_text()

start_old = (
    "        for _step in range(start_step, self.max_steps):\n"
    "            # set global step\n"
    "            self.actor.set_global_step(self.global_step)\n"
)
start_new = (
    "        for _step in range(start_step, self.max_steps):\n"
    "            import os as _os, torch as _pt\n"
    "            _gs = int(_os.environ.get(\"PROF_GATE_START\", \"2\"))\n"
    "            _gn = int(_os.environ.get(\"PROF_GATE_NSTEPS\", \"2\"))\n"
    "            if self.global_step == _gs:\n"
    "                _pt.cuda.synchronize(); _pt.cuda.profiler.start()\n"
    "                print(f\"[prof_gate] cudaProfilerStart at RL step {_gs}\", flush=True)\n"
    "            # set global step\n"
    "            self.actor.set_global_step(self.global_step)\n"
)

stop_old = (
    "                actor_training_metrics = actor_training_handle.wait()\n"
    "\n"
    "                self.global_step += 1\n"
)
stop_new = (
    "                actor_training_metrics = actor_training_handle.wait()\n"
    "\n"
    "                self.global_step += 1\n"
    "                if self.global_step == _gs + _gn:\n"
    "                    _pt.cuda.synchronize(); _pt.cuda.profiler.stop()\n"
    "                    print(f\"[prof_gate] cudaProfilerStop after RL step {_gs + _gn - 1}\", flush=True)\n"
)

if "prof_gate" in txt:
    print("[prof_gate_rlstep_patch] already patched")
else:
    for name, old, new in [("start", start_old, start_new), ("stop", stop_old, stop_new)]:
        if old not in txt:
            sys.exit(f"[prof_gate_rlstep_patch] ERROR: {name} anchor not found")
        assert txt.count(old) == 1, f"{name} anchor not unique: {txt.count(old)}"
        txt = txt.replace(old, new)
    F.write_text(txt)
    print("[prof_gate_rlstep_patch] RL-step gate added to EmbodiedRunner.run()")
