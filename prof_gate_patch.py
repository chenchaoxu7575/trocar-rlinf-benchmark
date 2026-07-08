#!/usr/bin/env python3
"""Step-gate nsys capture to env_interact_step steps [3,6] (skip warmup step 0-2).
Calls cudaProfilerStart at step 3 and cudaProfilerStop after step 6; run nsys with
--capture-range=cudaProfilerApi --capture-range-end=stop so only that window is
recorded -> tiny, openable .nsys-rep.
"""
import sys
from pathlib import Path

F = Path("/isaac-sim/kit/python/lib/python3.12/site-packages/rlinf/workers/env/env_worker.py")
txt = F.read_text()

anchor = (
    '        This function is used to interact with the environment.\n'
    '        """\n'
    '        chunk_actions = prepare_actions(\n'
)
gate = (
    '        This function is used to interact with the environment.\n'
    '        """\n'
    '        import torch as _t\n'
    '        if not hasattr(self, "_prof_step"):\n'
    '            self._prof_step = 0\n'
    '        if self._prof_step == 3:\n'
    '            _t.cuda.synchronize(); _t.cuda.profiler.start()\n'
    '            print("[prof_gate] cudaProfilerStart at env step 3", flush=True)\n'
    '        elif self._prof_step == 7:\n'
    '            _t.cuda.synchronize(); _t.cuda.profiler.stop()\n'
    '            print("[prof_gate] cudaProfilerStop after env step 6", flush=True)\n'
    '        self._prof_step += 1\n'
    '        chunk_actions = prepare_actions(\n'
)

if "_prof_step" in txt:
    print("[prof_gate_patch] already patched")
elif anchor in txt:
    assert txt.count(anchor) == 1, f"anchor not unique: {txt.count(anchor)}"
    F.write_text(txt.replace(anchor, gate, 1))
    print("[prof_gate_patch] step-gate [3,6] added to env_interact_step")
else:
    sys.exit("[prof_gate_patch] ERROR: anchor not found")
