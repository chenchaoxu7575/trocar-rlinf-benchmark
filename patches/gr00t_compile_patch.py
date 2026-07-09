#!/usr/bin/env python3
"""Give GR00T an enable_torch_compile() implementation (BasePolicy leaves it as
NotImplementedError). Mirrors cnn_policy / PR#968: compile the *bound forward
method* of the flow-matching DiT (self.action_head.model) — the module launched
once per denoise step, i.e. the CPU-launch-bound small-kernel hotspot.

Compiling a bound method (not wrapping the module object) keeps params and
state_dict keys intact, so actor->rollout weight sync is unaffected — no
_orig_mod / _convert_compiled_weight_names handling needed.

The rollout worker already calls hf_model.enable_torch_compile(mode) when
cfg.rollout.enable_torch_compile is true.
"""
import sys
from pathlib import Path

F = Path("/isaac-sim/kit/python/lib/python3.12/site-packages/rlinf/models/embodiment/gr00t/gr00t_action_model.py")
txt = F.read_text()

anchor = "    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):"
method = (
    "    def enable_torch_compile(self, mode: str = \"max-autotune-no-cudagraphs\"):\n"
    "        \"\"\"Compile the flow-matching DiT forward (launched per denoise step).\"\"\"\n"
    "        if getattr(self, \"torch_compile_enabled\", False):\n"
    "            return\n"
    "        import torch as _torch\n"
    "        self.action_head.model.forward = _torch.compile(\n"
    "            self.action_head.model.forward, mode=mode\n"
    "        )\n"
    "        self.torch_compile_enabled = True\n"
    "        print(f\"[gr00t] enable_torch_compile: DiT forward compiled (mode={mode})\")\n"
    "\n"
)

if "def enable_torch_compile" in txt:
    print("[gr00t_compile_patch] already patched")
elif anchor in txt:
    assert txt.count(anchor) == 1, f"anchor not unique: {txt.count(anchor)}"
    F.write_text(txt.replace(anchor, method + anchor, 1))
    print("[gr00t_compile_patch] added enable_torch_compile to GR00T_N1_5_ForRLActionPrediction")
else:
    sys.exit("[gr00t_compile_patch] ERROR: forward anchor not found")
