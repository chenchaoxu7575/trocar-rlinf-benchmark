#!/usr/bin/env python3
"""Make GR00T's Eagle2.5 LLM backbone (Qwen2, head_dim=128) use flash_attention_2
instead of the forced sdpa. Vision (SigLIP, head_dim=72) stays on sdpa because
flash-attn has no head_dim=72 kernel.

Anchor: modeling_eagle2_5_vl.py forces `config.text_config._attn_implementation
= "sdpa"` in the Qwen2 branch. We flip only the text/LLM one (vision uses
vision_config, left untouched).
"""
import sys
from pathlib import Path

F = Path("/workspace/isaaclab/Isaac-GR00T/gr00t/model/backbone/eagle2_hg_model/modeling_eagle2_5_vl.py")
txt = F.read_text()
old = '                config.text_config._attn_implementation = "sdpa"'
new = '                config.text_config._attn_implementation = "flash_attention_2"'
if new in txt:
    print("[gr00t_fa_patch] already patched")
elif old in txt:
    assert txt.count(old) == 1, f"expected 1 text sdpa site, found {txt.count(old)}"
    F.write_text(txt.replace(old, new))
    print("[gr00t_fa_patch] LLM -> flash_attention_2 (vision stays sdpa)")
else:
    sys.exit("[gr00t_fa_patch] ERROR: text_config sdpa anchor not found")
