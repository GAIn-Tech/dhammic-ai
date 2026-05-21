"""Fused Triton kernels for the dhammic-ai pretrain stack."""

from .rmsnorm import FusedRMSNorm, FusedRMSNormModule, fused_rmsnorm
from .inproj_split import FusedInProjSplit, FusedInProjSplitModule, fused_inproj_split
from .rope import FusedRoPE, fused_rope
from .trapezoidal import FusedTrapezoidal, fused_trapezoidal
from .ssd_scan import FusedSSDScan, fused_ssd_scan
from .swiglu import FusedSwiGLUAct, FusedSwiGLUMLP, fused_swiglu_act
from .sdr_topk import FusedSDRTopK, FusedSDRTopKModule, fused_sdr_topk
from .engram_gather import FusedEngramGatherModule, fused_engram_gather
from .lmhead_xent import FusedLMHeadXent, FusedLMHeadXentModule, fused_lmhead_xent

__all__ = [
    "FusedRMSNorm", "FusedRMSNormModule", "fused_rmsnorm",
    "FusedInProjSplit", "FusedInProjSplitModule", "fused_inproj_split",
    "FusedRoPE", "fused_rope",
    "FusedTrapezoidal", "fused_trapezoidal",
    "FusedSSDScan", "fused_ssd_scan",
    "FusedSwiGLUAct", "FusedSwiGLUMLP", "fused_swiglu_act",
    "FusedSDRTopK", "FusedSDRTopKModule", "fused_sdr_topk",
    "FusedEngramGatherModule", "fused_engram_gather",
    "FusedLMHeadXent", "FusedLMHeadXentModule", "fused_lmhead_xent",
]
