"""Streaming / chunked-offload runtime for the dhammic-ai pretrain stack."""

from .chunked_lmhead import ChunkedLMHeadXent

__all__ = ["ChunkedLMHeadXent"]
