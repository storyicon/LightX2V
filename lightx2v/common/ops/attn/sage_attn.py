import torch
from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate

from sageattention import sageattn


@ATTN_WEIGHT_REGISTER("sage_attn2")
class SageAttn2Weight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if len(q.shape) == 3:
            bs = 1
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        elif len(q.shape) == 4:
            bs = q.shape[0]
        x = sageattn(
            q,
            k,
            v,
            tensor_layout="NHD",
        ).view(bs * max_seqlen_q, -1)
        return x


@ATTN_WEIGHT_REGISTER("sage_attn3")
class SageAttn3Weight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if len(q.shape) == 3:
            bs = 1
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        elif len(q.shape) == 4:
            bs = q.shape[0]

        x = sageattn3_blackwell(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).reshape(bs * max_seqlen_q, -1)
        return x
