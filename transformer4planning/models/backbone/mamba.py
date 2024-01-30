from transformer4planning.models.backbone.str_base import STR, STRConfig
import torch.nn as nn

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from mamba_ssm.models.mixer_seq_simple import create_block

class STRMambaConfig(STRConfig):
    pass

class STRMamba(STR):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # initialize mamba block
        factory_kwargs = {"device": 'cuda', "dtype": None}
        self.residual_in_fp32 = kwargs.get('residual_in_fp32', False)
        self.fused_add_norm = kwargs.get('fused_add_norm', False)
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
        d_model = config.n_embd
        rms_norm = kwargs.get('rms_norm', False)
        norm_epsilon = kwargs.get('norm_epsilon', 1e-5)
        self.transformer = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=kwargs.get('ssm_cfg', None),
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=kwargs.get('residual_in_fp32', False),
                    fused_add_norm=kwargs.get('fused_add_norm', False),
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(config.n_layer)
            ]
        )
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )
        initializer_cfg = kwargs.get("initializer_cfg", None)

    def embedding_to_hidden(self, input_embeds, attention_mask=None, position_ids=None, return_dict=True):
        # mamba forward
        residual = None
        for layer in self.transformer:
            input_embeds, residual = layer(
                input_embeds, residual, inference_params=None
            )
        if not self.fused_add_norm:
            residual = (input_embeds + residual) if residual is not None else input_embeds
            input_embeds = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            input_embeds = fused_add_norm_fn(
                input_embeds,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        # end of mamba forward
        transformer_outputs_hidden_state = input_embeds  # batch_size, seq_len (125 default), hidden_size
        return transformer_outputs_hidden_state
