%%writefile src/quantization/rniq/utils/model_helper.py
import torch
from torch import nn
from src.aux.types import QScheme
from src.quantization.rniq.layers.rniq_conv2d import NoisyConv2d
from src.quantization.rniq.layers.rniq_linear import NoisyLinear
from src.quantization.rniq.layers.rniq_act import NoisyAct

class ModelHelper:
    @staticmethod
    def get_model_values(model: nn.Module, qscheme: QScheme = QScheme.PER_TENSOR):
        log_b_s, log_wght_s, log_w_n_b, log_act_q, log_act_s = [], [], [], [], []

        def collect_log_weights(module):
            if module.log_wght_s.requires_grad:
                if qscheme == QScheme.PER_CHANNEL:
                    log_wght_s.append(module.log_wght_s.ravel())
                    min = module.weight.amin((1, 2, 3))
                    max = module.weight.amax((1, 2, 3))
                    if getattr(module, "bias", None) is not None:
                        min_b = module.bias.amin()
                        max_b = module.bias.amax()
                    else:
                        min_b = torch.tensor([0.0], device=min.device)
                        max_b = torch.tensor([0.0], device=max.device)
                elif qscheme == QScheme.PER_TENSOR:
                    log_wght_s.append(module.log_wght_s)
                    log_wght_s.append(module.log_b_s)
                    min = module.weight.amin()
                    max = module.weight.amax()
                    min_b = module.bias.amin()
                    max_b = module.bias.amax()

                log_w_n_b.append(torch.log2(max - min + torch.exp2(module.log_wght_s.ravel())))
                if hasattr(module, "log_b_s"):
                    log_wght_s.append(module.log_b_s)
                    log_w_n_b.append(torch.log2(max_b - min_b + torch.exp2(module.log_b_s)))

        def collect_log_activations(module):
            if module.log_act_s.requires_grad:
                log_act_q.append(module.log_act_q)
                log_act_s.append(module.log_act_s)

        for _, module in model.named_modules():
            if isinstance(module, (NoisyConv2d, NoisyLinear)):
                collect_log_weights(module)
            elif isinstance(module, NoisyAct):
                collect_log_activations(module)

        if qscheme == QScheme.PER_TENSOR:
            return (
                torch.stack(log_act_s).ravel(),
                torch.stack(log_act_q).ravel(),
                torch.stack(log_wght_s).ravel(),
                torch.stack(log_w_n_b).ravel(),
            )
        else:
            return (
                torch.cat(log_act_s),
                torch.cat(log_act_q),
                torch.cat(log_wght_s),
                torch.cat(log_w_n_b),
            )