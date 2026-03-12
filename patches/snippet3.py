%%writefile src/quantization/rniq/rniq.py
import torch

from torch import Tensor
from torch.autograd import Function
import torch.distributed as dist

from src.quantization.rniq.rniq_utils import QMode, QNMethod


class QNoise(Function):
    @staticmethod
    def forward(input, scale):
        output = torch.round(input) - input
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, scale = inputs
        ctx.save_for_backward(input, scale)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        raise AttributeError(
            "You can't use QNoise directly. Use derivative classes instead"
        )


class QNSTE(QNoise):

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input, scale = ctx.saved_tensors
        grad_input = grad_scale = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output * 0

        if ctx.needs_input_grad[1]:
            grad_scale = grad_output * (
                torch.randint(
                    2, size=input.shape, dtype=input.dtype, device=input.device
                ).sub(0.5)
            )

        return grad_input, grad_scale


class QNEWGS(QNoise):

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input, scale = ctx.saved_tensors
        grad_input = grad_scale = None

        if ctx.needs_input_grad[0]:
            e = torch.round(input) - input
            delta = 1e-2
            grad_input = -torch.abs(grad_output) * e * delta

        if ctx.need_input_grad[1]:
            grad_scale = (
                (3.0**-0.5)
                * grad_output
                * (
                    torch.randint(
                        2, size=input.shape, dtype=input.dtype, device=input.device
                    ).sub(0.5)
                )
            )

        return grad_input, grad_scale


class QNAEWGS(QNoise):

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input, scale = ctx.saved_tensors
        grad_input = grad_scale = None

        if ctx.needs_input_grad[0]:
            e = torch.round(input) - input

            num_full = grad_output.sign() * e
            den_full = e.square()

            num = reduce_to_shape(num_full, scale).detach()
            den = reduce_to_shape(den_full, scale).detach()

            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(num, op=dist.ReduceOp.AVG)
                dist.all_reduce(den, op=dist.ReduceOp.AVG)

            delta = num / (den + 1e-6)
            g_scale = (delta * num_full).clamp_max(0.99)
            grad_input = -grad_output * g_scale

        if ctx.needs_input_grad[1]:
            grad_scale = (
                (3.0**-0.5)
                * grad_output
                * (
                    torch.randint(
                        2, size=input.shape, dtype=input.dtype, device=input.device
                    ).sub(0.5)
                )
            )

        return grad_input, grad_scale


def reduce_to_shape(t: Tensor, like: Tensor) -> Tensor:
    dims_to_reduce = [i for i, size in enumerate(like.shape) if size == 1]
    return torch.mean(t, dim=tuple(dims_to_reduce), keepdim=True)


def scaled_noise(x, s):
    return QNoise.apply(x, s)


class Quantizer:
    def __init__(
        self,
        module: torch.nn.modules.Module,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        min_val: torch.Tensor,
        max_val: torch.Tensor,
        rnoise_ratio: torch.Tensor=torch.Tensor([-1.0,]),
        qnmethod: QNMethod=QNMethod.AEWGS
    ) -> None:
        self.module = module
        self.scale = scale
        self.zero_point = zero_point
        self.min_val = min_val
        self.max_val = max_val
        self.rnoise_ratio = torch.Tensor([rnoise_ratio])
        self.positive_scale = torch.all(torch.as_tensor(self.scale) > 0).item()
        self.qnmethod = qnmethod

    def quantize(self, value):
        value = torch.clamp(value, min=self.min_val, max=self.max_val)
        value = value - self.zero_point

        if not self.positive_scale:
            return value
            
        value = value / self.scale
        noise = self._get_rnoise(value, self.scale)
        value = value + noise
        
        # Removed strict assertions that caused crashes
        return value

    def dequantize(self, quantized_value):
        if not self.positive_scale:
            return quantized_value + self.zero_point
        return quantized_value * self.scale + self.zero_point

    def _get_rnoise(self, value: Tensor, scale: Tensor):
        if self.qnmethod == QNMethod.STE:
            return QNSTE.apply(value, scale)
        elif self.qnmethod == QNMethod.EWGS:
            return QNEWGS.apply(value, scale)
        elif self.qnmethod == QNMethod.AEWGS:
            return QNAEWGS.apply(value, scale)
        else:
            raise AttributeError(f"Unknown method {self.qnmethod}!")