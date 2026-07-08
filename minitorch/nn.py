from typing import Tuple, Optional

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    #raise NotImplementedError("Need to implement for Task 4.3")

    print(input._tensor.is_contiguous())
    print(input._tensor.strides)

    new_height = height / kh
    new_width  = width / kw
    #output_tensor = input.contiguous().view(batch, channel, new_height, new_width, kw * kh)

    print("input:")
    print(input._tensor.to_string())
    print(input.shape)
    print(kernel)
    
    if (kh > kw):
        output_tensor2 = input.contiguous().permute(0,1,3,2).contiguous().view(batch, channel, new_height, new_width, kw, kh).permute(0,1,2,4,3,5).contiguous().view(batch, channel, new_height, new_width, kh * kw)
    else:
        output_tensor2 = input.contiguous().contiguous().view(batch, channel, new_height, new_width, kh, kw).permute(0,1,2,4,3,5).contiguous().view(batch, channel, new_height, new_width, kh * kw)

    print("tiled:")
    print(output_tensor2._tensor.to_string())
    print(output_tensor2.shape)

    return tuple([output_tensor2, new_height, new_width])


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    
    batch, channel, _, _ = input.shape

    tiled_tensor, new_h, new_w = tile(input, kernel)
    out_tensor = tiled_tensor.mean(4).view(batch, channel, new_h, new_w)

    return out_tensor

def argmax(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    raise NotImplementedError("Need to implement for Task 4.4")

class Max(Function):
    @staticmethod
    def forward(ctx: Context, t: Tensor, dim: Tensor) -> Tensor:
        #raise NotImplementedError("Need to implement for Task 4.4")
        #print("Enter Max forward:")
        ctx.save_for_backward(t, dim)
        max_reduce = FastOps.reduce(operators.max)
        #print(t._tensor._storage)
        #print(t.dims)
        test = max_reduce(t, int(dim.item()))
        #print(test)
        return test

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        raise NotImplementedError("Need to implement for Task 4.4")

def max (t: Tensor, dim: Optional[int] = None) -> float:
    if dim is None:
        return Max.apply(t.contiguous().view(t.size), t._ensure_tensor(0))[0]
    else:
        return Max.apply(t, t._ensure_tensor(dim))

    
def softmax(input: Tensor, dim: int) -> Tensor:
    #raise NotImplementedError("Need to implement for Task 4.4")
    #print("Enter softmax:")
    #print(input)
    #print(dim)
    #print(input.exp())
    #print(input.exp().sum(dim))
    #print(input.exp() / input.exp().sum(dim))
    #print("Exit softmax")
    return input.exp() / input.exp().sum(dim)

def logsoftmax(input: Tensor, dim: int) -> Tensor:
    #raise NotImplementedError("Need to implement for Task 4.4")
    return input - input.exp().sum(dim).log()

def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    batch, channel, _, _ = input.shape

    tiled_tensor, new_h, new_w = tile(input, kernel)
    out_tensor = max(tiled_tensor, 4).view(batch, channel, new_h, new_w)

    return out_tensor

def dropout(input: Tensor, prob: float, ignore: bool = False) -> Tensor:
    #raise NotImplementedError("Need to implement for Task 4.4")
    
    print(input.shape)
    #height, weight = input.shape
    return input
    


