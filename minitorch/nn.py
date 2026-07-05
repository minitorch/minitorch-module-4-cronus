from typing import Tuple

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
    output_tensor = input.contiguous().view(batch, channel, new_height * new_width * kw, kh)#.contiguous().view(batch, channel, new_height, new_width, kw * kh)
    return tuple([output_tensor, new_height, new_width])


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    
    batch, channel, _, _ = input.shape

    tiled_tensor, new_h, new_w = tile(input, kernel)
    #out_tensor = tiled_tensor.mean(4).view(batch, channel, new_h, new_w)

    print("input:")
    print(input._tensor.to_string())
    print(input.shape)
    print(kernel)
    print("tiled:")
    print(tiled_tensor._tensor.to_string())
    print(tiled_tensor.shape)
    #print("out:")
    #print(out_tensor._tensor.to_string())
    #print(out_tensor.shape)
    #return out_tensor

def argmax(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    raise NotImplementedError("Need to implement for Task 4.4")

def max() -> Tensor:
    raise NotImplementedError("Need to implement for Task 4.4")
    
def softmax(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    raise NotImplementedError("Need to implement for Task 4.4")

def logsoftmax(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    raise NotImplementedError("Need to implement for Task 4.4")

def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    raise NotImplementedError("Need to implement for Task 4.4")

def dropout(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    raise NotImplementedError("Need to implement for Task 4.4")
    


