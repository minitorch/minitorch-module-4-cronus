from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
from numba import cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage
    """

    # TODO: Implement for Task 2.1.
    #raise NotImplementedError("Need to implement for Task 2.1")
    position = 0
    for i in range(len(index)):
        position = position + index[i] * strides[i]
    return position

def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    # TODO: Implement for Task 2.1.
    #raise NotImplementedError("Need to implement for Task 2.1")
    dim = len(shape)
    #print("dim: "+ str(dim))
    #print(str(shape) + " ordinal: " + str(ordinal))
    total = 1
    for i in range(dim):
        total = total * shape[i]

    mapped_positions = 0
    for i in range(dim):
        #print("strides: " + str(strides[i]) + " for " + str(i))
        #if (i == 0):
        #    out_index[i] = ordinal // strides[i]
        #else:
        #    ordinal = ordinal - strides[i - 1] * out_index[i - 1]
        #    out_index[i] = ordinal // strides[i]

        stride = 1
        for j in range(i + 1, dim):
            stride = stride * shape[j] 

        out_index[i] = (ordinal - mapped_positions) // stride
        mapped_positions = mapped_positions + stride * out_index[i]

    #print(out_index)

def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
        None
    """
    # TODO: Implement for Task 2.2.
    #raise NotImplementedError("Need to implement for Task 2.2")
    
    #threadIdx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    for i in range(len(shape)):
        if (shape[i] == 1):
            out_index[i] = 0
        else:
            out_index[i] = big_index[len(big_shape) - len(shape) + i]

    #if (threadIdx == 1):
    #    print("inside broadcast_index")
    #    print(big_index[0])
    #    print(big_index_reversed[0])
    #    print(big_index_reversed_truncated[0])
    #    print(out_index[0])
    #    print("exit broadcast_index")


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    """
    # TODO: Implement for Task 2.2.
    #raise NotImplementedError("Need to implement for Task 2.2")
    #print("Inside shape_broadcast:")
    #print(shape1)
    #print(shape2)
    # check whether 2 shapes are broadcastable
    if (len(shape1) > len(shape2)):
        big_shape = shape1[::-1]
        small_shape = shape2[::-1]
    else:
        big_shape = shape2[::-1]
        small_shape = shape1[::-1]
    
    #print(big_shape)
    #print(small_shape)
    # check whetehr broadcastable
    for big_element, small_element in zip(big_shape, small_shape):
        if not ((big_element == small_element) or (big_element == 1) or (small_element == 1)):
            raise IndexingError(f"Cannot broadcast for big element {big_element} and small_element {small_element}")
    
    reversed_broadcasted_shape = list(big_shape)
    
    for i, small_element in enumerate(small_shape):
        if (big_shape[i] == 1):
            reversed_broadcasted_shape[i] = small_element
    
    #print(reversed_broadcasted_shape)

    #print("Leaving shape_broadcast:")
    return tuple(reversed(reversed_broadcasted_shape))

def strides_from_shape(shape: UserShape) -> UserStrides:
    shape_length = len(shape)
    layout = np.ones(shape_length, dtype=int)
    total = 1
    for i in range(shape_length):
        total = total * shape[i]
    for i in range(shape_length):
        layout[i] = total / shape[i]
        total = layout[i]
    #layout = [1]
    #offset = 1
    #for s in reversed(shape):
    #    layout.append(s * offset)
    #    offset = s * offset
    return tuple(layout)


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            order (list): a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        # TODO: Implement for Task 2.1.
        #raise NotImplementedError("Need to implement for Task 2.1")
        permuted_strides: UserStrides=[]
        permuted_shape: UserShape = []
        for i in range(self.dims):
            permuted_strides.append(self.strides[order[i]])
            permuted_shape.append(self.shape[order[i]])
        return TensorData(self._storage, tuple(permuted_shape), tuple(permuted_strides))

    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
