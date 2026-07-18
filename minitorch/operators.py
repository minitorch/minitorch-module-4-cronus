"""Collection of the core mathematical operators used throughout the code base."""

import math
from collections.abc import Iterable
from typing import Callable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """
    f(x, y) = x * y
    """
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -1.0 * x


def lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    return True if math.fabs(x - y) < 1e-2 else False


def sigmoid(x: float) -> float:
    """f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}"""
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def relu(x: float):
    """f(x) = x if x is grater than 0, else 0"""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def inv(x: float) -> float:
    return 1 / x


def log_back(x: float, y: float) -> float:
    return (1 / x) * y


def inv_back(x: float, y: float) -> float:
    return -(1 / (x * x)) * y


def relu_back(x: float, y: float) -> float:
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order function that applies a given function to each element of an iterable
    """
    def apply(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret
    
    return apply

def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order function that combines elements from two iterables using a given function
    """
    def apply(lsa: Iterable[float], lsb: Iterable[float]) -> Iterable[float]:
        ret = []
        for a, b in zip(lsa, lsb):
            ret.append(fn(a,b))
        return ret
    return apply

def reduce(fn: Callable[[float, float], float]) -> Callable[[Iterable[float]], float]:
    """
    Higher-order function that reduces an iterable to a single value using a given function
    """
    def apply(ls: Iterable[float]) -> float:
        if ls == []:
            return 0
        result = ls[0]
        for a in ls[1:]:
            result = fn(result, a)
        return result
    return apply



negList: Callable[[Iterable[float]], Iterable[float]] = map(neg)

addLists: Callable[[Iterable[float], Iterable[float]], Iterable[float]] = zipWith(add)
  
sum: Callable[[Iterable[float]], float] = reduce(add)

prod: Callable[[Iterable[float]], float] = reduce(mul)
