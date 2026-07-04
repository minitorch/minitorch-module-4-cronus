from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    #raise NotImplementedError("Need to implement for Task 1.1")
    
        
    if arg == 0:
        delta_vals = (vals[0] + epsilon,) + vals[1:]
    elif arg == len(vals) - 1:
        delta_vals = vals[0:arg - 2] + (vals[-1] + epsilon,)
    else:
        delta_vals = vals[0:arg - 1] + (vals[arg] + epsilon,) + vals[arg+1:]
    
    
    return (f(*delta_vals) - f(*vals))/epsilon

variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    #raise NotImplementedError("Need to implement for Task 1.4")
    #print("enter topo sort")
    visited_node = [variable]
    if not variable.is_constant():
        if not variable.is_leaf():
            #print("is_leaf:")
            #print(variable)
            #print(variable.parents)
            for parent in variable.parents:
                visited_node += topological_sort(parent)
    #print("after topo sort:")
    #print(visited_node)
    return visited_node

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    #raise NotImplementedError("Need to implement for Task 1.4")
    
    # dict for variable and derivatives
    variable.derivative = deriv
    for sorted_var in topological_sort(variable):
        #print(sorted_var)
        if not sorted_var.is_constant():
            if (not sorted_var.is_leaf()):
                for (var, local_deriv) in sorted_var.chain_rule(sorted_var.derivative):
                    #print("var:" + str(var.unique_id))
                    #print(var)
                    #print("local_deriv:")
                    #print(local_deriv)
                    if (var.is_leaf()):
                        var.accumulate_derivative(local_deriv)
                    else:
                        var.derivative = local_deriv
            


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
