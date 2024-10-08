from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        max_z = Z.max(axis=self.axes, keepdims=True)
        return array_api.log(array_api.sum(array_api.exp(Z - max_z), axis=self.axes)) + max_z.squeeze()

    def gradient(self, out_grad, node):
        z = node.inputs[0]
        maxz = z.realize_cached_data().max(axis=self.axes, keepdims=True)
        zexp = exp(z - maxz.broadcast_to(z.shape))
        zsumexp = summation(zexp, self.axes)
        grad_div_zse = out_grad / zsumexp.broadcast_to(out_grad.shape)
        grad_div_zse_b = grad_div_zse.reshape(maxz.shape).broadcast_to(z.shape)
        return grad_div_zse_b * zexp


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
