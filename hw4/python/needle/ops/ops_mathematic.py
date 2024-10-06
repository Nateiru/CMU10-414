"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return self.scalar * (power_scalar(a, self.scalar-1)) * out_grad



def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        grad_a = out_grad / b
        grad_b = -a / b / b * out_grad
        return grad_a, grad_b


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        transpose_axis = [x for x in range(len(a.shape))]
        if self.axes is None:
            transpose_axis[-2], transpose_axis[-1] = \
                transpose_axis[-1], transpose_axis[-2]
        else:
            transpose_axis[self.axes[0]], transpose_axis[self.axes[1]] = \
                self.axes[1], self.axes[0]
        return a.permute(transpose_axis)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        assert array_api.prod(self.shape) == array_api.prod(a.shape), "The reshape size is not compatible"
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        assert len(self.shape) >= len(a.shape), \
            "The target shape's dimension count {} should be greater than \
                or equal to the input shape's dimension count {}".format(len(self.shape), len(a.shape))
        for i in range(len(a.shape)):
            assert a.shape[-1 - i] == self.shape[-1 - i] or a.shape[-1 - i] == 1, \
                "The input shape {} is not compatible with the target shape {}".format(a.shape, self.shape)
        return a.broadcast_to(self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        shrink_dims = [i for i in range(len(self.shape))]
        for i, (ori, cur) in enumerate(zip(reversed(input_shape), reversed(self.shape))):
            if ori == cur:
                shrink_dims[len(self.shape) - i - 1] = -1
        shrink_dims = tuple(filter(lambda x: x >= 0, shrink_dims))
        return out_grad.sum(shrink_dims).reshape(input_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if isinstance(self.axes, (list, tuple)) and len(self.axes) > 1:
            # multiple axes case
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis=axis)
            return a
        b = array_api.sum(a, axis=self.axes)
        return b

    def gradient(self, out_grad, node):
        new_shape = list(node.inputs[0].shape)
        if self.axes is None:
            axes = range(len(new_shape))
        elif isinstance(self.axes, tuple):
            axes = self.axes
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            raise ValueError(
                "Unsupported axes type, must be int, tuple or None!")
        for axis in axes:
            new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a@b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        adjoint1 = out_grad @ transpose(b)
        adjoint2 = transpose(a) @ out_grad
        adjoint1 = summation(adjoint1, axes=tuple(range(len(adjoint1.shape) - len(a.shape))))
        adjoint2 = summation(adjoint2, axes=tuple(range(len(adjoint2.shape) - len(b.shape))))
        return adjoint1, adjoint2


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -1 * a

    def gradient(self, out_grad, node):
        return (-1 * out_grad, )


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        relu_mask = Tensor(node.inputs[0].cached_data > 0)
        return out_grad * relu_mask


def relu(a):
    return ReLU()(a)

class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        return out_grad * (1 - node ** 2)


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        if len(args) > 0:
            shape = args[0].shape
            for arg in args:
                assert arg.shape == shape, "The shape of all tensors should be the same"
            ret_shape = list(shape)
            ret_shape.insert(self.axis, len(args))
            ret = array_api.empty(ret_shape, device=args[0].device)
            for i, arg in enumerate(args):
                # 在一维数组中使用 slice(None) 等同于选择所有元素
                slices = [slice(None)] * len(ret_shape)
                slices[self.axis] = i
                ret[tuple(slices)] = arg
            return ret


    def gradient(self, out_grad, node):
        return split(out_grad, self.axis)


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ret = []
        ret_shape = list(A.shape)
        ret_shape.pop(self.axis)
        for i in range(A.shape[self.axis]):
            slices = [slice(None)] * len(A.shape)
            slices[self.axis] = i
            ret.append((A[tuple(slices)]).compact().reshape(ret_shape))
        return tuple(ret)

    def gradient(self, out_grad, node):
        return stack(out_grad, self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return a.flip(self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        new_shape = list(a.shape)
        for axis in self.axes:
            if axis >= len(a.shape):
                continue
            new_shape[axis] = new_shape[axis] * (self.dilation + 1)
        new_shape = tuple(new_shape)
        arr = a.device.full(new_shape, 0)
        slices = [slice(0, n) for n in arr.shape]
        for axis in self.axes:
            if axis >= len(a.shape):
                continue
            slices[axis] = slice(0, arr.shape[axis], self.dilation + 1)
        arr[tuple(slices)] = a
        return arr

    def gradient(self, out_grad, node):
        return UnDilate(self.axes, self.dilation)(out_grad)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        slices = [slice(0, n) for n in a.shape]
        for axis in self.axes:
            if axis >= len(a.shape):
                continue
            slices[axis] = slice(0, a.shape[axis], self.dilation + 1)
        return a[tuple(slices)].compact()

    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        A = A.pad(((0, 0), (self.padding, self.padding),
                  (self.padding, self.padding), (0, 0)))
        N, H, W, C_in = A.shape
        K, K_, C_in_, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        assert K == K_, "Conv kernel should be a square tensor"
        assert C_in == C_in_, "Conv kernel and input are not compatible"

        inner_dim = K * K * C_in
        out_H, out_W = (H - K + 1) // self.stride, (W - K + 1) // self.stride
        im2col = A.as_strided(shape=(N, out_H, out_W, K, K, C_in),
                              strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs))\
            .compact()\
            .reshape((N * out_H * out_W, inner_dim))
        out = im2col @ B.compact().reshape((K * K_ * C_in_, C_out))
        return out.compact().reshape((N, out_H, out_W, C_out))

    def gradient(self, out_grad, node):
        X, W = node.inputs
        K, _, _, _ = W.shape

        if self.stride > 1:
            out_grad = dilate(out_grad, (1, 2), self.stride - 1)
        W_permute = transpose(flip(W, (0, 1)), (2, 3))  # K * K * C_out * C_in
        # out_grad: # N * (H+2P-K+1) * (W+2P-K+1) * C_out
        X_grad = conv(out_grad, W_permute, padding=K - 1 - self.padding)

        X_permute = transpose(X, (0, 3))  # C_in * H * W * N
        # (H+2P-K+1) * (W+2P-K+1) * N * C_out
        grad_permute = transpose(transpose(out_grad, (0, 1)), (1, 2))
        # C_in * H * W * C_out
        W_grad = conv(X_permute, grad_permute, padding=self.padding)
        W_grad = transpose(transpose(W_grad, (0, 1)),
                           (1, 2))  # H * W * C_in * C_out

        return X_grad, W_grad


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
