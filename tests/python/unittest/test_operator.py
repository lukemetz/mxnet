# pylint: skip-file

import numpy as np
import mxnet as mx
from numpy.testing import assert_allclose

def reldiff(a, b):
    diff = np.sum(np.abs(a - b))
    norm = np.sum(np.abs(a))
    if diff == 0:
        return 0
    reldiff = diff  / norm
    return reldiff


def same(a, b):
    return np.sum(a != b) == 0


def check_elementwise_sum_with_shape(shape, n):
    # forward
    inputs = [mx.symbol.Variable('arg%d' % i) for i in range(n)]
    out = mx.symbol.ElementWiseSum(*inputs, name='esum')
    arr = [mx.nd.empty(shape) for i in range(n)]
    arr_grad = [mx.nd.empty(shape) for i in range(n)]
    for i in range(n):
        arr[i][:] = np.random.uniform(-10, 10, shape)
    exec1 = out.bind(mx.Context('cpu'),
                     args=arr,
                     args_grad=arr_grad)
    out1 = exec1.outputs[0].asnumpy()
    exec1.forward()
    out1 = exec1.outputs[0].asnumpy()
    out = sum(a.asnumpy() for a  in arr)
    assert reldiff(out, out1) < 1e-6
    out_grad = mx.nd.empty(shape)
    out_grad[:] = np.random.uniform(-10, 10, shape)
    # backward
    exec1.backward([out_grad])
    for a in arr_grad:
        assert same(a.asnumpy(), out_grad.asnumpy())


def test_elementwise_sum():
    np.random.seed(0)
    nrepeat = 2
    maxdim = 4
    for repeat in range(nrepeat):
        for dim in range(1, maxdim):
            shape = tuple(np.random.randint(1, int(1000**(1.0/dim)), size=dim))
            check_elementwise_sum_with_shape(shape, np.random.randint(1, 8))

def check_slice_channel(dim, num):
    ins = []
    if dim == 2:
        shape = (2,2)
    else:
        shape = (2, 2, 2 ,3)
    ins = [np.ones(shape) * i for i in range(num)]
    e = np.hstack(ins)

    e_nd = mx.nd.empty(e.shape)
    e_nd[:] = e
    data = mx.sym.Variable('data')
    op = mx.sym.SliceChannel(data=data, num_outputs=num)
    arg_shape, output_shape, aux_shape = op.infer_shape(data=e_nd.shape)
    grad_nd = [mx.nd.empty(shape) for shape in arg_shape]

    exe = op.bind(mx.cpu(), args=[e_nd], args_grad=grad_nd)
    assert len(exe.outputs) == num
    o_nd = [exe.outputs[i] for i in range(num)]
    # test forward
    exe.forward()
    for i in range(num):
        assert reldiff(o_nd[i].asnumpy(), ins[i]) < 1e-5
    # test backward
    for i in range(num):
        o_nd[i] += i
    exe.backward(o_nd)
    assert reldiff(grad_nd[0].asnumpy(), np.hstack([ins[i] + i for i in range(num)])) < 1e-5

def check_concat_with_shape(shapes):
    n = len(shapes)
    # forward
    target_dim = 0
    for shape in shapes:
        target_dim += shape[1]

    inputs = [mx.symbol.Variable('arg%d' % i) for i in range(n)]
    out = mx.symbol.Concat(*inputs, name='conc')
    arr = [mx.nd.empty(shape) for shape in shapes]
    for i in range(n):
        arr[i][:] = shapes[i][1]
    arr_np = [np.copy(narray.asnumpy()) for narray in arr]
    arr_grad = [mx.nd.empty(shape) for shape in shapes]
    args = out.list_arguments()
    arg_shapes, out_shapes, aux_shapes = out.infer_shape(**dict(zip(args, shapes)))
    out_grad = mx.nd.empty(out_shapes[0])
    exec1 = out.bind(mx.Context('cpu'),
                     args=arr,
                     args_grad=arr_grad)
    exec1.forward()
    out1 = exec1.outputs[0]
    ret = np.concatenate([narray.asnumpy() for narray in arr], axis=1)
    assert same(out1.asnumpy(), ret)
    # backward
    out1.copyto(out_grad)
    out_grad[:] += 1
    exec1.backward([out_grad])
    for grad, np_grad in zip(arr_grad, arr_np):
        assert same(grad.asnumpy(), np_grad + 1)

def test_concat():
    n = 2
    batch = 2
    ch = [2, 3, 4, 5, 6]
    h = 3
    w = 4
    # test  2D
    for dim in range(2, 6):
        shapes = []
        for i in range(dim):
            shapes.append((batch, ch[i]))
        check_concat_with_shape(shapes)
    # test 4D
    for dim in range(2, 6):
        shapes = []
        for i in range(dim):
            shapes.append((batch, ch[i], h, w))
        check_concat_with_shape(shapes)

def test_slice_channel():
    check_slice_channel(2, 4)
    check_slice_channel(4, 4)
    check_slice_channel(2, 16)

def check_regression(symbol, forward, backward):
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('label')
    out = symbol(data, label)
    shape = (3, 1)
    arr_data = mx.random.uniform(-1, 1, shape)
    arr_label = mx.random.uniform(0, 1, shape[0])
    arr_grad = mx.nd.empty(shape)
    exec1 = out.bind(mx.cpu(),
                     args=[arr_data, arr_label],
                     args_grad={"data" : arr_grad})
    exec1.forward()
    out1 = exec1.outputs[0].asnumpy()
    npout = forward(arr_data.asnumpy())
    assert reldiff(npout, out1) < 1e-6

    exec1.backward()
    npout = backward(npout,  arr_label.asnumpy().reshape(npout.shape))
    assert reldiff(npout, arr_grad.asnumpy()) < 1e-6

def test_regression():
    check_regression(mx.symbol.LogisticRegressionOutput,
                     lambda x: 1.0 / (1.0 + np.exp(-x)),
                     lambda x, y : x - y)
    check_regression(mx.symbol.LinearRegressionOutput,
                     lambda x: x,
                     lambda x, y : x - y)

def check_softmax_with_shape(shape, xpu):
    X = mx.symbol.Variable('X')
    L = mx.symbol.Variable('L')
    Y = mx.symbol.Softmax(data=X, label=L)
    x = mx.random.uniform(-1, 1, shape, ctx = xpu)
    l = mx.nd.empty((shape[0],), ctx = xpu)
    l[:] = np.random.randint(0, shape[0]-1, (shape[0],))
    grad = mx.nd.empty(shape, ctx = xpu)

    exec1 = Y.bind(xpu, args = [x, l], args_grad = {'X': grad})
    print('foward')
    exec1.forward()
    print(exec1.outputs[0].asnumpy())
    exec1.backward()
    print(grad.asnumpy())

def check_multi_softmax_with_shape(shape, xpu):
    X = mx.symbol.Variable('X')
    L = mx.symbol.Variable('L')
    Y = mx.symbol.Softmax(data=X, label=L, multi_output=True)
    x = mx.random.uniform(-1, 1, shape, ctx = xpu)
    l = mx.nd.empty((shape[0], shape[2]), ctx = xpu)
    l[:] = np.random.randint(0, shape[1]-1, (shape[0], shape[2]))
    grad = mx.nd.empty(shape, ctx = xpu)

    exec1 = Y.bind(xpu, args = [x, l], args_grad = {'X': grad})
    exec1.forward()
    print(exec1.outputs[0].asnumpy())
    exec1.backward()
    print(grad.asnumpy())

def test_python_op():
    X = mx.symbol.Variable('X')
    op = mx.operator.NumpyOp()
    s = op.get_symbol(X, name='numpy_op')

    x = mx.ndarray.ones((10))*10
    dx = mx.ndarray.zeros((10))
    dy = mx.ndarray.ones((10))
    exec1 = s.bind(mx.cpu(), args=[x], args_grad = {'X': dx})
    exec1.forward()
    assert reldiff(x.asnumpy(), exec1.outputs[0].asnumpy()) < 1e-5
    exec1.backward(dy)
    assert reldiff(dy.asnumpy(), dx.asnumpy()) < 1e-5

def test_swapaxes():
    data = mx.symbol.Variable('data')
    shape = (2, 3, 4)
    data_tmp = np.ones(shape)
    data_tmp[0] = 1
    data_tmp[1] = 2
    arr_data = mx.nd.array(data_tmp)
    swap0 = mx.symbol.SwapAxis(data=data, dim1=0, dim2=2)
    swap = mx.symbol.SwapAxis(data=swap0, dim1=1, dim2=2)
    exe_c = swap.bind(mx.cpu(), args=[arr_data])
    exe_c.forward()
    out = exe_c.outputs[0].asnumpy()

    swap0_ = np.swapaxes(data_tmp, 0, 2)
    swap_ = np.swapaxes(swap0_, 1, 2)

    assert reldiff(out, swap_) < 1e-6

def test_scalarop():
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:]=5
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:]=3

    test = 2 / (4-((1+data+1)*2/5)-0.2)
    exe_test = test.bind(mx.cpu(), args=[arr_data], args_grad=[arr_grad])
    exe_test.forward()
    out = exe_test.outputs[0].asnumpy()
    npout_1 = (4-((1+data_tmp+1)*2/5)-0.2)
    npout = 2/npout_1
    assert reldiff(out, npout) < 1e-6

    out_grad = mx.nd.empty(shape)
    out_grad[:] = 2;
    npout_grad = out_grad.asnumpy()
    npout_grad = npout_grad*2/5
    npout_grad = 2*npout_grad /(npout_1 *npout_1 )
    exe_test.backward(out_grad)
    assert reldiff(arr_grad.asnumpy(), npout_grad) < 1e-6

class NumericGrad(object):
    """ Class based on Theano's `theano.gradient.numeric_grad` [1]
    Calculates a numeric gradient via finite difference method.

    Parameters:
    -----------
    executor: `mxnet.executor.Executor`
        exectutor that computes the forward pass

    location: list np.ndarray
        location in which to compute gradient. list should be the same size
        as executor.arg_arrays

    References
    ---------
    ..[1] https://github.com/Theano/Theano/blob/master/theano/gradient.py
    """
    def __init__(self, executor, location, eps=1e-4):
        args = executor.arg_arrays
        for a, l in zip(args, location):
            a[:] = np.asarray(l)
        approx_grads = [np.zeros_like(l) for l in location]

        executor.forward()
        f_x = executor.outputs[0].asnumpy()

        x_copy = [np.copy(x) for x in location]
        for ap_grad, loc, reset in zip(approx_grads, location, x_copy):
            for i in range(np.prod(loc.shape)):
                # inplace update of memory
                loc.ravel()[i] += eps

                # set initial states. Need to set all due to inplace operations
                for inp, val in zip(args, location):
                    inp[:] = val
                executor.forward()
                f_eps = executor.outputs[0].asnumpy()
                ap_grad.ravel()[i] = (f_eps - f_x) / eps
                loc.ravel()[i] = reset.ravel()[i]

        self.f_x = f_x
        self.approx_grads = approx_grads
        print self.approx_grads, "approx grad"

    def rel_error_to(self, grad):
        abs_error = [np.abs(a - b) for a, b in zip(grad, self.approx_grads)]
        rel_error = [e / np.maximum(np.abs(a) + np.abs(b), 1e-8) for (e, a, b)
                     in zip(abs_error, grad, self.approx_grads)]
        return rel_error

def verify_op(sym, location, target, n_tests):
    """ Class based on Theano's `theano.gradient.numeric_grad` [1]

    References
    ---------
    ..[1] https://github.com/Theano/Theano/blob/master/theano/gradient.py
    """



    # random_projection should not have elements too small,
    # otherwise too much precision is lost in numerical gradient
    def random_projection(shape):
        plain = np.rng.rand(*shape) + 0.5
        return plain

    sym = mx.sym.SumAll(sym)
    out = mx.sym.Loss(sym)
    args = out.list_arguments()

    kwargs = {a:loc.shape for a,loc in zip(args, location)}
    print kwargs
    #executor = out.simple_bind(mx.cpu(), grad_req="write", **kwargs)
    #executor = out.bind(mx.cpu(), grad_req="write", grad_arrays=
    arr_data = [mx.nd.array(l) for l in location]
    arr_grad = [mx.nd.empty(l.shape) for l in location]

    executor = out.bind(mx.cpu(), args=arr_data, args_grad=arr_grad)

    inps = executor.arg_arrays
    if len(inps) != len(location):
        raise ValueError("Executor arg_arrays and and location len do not match."
                         "Got %d inputs and %d locations"%(len(inps), len(location))
        )
    for inp, source in zip(location, executor.arg_arrays):
        source[:] = inp

    for g in executor.grad_arrays:
        if g:
            g[:] = 0

    assert len(executor.outputs) == 1

    executor.forward()
    executor.backward()
    symbolic_grad = [g.asnumpy() for g in executor.grad_arrays]

    print location[0], "location"
    print symbolic_grad, "computed grad symbolic"
    error = NumericGrad(executor, location).rel_error_to(symbolic_grad)
    for e in error:
        assert_allclose(e, np.zeros_like(e), atol=1e-2)


def test_scalar_pow():
    data = mx.symbol.Variable('data')
    shape = (1, 1)
    data_tmp = np.ones(shape)
    test = data ** 2
    verify_op(test, [data_tmp], [data_tmp], n_tests=2)

def check_elementwise_op(sym_forward, np_forward):
    data = mx.symbol.Variable('data')
    test = sym_forward(data)

    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:] = 5
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:] = 3

    exe_test = test.bind(mx.cpu(), args=[arr_data], args_grad=[arr_grad])
    exe_test.forward()
    out = exe_test.outputs[0].asnumpy()
    npout = forward(data_tmp)

    assert_allclose(out, npout)

    out_grad = mx.nd.empty(shape)
    out_grad[:] = 2;
    npout_grad = out_grad.asnumpy()

    exe_test.backward(out_grad)
    grad = arr_grad.asnumpy()
    npgrad = data_tmp**3 * 4 * 2

    assert_allclose(grad, npgrad)


def test_symbol_pow():
    shape = (1, 1)

    data = mx.symbol.Variable('data')
    data_tmp = np.ones(shape)*2

    exp = mx.symbol.Variable('exp')
    exp_tmp = np.ones(shape)*3

    test = data**exp

    verify_op(test, [data_tmp, exp_tmp], [data_tmp**exp_tmp], 2)

test_symbol_pow()
import ipdb; ipdb.set_trace()


def test_pow_fn():
    shape = (3, 4)
    exp = mx.symbol.Variable("exp")
    y = mx.sym.pow(2, exp)
    x = np.ones(shape)*3
    verify_op(y, [x], [2**x], n_tests=2)
#test_pow_fn()

def test_sum():
    shape = (3, 4)
    x = mx.symbol.Variable("data")

    y = mx.sym.SumAll(x)
    verify_op(y, [np.ones(shape)], [np.ones((1,))*np.prod(shape)], n_tests=2)

#test_sum()

if __name__ == '__main__':
    test_elementwise_sum()
    test_concat()
    test_slice_channel()
    test_regression()
    test_python_op()
    test_swapaxes()
    test_scalarop();
    test_scalar_pow()
    test_symbol_pow()
    test_pow_fn()
    #check_softmax_with_shape((3,4), mx.cpu())
    #check_multi_softmax_with_shape((3,4,5), mx.cpu())
