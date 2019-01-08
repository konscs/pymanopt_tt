try:
    import t3f
except ImportError:
    t3f = None

from ._backend import Backend, assert_backend_available


class T3FBackend(Backend):

    def __str__(self):
        return "t3f"

    def is_available(self):
        return t3f is not None

    @assert_backend_available
    def is_compatible(self, objective, argument):
        if isinstance(argument, t3f.tensor_train.TensorTrain):
            return True
        return False

    @assert_backend_available
    def compile_function(self, objective, argument):
        def cfun(x):
            return objective(x)
        return cfun

    @assert_backend_available
    def compute_gradient(self, objective, argument):
        def gfun(x):
            return t3f.autodiff.gradients(objective, x)
        return gfun 

    @assert_backend_available
    def compute_hessian(self, objective, argument):
        def hfun(x,v):
            return t3f.autodiff.hessian_vector_product(objective, x, v)
        return hfun
