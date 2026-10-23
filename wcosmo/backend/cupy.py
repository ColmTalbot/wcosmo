import cupy as cp
from cupy_backends.cuda.api.runtime import CUDARuntimeError

try:
    cp.cuda.Device()
except CUDARuntimeError:
    raise ImportError
