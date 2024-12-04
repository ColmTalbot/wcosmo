from importlib import import_module
from importlib.util import find_spec

AVAILABLE_BACKENDS = list()

for backend in ["numpy", "wcosmo.backend.jax", "wcosmo.backend.cupy"]:
    if find_spec(backend) is not None:
        if "wcosmo" in backend:
            import_module(backend)
        AVAILABLE_BACKENDS.append(backend.split(".")[-1])
