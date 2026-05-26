# -*- coding: utf-8 -*-
"""Array backend selection for CPU NumPy or GPU CuPy execution."""

import numpy as _np

try:
    import cupy as _cp

    xp = _cp
    GPU_ENABLED = True
except ImportError:
    _cp = None
    xp = _np
    GPU_ENABLED = False


def to_device(array):
    """Move an array-like value to the active backend."""
    return xp.asarray(array)


def to_cpu(array):
    """Move an array-like value to NumPy for file IO or plotting."""
    if GPU_ENABLED and isinstance(array, _cp.ndarray):
        return _cp.asnumpy(array)
    return _np.asarray(array)


def scalar_to_float(value):
    """Convert NumPy/CuPy scalar-like values to a Python float."""
    if GPU_ENABLED and isinstance(value, _cp.ndarray):
        return float(_cp.asnumpy(value))
    return float(value)
