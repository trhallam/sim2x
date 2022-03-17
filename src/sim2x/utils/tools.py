"""Contains special tools and classes useful to all parts of sim2x
"""
import importlib
import sys
import pathlib


import numpy as np

from ..typing import NDArrayOrFloat

##  LIST TOOLS


def strictly_increasing(L):
    return all(x < y for x, y in zip(L, L[1:]))


def strictly_decreasing(L):
    return all(x > y for x, y in zip(L, L[1:]))


def increasing(L):
    return all(x <= y for x, y in zip(L, L[1:]))


def decreasing(L):
    return all(x >= y for x, y in zip(L, L[1:]))


def uniqify(seq, idfun=None):
    """Return only unique values in a sequence"""
    # order preserving
    if idfun is None:

        def idfun(x):
            return x

    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)
    return result


def ndim_index_list(n):
    """Creates the list for indexing input data based upon dimensions in list n.

    As input takes a list n = [n0,n1,n2] of the length of each axis in order of
    slowest to fastest cycling. Such that:
    n0 is numbered 1,1,1, 1,1,1, 1,1,1; 2,2,2, 2,2,2, 2,2,2; 3,3,3, 3,3,3, 3,3,3;
    n1 is numbered 1,1,1, 2,2,2, 3,3,3; 1,1,1, 2,2,2, 3,3,3; 1,1,1, 2,2,2, 3,3,3;
    n2 is numbered 1,2,3, 1,2,3, 1,2,3; 1,2,3, 1,2,3, 1,2,3; 1,2,3, 1,2,3, 1,2,3;

    Args:
        n (list): a list of intergers giving the dimensions of the axes.

    Note: axes not required are given values n = 1
          indexing starts from 1, Zero based indexing is available by subtracting
          1 from all values.
    """
    if isinstance(n, list):
        ind = list()
        ind.append(np.arange(1, n[0] + 1).repeat(np.product(n[1:])))
        for i in range(1, len(n) - 1):
            ni = i + 1
            t = np.arange(1, n[i] + 1)
            t = t.repeat(np.product(n[ni:]))
            t = np.tile(t, np.product(n[:i]))
            ind.append(t)
        ind.append(np.tile(np.arange(1, n[-1] + 1), np.product(n[:-1])))
        return ind
    else:
        raise ValueError("n must be of type list containing integer values")


def module_loader(module_name, module_path):
    """Loads a Python file or module dynamically for use in scripts.
    Args:
        module_name (str): The name of the module
        module_path (str/Path): The complete module Path
    """
    module_path = pathlib.Path(module_path)
    module_loc = module_path.parent
    if module_loc not in sys.path:
        sys.path.append(module_loc)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_broadcastable(**kwargs: NDArrayOrFloat) -> tuple:
    """Check that kwargs can be numpy broadcast against each other for matrix
    operations.
    Args:
        kwargs: Keywords to check with float of array-like values.
    Returns
        Shapes of broadcast keywords
    Raises:
        ValueError if not broadcastable
    """
    check_argshapes = [np.atleast_1d(arg).shape for arg in kwargs.values()]

    try:
        shapes = np.broadcast_shapes(*tuple(check_argshapes))
    except:
        msg = f"Cannot broadcast shapes: " + ", ".join(
            [f"{name}:{shp}" for name, shp in zip(kwargs, check_argshapes)]
        )
        raise ValueError(msg)

    return shapes


def safe_divide(a: NDArrayOrFloat, b: NDArrayOrFloat) -> NDArrayOrFloat:
    """Helper function to avoid divide by zero in arrays and floats.
    Args:
        a: Numerator
        b: Denominator
    Returns:
        a/b replace div0 by 0
    """
    bc_shp = check_broadcastable(a=a, b=b)
    return np.divide(a, b, out=np.zeros(bc_shp), where=b != 0.0)


def nan_divide(a: NDArrayOrFloat, b: NDArrayOrFloat) -> NDArrayOrFloat:
    """Helper function to avoid divide by zero in arrays and floats.
    Args:
        a: Numerator
        b: Denominator
    Returns:
        a/b replace div0 by np.nan
    """
    bc_shp = check_broadcastable(a=a, b=b)
    return np.divide(a, b, out=np.full(bc_shp, np.nan), where=b != 0.0)
