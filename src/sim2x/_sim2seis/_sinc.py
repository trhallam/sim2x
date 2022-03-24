from functools import partial
from typing import Any, Callable
import numpy as np
from numpy import typing as npt
import numba
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz, solve


def resample(x, t, u, interp="sinc", sinc_tab=None):
    """Signal resampling/interpolation

    Args:
        x (array-like): Amplitude values x(t)
        t (array-like): Input x values, must be regularly sampled.
        u (array-like): Output x values
        interp (str, optional): Type of interpolation to use. One of 'sinc',
            'cubic'. Defaults to 'sinc'.
        sinc_tab (array-like, optional): Pre-defined sinc table.
            See make_sinc_table. Defaults to None. If None and interp = 'sinc',
            sinc table as 12 samples and 30 functions.
    """
    if sinc_tab is None and interp == "sinc":
        sinc_tab = make_sinc_table(ns=12, nf=55)

    if interp == "sinc":
        return fast_sinc_interp(x, t, u, sinc_tab)
    elif interp == "cubic" or interp == "linear":
        xft = interp1d(t, x, kind=interp, bounds_error=False, fill_value=0.0, copy=True)
        return xft(u)
    else:
        raise ValueError(
            "interp {} is unknown, should be one of 'sinc', 'cubic'".format(interp)
        )


@numba.jit
def sinc(x):
    """Private sinc function

    Args:
        x (array-like): Funciton to apply sinc to.

    Returns:
        array-like: sinc(x)
    """
    y = np.pi * np.where(x == 0, 1.0e-20, x)
    return np.sin(y) / y


# @numba.jit
def make_sinc_table(ns=8, nf=25):
    """sinc function table with size [ns, nf]

    For a sample interval dt there will be a unique, optimised sinc function
    designed for interpolation every 1/nf of dt.

    Args:
        ns (int): Defaults to 8. The number of samples per sinc function.
        nf (int): Defaults to 25. The number of sinc functions.
    """
    frac = np.arange(nf) / nf
    sinc_table = np.zeros((ns, nf))
    jmax = int(np.fix(nf / 2) + 1)

    # first half of table is computed by least squares
    # second half by symetry
    fmax = np.min([0.066 + 0.265 * np.log(ns), 1.0])
    a = sinc(fmax * np.arange(0, ns))
    a = toeplitz(a.T, a)
    for j in range(jmax):
        b = fmax * (np.arange(ns / 2 - 1, -ns / 2 - 1, -1) + frac[j] * np.ones(ns))
        c = sinc(b)
        sinc_table[:, j] = solve(a, c.T)

    jtable = nf - 1
    ktable = 2
    while np.all(sinc_table[:, jtable] == 0.0):
        sinc_table[:, jtable] = np.flipud(sinc_table[:, ktable])
        jtable -= 1
        ktable += 1

    return sinc_table


@numba.njit
def fast_sinc_interp(x, t, u, sinc_table):
    """Faster sinc interpolater based upon the 8 point interpolation tables used
    in Seismic Unix. This is an approximation to a full sinc interpolation but
    is much faster.

    Speed improvements can also be achieved by suppling a sinc_table. This
    negates the need to recalculate sinc_tables for sinc interpolation upon
    multiple time series.

    Args:
        x (array-like): Vector of input signal amplitudes
        t (array-like): Vector of input signal times (regularly sampled)
        u (array-like): Vector of output sample times (any sampling)
        sinc_table (array-like): This is the output of make_sinc_table

    Returns:
        (array-like): Vector x(u) sampled from x(t).
    """
    y = np.zeros(len(u))

    ns, nf = sinc_table.shape

    # perform interpolation by first extrapolating constant end values
    # for top
    ind = (u <= t[0]).nonzero()
    ind_len = len(ind)
    if len(ind) > 0:
        y[ind] = x[0] * np.ones(ind_len)
    # for tail
    ind = (u >= t[-1]).nonzero()
    if len(ind) > 0:
        y[ind] = x[-1] * np.ones(ind_len)
    dt = t[1] - t[0]  # get sample rate - must be regularly sampled input
    ind = np.logical_and(u > t[0], u < t[-1]).nonzero()
    if ind[0].shape[0] != 0:
        pdata = (u[ind] - t[0]) / dt + 1
        # fractional sample increment for each interpolation point
        delta = pdata - (np.abs(pdata) // 1 * np.sign(pdata))  # np.fix(pdata)
        # compute row number in interpolation table
        ptable = np.zeros_like(delta)
        np.round(nf * delta, 0, ptable)
        ptable = 1 + ptable
        # compute pointer to input data
        pdata = (np.abs(pdata) // 1 * np.sign(pdata)) + ns / 2 - 1
        # pad input data with end values
        xpad = np.ones((ns - 1 + x.size))
        xpad[: ns // 2 - 1] = x[0]
        xpad[ns // 2 :] = x[-1]
        xpad[ns // 2 - 1 : -ns // 2] = x

        ij = (ptable == nf + 1).nonzero()
        ptable[ij] = np.ones(len(ij[0]))
        pdata[ij] = pdata[ij] + 1
        pdata_int = pdata.astype(numba.int64)
        ptable_int = ptable.astype(numba.int64) - 1
        for k, l in enumerate(ind[0]):
            i = pdata_int[k] - ns // 2 - 1
            j = pdata_int[k] + ns // 2
            # dot product
            y[l] = np.sum(xpad[i + 1 : j] * sinc_table[:, ptable_int[k]])
    return y


def sinc_interp1d(
    x: npt.NDArray[np.float_], y: npt.NDArray[np.float_], sinc_tab=None
) -> Callable:
    """1D sinc interpolation following the format of scipy interp1d

    Args:
        x:
        y:
        sinc_tab: Improve performance by providing a pre-computed sinc table.

    Returns:
        interpolation function for y(x)
    """
    if sinc_tab is None:
        sinc_tab = make_sinc_table(ns=12, nf=55)

    def _interp(u: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        i = fast_sinc_interp(y, x, u, sinc_tab)
        i[u < x.min()] = np.nan
        i[u > x.max()] = np.nan
        return i

    return _interp
