"""Functions for interface reflectivity modelling.

This module contains functions for.

Zoeppritz full scattering matrix.


"""
from typing import Tuple, Literal
import numpy as np
import numba


@numba.njit(error_model="numpy")
def _denom_zdiv(a, b):
    """Helper function to avoid divide by zero in many areas.

    Args:
        a (array-like): Numerator
        b (array-like): Deominator

    Returns:
        a/b (int): Replace div0 by 0
    """
    c = np.divide(a, b)
    c[np.isinf(c)] = 0.0
    return c


@numba.njit(error_model="numpy")
def snellrr(
    thetai: float,
    vp1: np.ndarray,
    vs1: np.ndarray,
    vp2: np.ndarray,
    vs2: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Snell's reflection and refraction angles for a two layered half space.

    Args:
        thetai: downgoing p-wave angle of incidence for wavefront (radians)
        vp1: p-velocity layer 1
        vs1: s-velocity layer 1
        vp2: p-velocity layer 2
        vs2: s-veloicty layer 2

    Returns:
        returns a list of calculated angles using Snell's Law
            thetai = input angle of P-wave incidence and output angle P-wave reflection
            thetat = output angle of P-wave transmission
            phir   = output angle of S-wave reflection
            phit   = output angle of S-wave transmission
    """
    return (
        thetai,
        np.arcsin(_denom_zdiv(vp2 * np.sin(thetai), vp1)),
        np.arcsin(_denom_zdiv(vs1 * np.sin(thetai), vp1)),
        np.arcsin(_denom_zdiv(vs2 * np.sin(thetai), vp1)),
    )


def zoeppritzfull(
    thetai: float,
    vp1: np.ndarray,
    vs1: np.ndarray,
    rho1: np.ndarray,
    vp2: np.ndarray,
    vs2: np.ndarray,
    rho2: np.ndarray,
) -> np.ndarray:
    """Full Zoeppritz scattering matrix solution

    Args:
        thetai: P-wave angle of incidence for wavefront in radians
        vp1: p-velocity layer 1
        vs1: s-velocity layer 1
        rho1: density layer 1
        vp2: p-velocity layer 2
        vs2: s-veloicty layer 2
        rho2: density layer 2

    Returns:
        [Rp, Rs, Tp, Ts] of amplitudes for reflected and transmitted rays
    """
    ang = snellrr(thetai, vp1, vs1, vp2, vs2)
    p = _denom_zdiv(np.sin(thetai), vp1)
    c1 = 1 - 2 * np.square(vs1 * p)
    c2 = 1 - 2 * np.square(vs2 * p)
    c3 = 2 * rho1 * vs1 * vs1 * p
    c4 = 2 * rho2 * vs2 * vs2 * p
    M = np.array(
        [
            [-vp1 * p, -np.cos(ang[2]), vp2 * p, np.cos(ang[3])],
            [np.cos(ang[0]), -vs1 * p, np.cos(ang[1]), -vs2 * p],
            [
                c3 * np.cos(ang[0]),
                rho1 * vs1 * c1,
                c4 * np.cos(ang[1]),
                rho2 * vs2 * c2,
            ],
            [
                -rho1 * vp1 * c1,
                c3 * np.cos(ang[2]),
                rho2 * vp2 * c2,
                -c4 * np.cos(ang[3]),
            ],
        ]
    )
    Nu = np.array([[-1, -1, -1, -1], [1, 1, 1, 1], [1, 1, 1, 1], [-1, -1, -1, -1]])
    # following borrowed from bruges reflection.py by Wes Hamlyn
    M = np.moveaxis(M, [0, 1], [-2, -1])
    N = M * Nu
    Z = np.matmul(np.linalg.inv(M), N)
    return np.transpose(Z, axes=list(range(Z.ndim - 2)) + [-1, -2])


@numba.njit(error_model="numpy")
def zoeppritz_ponly(
    thetai: float,
    vp1: np.ndarray,
    vs1: np.ndarray,
    rho1: np.ndarray,
    vp2: np.ndarray,
    vs2: np.ndarray,
    rho2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the solution to an incident down-going P-wave ray.

    Args:
        thetai float: P-wave angle of incidence for wavefront in radians
        vp1: p-velocity layer 1
        vs1: s-velocity layer 1
        rho1: density layer 1
        vp2: p-velocity layer 2
        vs2: s-veloicty layer 2
        rho2: density layer 2

    Returns:
        (Rp, Rs, Tp, Ts) of amplitudes for reflected and transmitted rays
    """
    ang = snellrr(thetai, vp1, vs1, vp2, vs2)
    p = _denom_zdiv(np.sin(ang[0]), vp1)
    p2 = p * p
    a = rho2 * (1 - 2 * vs2**2 * p2) - rho1 * (1 - 2 * vs1**2 * p2)
    b = rho2 * (1 - 2 * vs2**2 * p2) + 2 * rho1 * vs1**2 * p2
    c = rho1 * (1 - 2 * vs1**2 * p2) + 2 * rho2 * vs2**2 * p2
    d = 2 * (rho2 * vs2**2 - rho1 * vs1**2)
    E = _denom_zdiv(b * np.cos(ang[0]), vp1) + _denom_zdiv(c * np.cos(ang[1]), vp2)
    F = _denom_zdiv(b * np.cos(ang[2]), vs1) + _denom_zdiv(c * np.cos(ang[3]), vs2)
    G = a - _denom_zdiv(d * np.cos(ang[0]) * np.cos(ang[3]), vs2 * vp1)
    H = a - _denom_zdiv(d * np.cos(ang[1]) * np.cos(ang[2]), vs1 * vp2)
    D = E * F + G * H * p2

    PdPu = _denom_zdiv(1, D) * (
        (_denom_zdiv(b * np.cos(ang[0]), vp1) - _denom_zdiv(c * np.cos(ang[1]), vp2))
        * F
        - (a + _denom_zdiv(d * np.cos(ang[0]), vp1) * _denom_zdiv(np.cos(ang[3]), vs2))
        * H
        * p2
    )
    PdPd = _denom_zdiv(2 * rho1 * np.cos(ang[0]), vp1) * F * _denom_zdiv(vp1, (vp2 * D))
    PdSu = (
        _denom_zdiv(-2 * np.cos(ang[0]), vp1)
        * (
            a * b
            + _denom_zdiv(c * d * np.cos(ang[1]), vp2)
            * _denom_zdiv(np.cos(ang[3]), vs2)
        )
        * p
        * _denom_zdiv(vp1, vs2 * D)
    )
    PdSd = _denom_zdiv(2 * rho1 * np.cos(ang[0]), vp1) * _denom_zdiv(
        H * p * vp1, vs2 * D
    )
    return (PdPu, PdPd, PdSu, PdSd)


@numba.njit(error_model="numpy")
def zoeppritz_pdpu_only(
    thetai: float,
    vp1: np.ndarray,
    vs1: np.ndarray,
    rho1: np.ndarray,
    vp2: np.ndarray,
    vs2: np.ndarray,
    rho2: np.ndarray,
):
    """Calculates the solution to an incident down-going P-wave ray and upgoing P-wave only.

    Args:
        thetai float: P-wave angle of incidence for wavefront in radians
        vp1: p-velocity layer 1
        vs1: s-velocity layer 1
        rho1: density layer 1
        vp2: p-velocity layer 2
        vs2: s-veloicty layer 2
        rho2: density layer 2

    Returns:
        Rp
    """
    ang = snellrr(thetai, vp1, vs1, vp2, vs2)
    p = _denom_zdiv(np.sin(ang[0]), vp1)
    p2 = p * p
    a = rho2 * (1 - 2 * vs2**2 * p2) - rho1 * (1 - 2 * vs1**2 * p2)
    b = rho2 * (1 - 2 * vs2**2 * p2) + 2 * rho1 * vs1**2 * p2
    c = rho1 * (1 - 2 * vs1**2 * p2) + 2 * rho2 * vs2**2 * p2
    d = 2 * (rho2 * vs2**2 - rho1 * vs1**2)
    E = _denom_zdiv(b * np.cos(ang[0]), vp1) + _denom_zdiv(c * np.cos(ang[1]), vp2)
    F = _denom_zdiv(b * np.cos(ang[2]), vs1) + _denom_zdiv(c * np.cos(ang[3]), vs2)
    G = a - _denom_zdiv(d * np.cos(ang[0]) * np.cos(ang[3]), vs2 * vp1)
    H = a - _denom_zdiv(d * np.cos(ang[1]) * np.cos(ang[2]), vs1 * vp2)
    D = E * F + G * H * p2

    return _denom_zdiv(1, D) * (
        (_denom_zdiv(b * np.cos(ang[0]), vp1) - _denom_zdiv(c * np.cos(ang[1]), vp2))
        * F
        - (a + _denom_zdiv(d * np.cos(ang[0]), vp1) * _denom_zdiv(np.cos(ang[3]), vs2))
        * H
        * p2
    )


@numba.njit(error_model="numpy")
def calcreflp(
    vp1: np.ndarray,
    vs1: np.ndarray,
    rho1: np.ndarray,
    vp2: np.ndarray,
    vs2: np.ndarray,
    rho2: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Calculates the reflectivity parameters of an interface.

    Args:
        vp1: p-velocity layer 1
        vs1: s-velocity layer 1
        rho1: density layer 1
        vp2: p-velocity layer 2
        vs2: s-veloicty layer 2
        rho2: density layer 2

    Returns:
        [rVp, rVs, rrho, rVsVp, dVp, dVs, drho]
    """
    rVp = 0.5 * (vp1 + vp2)
    rVs = 0.5 * (vs1 + vs2)
    rrho = 0.5 * (rho1 + rho2)
    rVsVp = rVs / rVp
    dVp = vp2 - vp1
    dVs = vs2 - vs1
    drho = rho2 - rho1
    return (rVp, rVs, rrho, rVsVp, dVp, dVs, drho)


def bortfeld(
    thetai: float,
    vp1: np.ndarray,
    vs1: np.ndarray,
    rho1: np.ndarray,
    vp2: np.ndarray,
    vs2: np.ndarray,
    rho2: np.ndarray,
) -> np.ndarray:
    """Calculates the solution to the full Bortfeld equations (1961)

    These are approximations which work well when the interval velocity is well defined.

    Args:
        thetai: P-wave angle of incidence for wavefront in radians
        vp1: p-velocity layer 1
        vs1: s-velocity layer 1
        rho1: density layer 1
        vp2: p-velocity layer 2
        vs2: s-veloicty layer 2
        rho2: density layer 2

    Returns:
        Rp(thetai) P-wave reflectivity of angle theta
    """
    rVp, rVs, rrho, rVsVp, dVp, dVs, drho = calcreflp(vp1, vs1, rho1, vp2, vs2, rho2)
    Rp = dVp / (2 * rVp)
    Rrho = drho / (2 * rrho)
    k = (2 * rVs / rVp) ** 2
    R0 = Rp + Rrho
    Rsh = 0.5 * (dVp / rVp - k * drho / (2 * rrho) - 2 * k * dVs / rVs)
    return (
        R0
        + Rsh * np.sin(thetai) ** 2.0
        + Rp * (np.tan(thetai) ** 2) * (np.sin(thetai) ** 2)
    )


@numba.njit(error_model="numpy")
def akirichards(
    thetai: float,
    vp1: np.ndarray,
    vs1: np.ndarray,
    rho1: np.ndarray,
    vp2: np.ndarray,
    vs2: np.ndarray,
    rho2: np.ndarray,
    method: Literal["avseth", "ar"] = "avseth",
) -> np.ndarray:
    """Aki-Richards forumlation of reflectivity functions.

    Args:
        thetai: P-wave angle of incidence for wavefront in radians
        vp1: p-velocity layer 1
        vs1: s-velocity layer 1
        rho1: density layer 1
        vp2: p-velocity layer 2
        vs2: s-veloicty layer 2
        rho2: density layer 2
        method (Optional): Defaults to 'avseth' - avseth formulation
            'ar' - original aki-richards

    Returns:
        (numpy.ndarray): Rp(theta)
    """
    rVp, rVs, rrho, rVsVp, dVp, dVs, drho = calcreflp(vp1, vs1, rho1, vp2, vs2, rho2)
    ang = snellrr(thetai, vp1, vs1, vp2, vs2)
    ang_Pavg = (ang[0] + ang[1]) / 2
    if method == "avseth":
        W = 0.5 * drho / rrho
        X = 2 * rVs * rVs * drho / (vp1 * vp1 * rrho)
        Y = 0.5 * dVp / (rVp)
        Z = 4 * rVs * rVs * dVs / (vp1 * vp1 * rVs)
        return (
            W
            - X * np.sin(thetai) * np.sin(thetai)
            + Y / (np.cos(ang_Pavg) * np.cos(ang_Pavg))
            - Z * np.sin(thetai) * np.sin(thetai)
        )
    elif method == "ar":
        return (
            0.5 * (dVp / rVp + drho / rrho)
            + 0.5
            * (dVp / rVp - 4 * rVsVp * rVsVp * (drho / rrho + 2 * dVs / rVs))
            * thetai
            * thetai
        )


def shuey(
    thetai: float,
    vp1: np.ndarray,
    vs1: np.ndarray,
    rho1: np.ndarray,
    vp2: np.ndarray,
    vs2: np.ndarray,
    rho2: np.ndarray,
    mode: Literal["rtheta", "R0_G"] = "rtheta",
) -> np.ndarray:
    """Shuey approximation to the Aki-Richards equations.

    Args:
        thetai: P-wave angle of incidence for wavefront in radians
        vp1: p-velocity layer 1
        vs1: s-velocity layer 1
        rho1: density layer 1
        vp2: p-velocity layer 2
        vs2: s-veloicty layer 2
        rho2: density layer 2
        mode:  what to return 'rtheta' returns Rp(theta)
            'R0_G'   returns [R0,G] aka [A,B]

    Returns:
        if `mode='rtheta'` Rp(theta); if `mode='R0_G'` [R0, G]
    """
    rVp, rVs, rrho, rVsVp, dVp, dVs, drho = calcreflp(vp1, vs1, rho1, vp2, vs2, rho2)
    R0 = 0.5 * (dVp / rVp + drho / rrho)
    G = 0.5 * dVp / rVp - 2.0 * (rVs * rVs) / (rVp * rVp) * (
        drho / rrho + 2.0 * dVs / rVs
    )
    if mode == "rtheta":
        return R0 + G * np.sin(thetai) * np.sin(thetai)
    elif mode == "R0_G":
        return np.array([R0, G])
