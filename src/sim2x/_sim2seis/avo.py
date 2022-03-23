"""Method for calculating the AVO Intercept and Gradient

Returns:
    [type]: [description]
"""

def akirichards(vp1, vs1, rho1, vp2, vs2, rho2):
    """Aki-Richards linearisation of the Zoeppritz equations for A, B and C.

    R(th) ~ A + Bsin2(th) + Csin2(th)tan2(th)

    A = 0.5 (dVP/VP + dRho/rho)

    B = dVp/2Vp - 4*(Vs/Vp)**2 * (dVs/Vs) - 2*(Vs/Vp)**2 * (dRho/rho)

    C = dVp/2*Vp

    Args:
        vp1, vs1, vp2, vs2 (array-like [MxN]): velocities for 2 halfspaces
        rho1, rho2 (array-like [MxN]): densities for 2 halfspaces

    Returns:
        (numpy.ndarray): Rp(theta)
    """
    dvp = vp2 - vp1
    dvs = vs2 - vs1
    drho = rho2 - rho1
    vp = (vp1 + vp2)/2.0
    vs = (vs1 + vs2)/2.0
    rho = (rho1 + rho2)/2.0
    k = (vs/vp)**2
    avo_a = 0.5 * (dvp/vp + drho/rho)
    avo_b = dvp/(2*vp) - 4*k*(dvs/vs) - 2*k*(drho/rho)
    avo_c = dvp/(2*vp)
    return avo_a, avo_b, avo_c
