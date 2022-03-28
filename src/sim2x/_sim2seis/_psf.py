from typing import Type, Tuple, Union
import copy

import numpy as np
import xarray as xr

from scipy import fftpack
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

from ..utils.tools import safe_divide
from ._wavelets import Wavelet


def amp_spec(wavelet: Type[Wavelet], n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the amplitude spectrum of a wavelet

    Args:
        wavelet: Input wavelet.
        n: Samples for FFT, will pad the function.

    Returns:
        Frequencies and complex spectrum.
    """
    freqs = fftpack.fftshift(fftpack.fftfreq(n, wavelet.dt))[n // 2 :]
    return freqs, fftpack.fftshift(np.abs(fftpack.fft(wavelet.amp, n)))[n // 2 :]


def full_spec(wavelet: Type[Wavelet], n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the full spectrum of a wavelet

    Args:
        wavelet: Input wavelet.
        n: Samples for FFT, will pad the function.

    Returns:
        tuple: Frequencies and complex spectrum.
    """
    freqs = fftpack.fftshift(fftpack.fftfreq(n, wavelet.dt))
    return freqs, fftpack.fftshift(fftpack.fft(wavelet.amp, n))


def lowest_fdom_n(wavelet: Type[Wavelet], tol: float = 0.25, maxn: int = 15) -> int:
    """Finds the smallest number of samples needed  (2**n) to accurately represent a signal in the frequency
    domain. The acceptable error tolerance can be adjusted.

    Minimum n is always 6 or 2**6 = 64 samples.

    Args:
        wave: The input wavelet
        tol: The error tolerance for convergence. Defaults to 0.1.
        maxn: If convergence is not achieved by maximum n an error will be raised.

    Raises:
        ValueError: If convergence not reached.

    Returns:
        The n which accurately represents the wavelet in the frequency domain.
    """
    n = 6
    # normalise so tolerance is good
    nor_wave = copy.deepcopy(wavelet)
    nor_wave.set_wavelet(amp=nor_wave.amp / np.max(np.abs(nor_wave.amp)))
    df, freq = amp_spec(nor_wave, 2**n)
    error = 1
    while error >= tol:
        n += 1
        if n > maxn:
            break
        temp_df, temp_freq = amp_spec(nor_wave, 2**n)
        func = interp1d(df, freq, bounds_error=False, fill_value=0)
        error = np.sum(np.sqrt(np.power(temp_freq - func(temp_df), 2)))
        df = temp_df.copy()
        freq = temp_freq.copy()
    if error <= tol:
        return n
    else:
        raise ValueError(
            "Convergence to tolerance did not occur. "
            f"Final converges error was {error} with tolerance {tol}"
        )


def zero_grad(wavelet: Type[Wavelet], n: int, tol: float = 1e-3) -> np.ndarray:
    """Find when amplitude spectrum stops changing greater than tol.

    Used to truncate frequency domain which typically requires high input sampling.

    Args:
        wavelet: Input wavelet
        n: 2**n is size of wavelet.
        tol: Tolerance to signify no change. Defaults to 1E-3.

    Returns:
        frequency range after convergence
    """
    df, freq = amp_spec(wavelet, 2**n)
    grad = np.gradient(freq, df)
    slopen = np.arange(0, grad.size, 1)[np.abs(grad) >= tol][-1]
    return slopen


def analytic_illumination_kdom(
    wavelet: Type[Wavelet],
    dil: float,
    dxl: float,
    vavg: float,
    angi: Union[float, Tuple[float, float]],
    angx: Union[float, Tuple[float, float]],
    size=(32, 32, 128),
) -> xr.DataArray:
    """Create an analytical illumination vector set in the wavenumber domain based upon idealised
    maximum illumination angles to an input

    Velocity is used to convert wavelet to depth/length so it can be converted into the wavenumber domain.

    Args:
        wave: Input wavelet
        dil: The iline spacing of samples (m).
        dxl: The xline spacing of samples (m).
        vavg: Average velocity (m/s) used to transform wavelet form TWT to depth.
        angi: The inline angle range, if float assumes minimum angle is zero.
        angx: The xline angle range, if float assumes minimum angle is zero.
        size: The size of the illumination representation

    Returns:
        The K-domain representation of the illumination functions modified for angles and wavelet
    """

    def ina2off(ang, height):
        return height * np.tan(ang)

    ang_range = False
    try:
        ai_min, angi = angi
        ax_min, angx = angx
        ang_range = True
    except TypeError:
        pass

    req_n = lowest_fdom_n(wavelet)
    n = 2**req_n
    maxk_n = zero_grad(wavelet, req_n)
    freq, spec = full_spec(wavelet, n)
    # get the vertical wavenumber sampling
    dz = wavelet.dt / 2 * vavg
    kz = fftpack.fftshift(fftpack.fftfreq(n, dz))
    max_kz = kz.size

    spec_func = interp1d(kz, np.abs(spec), bounds_error=False, fill_value=0)

    kx = fftpack.fftshift(fftpack.fftfreq(n, dil))
    ky = fftpack.fftshift(fftpack.fftfreq(n, dxl))

    try:
        sx, sy, sz = size
    except TypeError:
        sx = sy = sz = size

    # create mesh in k-wavenumber domain
    kx = kx[:: kx.size // sx]
    ky = ky[:: ky.size // sy]
    kz = kz[:: kz.size // sz]
    kx_, ky_, kz_ = np.meshgrid(kx, ky, kz)

    # calculate distance from origin
    out = np.sqrt(np.power(kx_, 2) + np.power(ky_, 2) + np.power(kz_, 2))
    # apply wavelet
    out = spec_func(out)
    # create dip mask
    # a and b of cone
    a = ina2off(np.radians(angi), max_kz)
    b = ina2off(np.radians(angx), max_kz)
    incone = np.logical_and(
        np.power(kx_ / a, 2) + np.power(ky_ / b, 2) - np.power(kz_ / max_kz, 2) < 0,
        np.abs(kz_) < max_kz,
    )
    incone = np.logical_and(incone, kz_ <= 0)

    out[~incone] = 0

    if ang_range:
        a_inner = ina2off(np.radians(ai_min), max_kz)
        b_inner = ina2off(np.radians(ax_min), max_kz)

        inner_cone = np.logical_and(
            np.power(safe_divide(kx_, a_inner), 2)
            + np.power(safe_divide(ky_, b_inner), 2)
            - np.power(safe_divide(kz_, max_kz), 2)
            < 0,
            np.abs(kz_) < max_kz,
        )
        out[inner_cone] = 0

    ds = xr.DataArray(
        out,
        dims=["kx", "ky", "kz"],
        coords={"kx": (("kx"), kx), "ky": (("ky"), ky), "kz": (("kz"), kz)},
        name="fdom",
        attrs=dict(
            dil=dil,
            dxl=dxl,
            vavg=vavg,
            angi=angi,
            angx=angx,
        ),
    )
    return ds


def psf(
    wavelet: Type[Wavelet],
    dil: float,
    dxl: float,
    vavg: float,
    angi: Union[float, Tuple[float, float]],
    angx: Union[float, Tuple[float, float]],
    size=(32, 32, 128),
    gaussian_sigma: float = 0.5,
    twt: bool = False,
) -> xr.DataArray:
    """Create a point spread function PSF using analytical inputs

    Args:
        wave: Input wavelet
        dil: The iline spacing of samples (m)
        dxl: The xline spacing of samples (m)
        vavg: Average velocity (m/s) used to transform wavelet form TWT to depth
        angi: The inline angle range, if float assumes minimum angle is zero
        angx: The xline angle range, if float assumes minimum angle is zero
        size: The size of the illumination representation
        gaussian_simga: The size of the filter to use, no filter if None.
        twt: If true returns the vertical axis as twt (ms). Defaults to False.

    Returns:
        The point spread filter (wavelet)
    """
    ilum = analytic_illumination_kdom(wavelet, dil, dxl, vavg, angi, angx, size=size)
    kx = ilum.kx.values
    ky = ilum.ky.values
    kz = ilum.kz.values
    x = fftpack.fftshift(fftpack.fftfreq(kx.size, np.mean(np.diff(kx))))
    y = fftpack.fftshift(fftpack.fftfreq(ky.size, np.mean(np.diff(ky))))
    vert = fftpack.fftshift(fftpack.fftfreq(kz.size, np.mean(np.diff(kz))))

    if twt:
        vert = 2 * vert / vavg

    if gaussian_sigma is not None:
        ilum = gaussian_filter(ilum, (0.5))

    return xr.DataArray(
        fftpack.fftshift(
            fftpack.ifftn(fftpack.ifftshift(ilum.transpose("kx", "ky", "kz").data))
        ).real,
        dims=["x", "y", "z"],
        coords={"x": (("x"), x), "y": (("y"), y), "z": (("z"), vert)},
        name="psf",
        attrs=dict(
            dil=dil,
            dxl=dxl,
            vavg=vavg,
            angi=angi,
            angx=angx,
            sigma=gaussian_sigma,
            twt=twt,
        ),
    )
