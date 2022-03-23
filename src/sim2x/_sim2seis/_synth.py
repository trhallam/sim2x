"""Functions and classes necessary for synthetic seismic
modelling.

    e.g sim2seis

"""
from typing import Any, Literal, List, Sequence

import numpy as np
import xarray as xr
from numpy import typing as npt
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
import scipy.ndimage
from tqdm import tqdm

from ._interface import zoeppritz_pdpu_only, zoeppritz_ponly, akirichards
from etlpy.seismic.wavelets import psf
from etlpy.models._rem import get_dxdy


def nanfill(x, filter="mean", size=3):
    filters = {"mean": np.mean}
    xf = np.where(np.isnan(x), 0, x)
    xf = scipy.ndimage.generic_filter(xf, filters[filter], size=size)
    return np.where(np.isnan(x), xf, x)


def reflectivity(
    theta: float,
    velp: npt.ArrayLike,
    vels: npt.ArrayLike,
    rho: npt.ArrayLike,
    method: Literal["full", "zoep_pud", "ar"] = "full",
) -> npt.ArrayLike:
    """Reflectivity for theta from acoustic vectors

    Args:
        theta: P-wave incidence angle
        velp: P-wave velcoities
        vels: S-wave velocities
        rho: density values
        method: The modelling method to use

    Returns:
        interface reflectivity arrays (length N-1)
    """
    th = np.full_like(velp[:-1], theta)
    args = (th, velp[:-1], vels[:-1], rho[:-1], velp[1:], vels[1:], rho[1:])

    if method == "zoep_pud":
        refl = np.array(
            zoeppritz_pdpu_only(*args),
            dtype=[
                ("Rp", np.float_),
            ],
        )
    elif method == "full":
        a = zoeppritz_ponly(*args)
        refl = np.array(
            [v for v in zip(*a)],
            dtype=(
                [
                    ("Rp", np.float_),
                    ("Rs", np.float_),
                    ("Tp", np.float_),
                    ("Ts", np.float_),
                ]
            ),
        )
    elif method == "ar":
        refl = np.array(
            akirichards(*args),
            dtype=[
                ("Rp", np.float_),
            ],
        )

    for name in refl.dtype.names:
        refl[name] = np.where(np.isnan(refl[name]), 0, refl[name])
    return np.pad(refl, (1, 0))


def reflectivity_vol(
    ds: xr.Dataset,
    theta: float,
    mapping_dims: Sequence[str] = ("xline", "iline"),
) -> xr.Dataset:
    """Convert elastic properties "vp", "vs" and "density" to reflectivity

    Args:
        ds: A Dataset with the properties to depth convert. Has same dims s `twt_vol`
        mapping_dims: The dimensions over which to map the funciton

    Returns:
        The properties of `depth_ds` converted to twt using `twt_vol`
    """

    for dim in mapping_dims:
        assert dim in ds.dims

    def _reflectivity_mapper(trace):
        trace[f"refl_{theta}"] = (
            ("twt"),
            reflectivity(
                theta,
                trace.vp.values,
                trace.vs.values,
                trace.density.values,
                method="zoep_pud",
            )["Rp"],
        )
        return trace

    def _blocks_reflectivity_mapper(ds):
        stack = ds.stack({"trace": mapping_dims})
        preserve_dim_order = tuple(key for key in ds.dims)
        refl_block = stack.groupby("trace").map(_reflectivity_mapper).unstack("trace")

        return refl_block.transpose(*preserve_dim_order)

    refl_ds = ds.map_blocks(_blocks_reflectivity_mapper, template=ds)
    return refl_ds


def convolution1d_vol(
    ds: xr.Dataset,
    reflectivity_key: str,
    wavelet_amp,
    mapping_dims: Sequence[str] = ("xline", "iline"),
) -> xr.Dataset:
    """Convert elastic properties "vp", "vs" and "density" to reflectivity using 1d convolution and a wavelet

    Wavelet must have same sample rate as seismic twt

    Args:
        ds: A Dataset with the properties to depth convert. Has same dims s `twt_vol`
        mapping_dims: The dimensions over which to map the funciton

    Returns:
        The properties of `depth_ds` converted to twt using `twt_vol`
    """

    for dim in mapping_dims:
        assert dim in ds.dims

    def _conv_mapper(trace):
        trace["amp"] = (
            ("twt"),
            fftconvolve(trace[reflectivity_key].values, wavelet_amp, mode="same"),
        )
        return trace

    def _blocks_conv_mapper(ds):
        stack = ds.stack({"trace": mapping_dims})
        preserve_dim_order = tuple(key for key in ds.dims)
        refl_block = stack.groupby("trace").map(_conv_mapper).unstack("trace")

        return refl_block.transpose(*preserve_dim_order)

    seis_ds = ds.map_blocks(_blocks_conv_mapper, template=ds)
    return seis_ds


def _convolution_psf_line_inner(
    vp, vs, rho, twt, refl_method=None, th=None, subtwt=None, pbar=None, intp_kwargs={}
):
    reflc_ = np.zeros((vp.shape[0], subtwt.size))
    for j, (vp_, vs_, rho_, twt_) in enumerate(zip(vp, vs, rho, twt)):
        if np.all(np.isnan(vp_)):
            # empty TWT
            reflc_[j, :] = 0.0
        else:
            # interpolate values to TWT domain
            vp_subtwt = interp1d(twt_, vp_, **intp_kwargs)(subtwt)
            vs_subtwt = interp1d(twt_, vs_, **intp_kwargs)(subtwt)
            rho_subtwt = interp1d(twt_, rho_, **intp_kwargs)(subtwt)
            reflc_[j, :-1] = reflectivity(
                th, vp_subtwt, vs_subtwt, rho_subtwt, method=refl_method
            )
        if pbar is not None:
            pbar.update()
    return reflc_


def convolution_psf(
    dataset,
    wavelet,
    theta,
    vp_var,
    vs_var,
    rho_var,
    twt_var,
    twt=None,
    conv_dt=1,
    silent=False,
    refl_method="zoep_pud",
    maximum_offsets=None,
    psf_size=(64, 64, 128),
    dask_client=None,
):
    """Perform 1D convolution on a 3D xarray dataset.

    Assumes model is in depth rather than time. Although you can set twt_var to
    be the k dimension coordinate.

    Args:
        dataset (xarray.Dataset): Should have dimensions (i, j, k), easiest to
         get from etlpy.models.EarthModel
        wavelet (etlpy.seismic.Wavelet): wavelet to use for convultion
        theta (float/list[float]): A float of list of float angles to perform
            convultion modelling over.
        vp_var (str): P-velocity variable in dataset to use for modelling.
        vs_var (str): S-velocity variable in dataset to use for modelling.
        rho_var (str): Density variable in dataset to use for modelling.
        twt_var (str): TWT variable in dataset to use for zstick conversion.
        twt (array-like): Specify an out twt sampling.
        conv_dt (float): The sample rate at which to perform convolution.
        silent (bool, optional): Turn off the progress bar. Defaults to False
        refl_method (str, optional): The reflectivity calculatio nmethod.
            Choose from 'zoep_pud' and 'ar'. Defaults to 'zoep_pud'.
        maximum_offset (tuple, optional): A tuple of maximum inline anx xline offsets (m) to
            constrain the PSF illumination.

    Returns:
        (array-like): Synthetic trace volume.
    """
    try:
        ni, nj, nk = dataset.dims.values()
    except ValueError:
        raise ValueError(f"expect dimensions (i, j, k) but got {dataset.dims}")

    theta = np.atleast_1d(theta)
    angles = theta.copy()
    theta = np.deg2rad(theta)

    if twt is None:
        vert_size = dataset.vert.size
        twt_min = dataset[twt_var].min()
        twt_max = dataset[twt_var].max()
        # twt_stick = np.linspace(twt_min, twt_max + conv_dt, vert_size)
        twt_sticks = dataset[twt_var].values
        subtwt = np.arange(twt_min, twt_max + conv_dt, conv_dt)
    else:
        vert_size = twt.size
        # create a block twt array -> this could potentially be replaced by a sparse array
        twt_sticks = (
            np.array([np.asarray(twt)]).repeat(ni * nj, axis=0).reshape(ni, nj, -1)
        )
        subtwt = np.arange(twt_sticks.min(), twt_sticks.max(), conv_dt)

    reflc_ = np.zeros((ni, nj, subtwt.size))  # preallocate output memory
    psfc_ = np.zeros((ni, nj, subtwt.size))

    # psf setup
    avgz = np.mean(dataset.vert.values)
    if maximum_offsets is not None:
        max_il, max_xl = maximum_offsets
        max_il = np.degrees(np.arctan(max_il / avgz))
        max_xl = np.degrees(np.arctan(max_xl / avgz))
    else:  # no limit
        max_il = 90
        max_xl = 90

    # limit the maximum angle of the psf based upon the surface acquisition patch size
    if angles.size > 1:
        angi = min(angles[-1], max_il)
        angx = min(angles[-1], max_xl)
    else:
        angi = min(angles, max_il)
        angx = min(angles, max_xl)

    dil, dxl = get_dxdy(dataset)

    # wavelet needs to match the psf sampling
    wavelet = wavelet.copy()
    wavelet.as_miliseconds()
    if wavelet.dt != conv_dt:
        wavelet.resample(conv_dt)
    wavelet.as_seconds()

    # create the psf, use any vavg because we convert back to twt anyway.
    the_psf = psf(wavelet, dil, dxl, 3000, angi, angx, twt=True, size=psf_size)

    intp_kwargs = {"kind": "linear", "bounds_error": False, "assume_sorted": True}

    # loop angles to create reflectivities which are convolved with the psf, then summed together
    tqdm_th_pbar = tqdm(total=theta.size, desc="Convolving Angle", disable=silent)
    for th in theta.ravel():
        if dask_client is None:
            tqdm_rfl_pbar = tqdm(
                total=ni * nj, desc="Calculating Reflectivity", disable=silent
            )
            for i in dataset.iind.values:
                trace_s_ = np.s_[i, :, :]
                reflc_[i, :, :] = _convolution_psf_line_inner(
                    dataset[vp_var][trace_s_].values,
                    dataset[vs_var][trace_s_].values,
                    dataset[rho_var][trace_s_].values,
                    dataset[twt_var][trace_s_].values,
                    subtwt=subtwt,
                    refl_method=refl_method,
                    th=th,
                    pbar=tqdm_rfl_pbar,
                    intp_kwargs=intp_kwargs,
                )
                # for j in dataset.jind.values:
                #     trace_s_ = np.s_[i, j, :]
                #     if np.all(np.isnan(dataset[vp_var][trace_s_].values)):
                #         # empty TWT
                #         psfc_[i, j, :] = 0.0
                #     else:
                #         # if twt is None:
                #         #     twt_stick = dataset[twt_var][trace_s_].values
                #         #     subtwt = np.arange(twt_stick.min(), twt_stick.max(), conv_dt)

                #         # interpolate values to TWT domain
                #         vp_subtwt = interp1d(
                #             dataset[twt_var][trace_s_].values,
                #             dataset[vp_var][trace_s_].values,
                #             **intp_kwargs,
                #         )(subtwt)

                #         vs_subtwt = interp1d(
                #             dataset[twt_var][trace_s_].values,
                #             dataset[vs_var][trace_s_].values,
                #             **intp_kwargs,
                #         )(subtwt)

                #         rho_subtwt = interp1d(
                #             dataset[twt_var][trace_s_].values,
                #             dataset[rho_var][trace_s_].values,
                #             **intp_kwargs,
                #         )(subtwt)

                #         reflc = reflectivity(
                #             th, vp_subtwt, vs_subtwt, rho_subtwt, method=refl_method
                #         )
                #         reflc_[i, j, :-1] = reflc
            tqdm_rfl_pbar.close()
        else:
            futures = dask_client.map(
                _convolution_psf_line_inner,
                [dataset[vp_var][i, :, :].values for i in dataset.iind.values],
                [dataset[vs_var][i, :, :].values for i in dataset.iind.values],
                [dataset[rho_var][i, :, :].values for i in dataset.iind.values],
                [dataset[twt_var][i, :, :].values for i in dataset.iind.values],
                subtwt=subtwt,
                refl_method=refl_method,
                th=th,
                batch_size=10,
                resources={"process": 1},
                key="psf_reflectivity",
            )
            secede()
            results = dask_client.gather(futures)
            rejoin()
            reflc_[:, :, :] = np.concatenate(results)
        psfc_ = (
            fftconvolve(
                reflc_,
                the_psf.psf.values,
                "same",
            )
            + psfc_
        )
        tqdm_th_pbar.update()
    tqdm_th_pbar.close()
    psfc_ = psfc_ / np.size(theta.ravel())

    del reflc_  # free up some memory
    synth = np.zeros((ni, nj, vert_size))  # preallocate output memory
    for i in dataset.iind.values:
        for j in dataset.jind.values:
            # don't interpolate blank traces
            if np.all(np.isnan(twt_sticks[i, j, :])):
                synth[i, j, :] = 0.0
            else:
                synth[i, j, :] = interp1d(
                    subtwt,
                    psfc_[i, j, :],
                    kind="cubic",
                    bounds_error=False,
                    fill_value=0.0,
                    assume_sorted=True,
                )(twt_sticks[i, j, :])

    return synth
