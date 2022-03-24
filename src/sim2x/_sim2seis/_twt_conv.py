from typing import Tuple, Sequence, Literal, Union
from functools import partial


import numpy as np
from numpy import typing as npt
import xarray as xr
from scipy.interpolate import interp1d

from ._sinc import sinc_interp1d, make_sinc_table


def time_integral(vp, depth_dim="depth"):
    """Get the time integral from a vp Dataset with axis `depth`"""
    # display(v
    trace_dims = [dim for dim in vp.dims if dim != depth_dim]
    depth_diff = np.diff(vp.depth)
    depth_diff = np.pad(depth_diff, (0, 1), mode="edge")

    stack = vp.stack({"trace": trace_dims})
    # display(stack)
    def _twt_integral(trace):
        twt = np.cumsum(2 * depth_diff / trace.values)
        return twt

    twt_integral = stack.groupby("trace").map(
        _twt_integral,
    )
    twt_integral.name = "twt"
    return twt_integral.unstack("trace")


def time_convert_surface(
    twt_vol: xr.DataArray,
    depth_surf: xr.DataArray,
    mapping_dims: Sequence[str] = ("xline", "iline"),
) -> xr.DataArray:
    for dim in mapping_dims:
        assert dim in twt_vol.dims
        assert dim in depth_surf.dims

    stack = twt_vol.stack({"trace": mapping_dims})
    stack_depth_surf = depth_surf.stack({"trace": mapping_dims})

    stack["depth_surf"] = stack_depth_surf

    def _internal_interp(trace):
        twt = interp1d(
            trace.depth,
            trace.values,
        )(trace.depth_surf)
        trace["twt_surf"] = twt
        return trace

    twt_surf = stack.groupby("trace").map(_internal_interp)
    return twt_surf.unstack("trace")["twt_surf"]


def peg_time_interval(
    twt_vol: xr.DataArray,
    depth_surf: xr.DataArray,
    twt_surf: xr.DataArray,
    mapping_dims: Sequence[str] = ("xline", "iline"),
) -> xr.Dataset:
    """
    Shifts twt_vol to create such that the depth conversion of `depth_surf` with `twt_vol` matches
    `twt_surf`

    Args:
        twt_vol: TWT volume in depth
        depth_surf: The depth surface at `twt_surf`
        twt_surf: The TWT surface at `depth_surf`
    """
    for dim in mapping_dims:
        assert dim in twt_vol.dims
        assert dim in depth_surf.dims
        assert dim in twt_surf.dims

    assert depth_surf.dims == twt_surf.dims

    twt_vol["twt_interp"] = time_convert_surface(twt_vol, depth_surf, mapping_dims)
    twt_interp = twt_vol.copy() + (twt_surf - twt_vol["twt_interp"])
    return twt_interp


def calc_domain_range(domain_vol: xr.DataArray, srate: float = 0.001):
    """Calculate the domain range so we convieniently land on whole numbers starting from near the limit of our data and
    incrementing at srate.

    Args:
        vol: Volume mapping dims of `domain_vol` to target domain
        srate: The sample rate in the target domain

    Returns:
        appropriate domain_stick for domain_vol to convert to
    """
    srate_str = str(srate)
    dp_in_srate = len(srate_str) - srate_str.find(".") - 1
    minround = np.around(domain_vol.min().values - srate, dp_in_srate)
    maxround = np.around(domain_vol.max().values + srate, dp_in_srate)

    domain_range = np.arange(minround, maxround, srate)
    return domain_range


def domain_convert_vol(
    domain_vol: xr.DataArray,
    ds: xr.Dataset,
    mapping_dims: Sequence[str] = ("xline", "iline"),
    interpolation_kind="linear",
    to_dim: str = "twt",
    from_dim: str = "depth",
    domain_vol_direction: Literal["forward", "reverse"] = "forward",
    output_samples: Union[None, npt.ArrayLike] = None,
) -> xr.Dataset:
    """Convert depth volumes to TWT

    Args:
        domain_vol: A DataArray with the `to_dim` values for `from_dim`
        ds: A Dataset with the properties to depth convert. Has same dims s `twt_vol`
        mapping_dims: The dimensions over which to map the funciton
        interpolation_kind: As for scipy.interpolate.interp1d(kind=) or `sinc`. `sinc` uses internal algorithm
        to_dim: The domain dimension for the current input `ds`
        from_dim: The domain dimension to map to
        domain_vol_direction: If domain volume maps `to_dim` to `from_dim` use forward, otherwise `reverse.
        output_samples: Specify the samples at which ds will be interpolated to using `domain_vol`. Should be a 1D array. Ignored if `domain_vol_direction="reverse"`

    Returns:
        The properties of `ds` converted to twt using `twt_vol`
    """
    if to_dim in ds.data_vars:
        raise ValueError(f"{to_dim} variable cannot be in `ds`")

    for dim in mapping_dims:
        assert dim in domain_vol.dims
        assert dim in ds.dims

    if output_samples is None and domain_vol_direction == "forward":
        # going forward, don't know zstick of output
        domain_range = calc_domain_range(domain_vol)
    elif output_samples is not None and domain_vol_direction == "forward":
        # output has been explicitly specified
        domain_range = np.atleast_1d(output_samples)
        assert len(domain_range.shape) == 1
    elif domain_vol_direction == "reverse":
        # use the z stick of the domain vol
        domain_range = domain_vol[to_dim].values

    template = ds.drop_dims(from_dim)
    template[to_dim] = ((to_dim,), domain_range)
    ds[f"{to_dim}_trace"] = domain_vol

    if interpolation_kind == "sinc":
        sinc_tab = make_sinc_table(16, 100)
        interpolator = partial(sinc_interp1d, sinc_tab=sinc_tab)
    else:
        interpolator = partial(
            interp1d, bounds_error=False, fill_value=np.nan, kind=interpolation_kind
        )
    # print(ds)

    def _trace_time_conv_mapper(trace, domain_range=None):
        out = trace.drop_dims(from_dim).copy()
        out[to_dim] = ((to_dim,), domain_range)

        for var in trace.data_vars:
            if not from_dim in trace[var].dims:
                continue
            if domain_vol_direction == "forward":
                out[var] = (
                    (to_dim,),
                    interpolator(
                        trace[f"{to_dim}_trace"].values,
                        trace[var].values,
                    )(domain_range),
                )
            else:
                out[var] = (
                    (to_dim,),
                    interpolator(
                        trace[f"{from_dim}"].values,
                        trace[var].values,
                    )(trace[f"{to_dim}_trace"].values),
                )
        return out

    def _blocks_time_conv_mapper(ds):
        stack = ds.stack({"trace": mapping_dims})
        # preserve_dim_order = tuple(key for key in ds.dims)
        block = (
            stack.groupby("trace")
            .map(_trace_time_conv_mapper, domain_range=domain_range)
            .unstack("trace")
        )
        return block  # .transpose(*preserve_dim_order)

    dom_ds = ds.map_blocks(_blocks_time_conv_mapper, template=template)
    return dom_ds
