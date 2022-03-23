from typing import Tuple, Sequence

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d


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


def calc_twt_range(twt_vol: xr.DataArray, srate: float = 0.001):
    """Calculate the twt range so we convieniently land on whole numbers start from zero and
    incrementing at dt.

    Args:
        twt_vol:
        srate:
    """
    srate_str = str(srate)
    dp_in_srate = len(srate_str) - srate_str.find(".") - 1
    minround = np.around(twt_vol.min().values - srate, dp_in_srate)
    maxround = np.around(twt_vol.max().values + srate, dp_in_srate)

    twt_range = np.arange(minround, maxround, srate)
    return twt_range


def time_convert_vol(
    twt_vol: xr.DataArray,
    depth_ds: xr.Dataset,
    mapping_dims: Sequence[str] = ("xline", "iline"),
    interpolation_kind="linear",
) -> xr.Dataset:
    """Convert depth volumes to TWT

    Args:
        twt_vol: A DataArray with the twt values for depth
        depth_ds: A Dataset with the properties to depth convert. Has same dims s `twt_vol`
        mapping_dims: The dimensions over which to map the funciton
        interpolation_kind: As for scipy.interpolate.interp1d(kind=)

    Returns:
        The properties of `depth_ds` converted to twt using `twt_vol`
    """
    if "twt" in depth_ds.data_vars:
        raise ValueError("twt variable cannot be in `depth_ds`")

    for dim in mapping_dims:
        assert dim in twt_vol.dims
        assert dim in depth_ds.dims

    twt_range = calc_twt_range(twt_vol)

    template = depth_ds.drop_dims("depth")
    template["twt"] = (("twt",), twt_range)
    depth_ds["twt_trace"] = twt_vol

    def _trace_time_conv_mapper(trace, twt_range=None):
        out = trace.drop_dims("depth").copy()
        out["twt"] = (("twt",), twt_range)
        for var in trace.data_vars:
            out[var] = (
                ("twt",),
                interp1d(
                    trace.twt_trace.values,
                    trace[var].values,
                    bounds_error=False,
                    fill_value=np.nan,
                    kind=interpolation_kind,
                )(twt_range),
            )
        return out

    def _blocks_time_conv_mapper(ds):
        stack = depth_ds.stack({"trace": mapping_dims})
        # preserve_dim_order = tuple(key for key in ds.dims)
        twt_block = (
            stack.groupby("trace")
            .map(_trace_time_conv_mapper, twt_range=twt_range)
            .unstack("trace")
        )
        return twt_block  # .transpose(*preserve_dim_order)

    twt_ds = depth_ds.map_blocks(_blocks_time_conv_mapper, template=template)
    return twt_ds
