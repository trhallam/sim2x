from typing import Tuple, List, Union, Sequence
from functools import reduce

import pandas as pd
import numpy as np
import xarray as xr
from scipy.spatial import Delaunay

from dask.distributed import Client

from eclx._grid import _xcorn_names, _ycorn_names, _zcorn_names


def get_active_cell_range(
    xyzcorn_df: pd.DataFrame,
) -> Tuple[float, float, float, float, float, float]:
    """Get the xyz ranges of the active cells in the grid

    Args:
        xyzcorn_df: This is the xyzcorn df of an eclx.EclDeck

    Returns:
        The min and max coordinates in each dimension
            minx, maxx, miny, maxy, minz, maxz
    """
    active = xyzcorn_df.query("active >= 0")
    minx = active[_xcorn_names()].min().min()
    maxx = active[_xcorn_names()].max().max()
    miny = active[_ycorn_names()].min().min()
    maxy = active[_ycorn_names()].max().max()
    minz = active[_zcorn_names()].min().min()
    maxz = active[_zcorn_names()].max().max()
    return minx, maxx, miny, maxy, minz, maxz


def remove_collapsed_cells(df: pd.DataFrame) -> pd.DataFrame:
    # use query to remove cells with no height
    query = " & ".join(
        [f"{a} != {b}" for a, b in zip(_zcorn_names()[:4], _zcorn_names()[4:])]
    )
    return df.query(query)


def get_points_gi(xyzcorn_df: pd.DataFrame, points: np.ndarray) -> np.ndarray:
    """Return the global index (gi) for a set of points in XYZ

    Args:
        xyzcorn_df:
        points (array-like): A list of points of length n containing
            sub-arrays of length 3 (X, Y, Z)
    Returns:
        Array of length n global index values corresponding to points.
            If no cell contains a given point, -1 is returned.
    """
    names: List[str] = list(
        reduce(
            lambda x, y: x + y,
            zip(_xcorn_names(), _ycorn_names(), _zcorn_names()),
            tuple(),
        )
    )
    # remove collapsed cells
    _df = remove_collapsed_cells(xyzcorn_df)

    ind_log = np.full_like(points.shape[0], -1)
    # iterate cells instead of points using mask to find where in
    for ind, cell in _df.iterrows():
        delaunay = Delaunay(cell[names].values.reshape(8, 3))
        points_mask = np.logical_and(delaunay.find_simplex(points) >= 0, ind_log == -1)
        ind_log = np.where(points_mask, ind, ind_log)
    return ind_log


def filter_index_xyzcorn2loc(
    xyzcorn_df: pd.DataFrame, xloc: float, yloc: float, atol: float = 0
) -> np.ndarray:
    """Return an index filter of a xyzcorn df based on whether the data is near xloc/yloc"""
    xloc_mpc = np.min(xloc) - atol
    xloc_ppc = np.max(xloc) + atol
    yloc_mpc = np.min(yloc) - atol
    yloc_ppc = np.max(yloc) + atol
    xnames = _xcorn_names()
    ynames = _ycorn_names()
    masks = np.vstack(
        [
            xyzcorn_df[xnames].min(axis=1).values <= xloc_ppc,
            xyzcorn_df[xnames].max(axis=1).values >= xloc_mpc,
            xyzcorn_df[ynames].min(axis=1).values <= yloc_ppc,
            xyzcorn_df[ynames].max(axis=1).values >= yloc_mpc,
        ]
    )
    return np.all(masks, axis=0)


def extract_xy_gilog(
    xyzcorn_df: pd.DataFrame,
    xloc: float,
    yloc: float,
    zloc: Union[None, np.ndarray] = None,
    srate: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a vertical or arbitrary log from the model and return a DataFrame
    of depth vs global index (gi) logs. Logs are extracted over the minimum
    to maximum depth of the entire model if z is not set. The global index
    logs can be used to extract the properties from cells in eclx.EclDeck.data
    to make a real log.
    Args:
        xyzcorn_df: xyzcorn DataFrame from `eclx`
        xloc (array_like): X Cartesian locations to perform the extraction
        yloc (array_like): Y Cartesian locations to perform the extraction
        zloc (array_like, optional): Defaults to None and z is assumed to be vertical.
            If zloc is specified srate will be ignored.
            The trace will be extracted along a vector path defined by x, y, z.
        srate (float): The sample rate of the log to be extracted (m). If interfaces=True
            this will be reduced to the number of interfaces, appropximate the k dimension of
            the grid plus 1 -   ~(k+1)

    Returns:
        zloc, gilog: zloc is calculated from range of input cells or as input, gilog is gi values at zloc for xloc/yloc

    Notes:
        The core code to establish whether or not a point lies within any given cell
        is based upon convex geometry theory explained by
        https://stackoverflow.com/a/43564754
        and
        Wikipedia - Convex Combination
        Points can be expressed as a convex combination of another set of points
        can be formulated as a linear programming problem.
    """
    localised_df_mask = filter_index_xyzcorn2loc(xyzcorn_df, xloc, yloc)
    localised_df = xyzcorn_df[localised_df_mask]

    if localised_df.empty:
        raise ValueError("The point(xloc, yloc) does not intersect any cells")

    if zloc is None:
        zmin = np.min(localised_df[_zcorn_names()].values)
        zmax = np.max(localised_df[_zcorn_names()].values)
        zloc = np.arange(zmin, zmax, srate)

    xyz = np.vstack(
        [np.array(xloc).repeat(zloc.size), np.array(yloc).repeat(zloc.size), zloc]
    ).T

    # aggregate corners and calculate Delaunay function for each cell
    indlog = get_points_gi(localised_df, xyz)

    return zloc, indlog


def median_down_interpolation(x1, y1, x2, bounds_error=False, fill_value=None):
    """Returns an interpolated trace y2 according to x2 where the values of
    y2 are the median values of y1 between x2[i] and x2[i+1].
    x1 should be sampled such that dx1 << dx2
    Args:
        x1 (array-like): Input index usually of increasing order
        y1 (array-like): Input function for x1 to be interpolated to x2
        x2 (array-like): Output index
        bounds_error (bool, optional): [description]. Defaults to False.
            Not implemented.
        fill_value ([type], optional): Fill value for outside valid bounds.
            Defaults to None.
    """
    # TODO: Add bounds check

    if fill_value is None:
        fill_value = np.nan

    y2 = np.full_like(x2, fill_value)
    for xtop, xbase, val in zip(x1[:-1], x1[1:], y1[:-1]):
        mask = np.logical_and(np.less(x2, xbase), np.greater_equal(x2, xtop))
        y2[mask] = val
    return y2


def cpgrid_to_rg(
    volume: xr.Dataset,
    xyzcorn_df: pd.DataFrame,
    mapping_dims: Sequence[str] = ("xline", "iline"),
    depth_dim: Union[str, None] = None,
    buffer: int = 0,
    srate: float = 0.1,
    client: Union[None, Client] = None,
) -> xr.Dataset:
    """
    Args:
        volume: A dataset with geometry to extract logs at
        xyzcorn_df: The xyzcorn Dataframe from eclx.EclDeck.xyzcorn
        mapping_dims: The dims in volume which signify the trace dimensions
        depth_dim: The depth dimension to use for modelling from volume
        buffer: The number of samples above and below the active sim grid to include in the output
        srate: The sample rate of the output in depth
        client: A dask client if using multiprocessing (volume should be chunked)

    Returns:
        A Dataset containing `gi` with geometry matching volume but a different z dimension
            if depth_dim was not specified
    """
    xkey = "cdp_x"
    ykey = "cdp_y"

    for dim in mapping_dims:
        assert dim in volume.dims

    if depth_dim is None:
        depth_dim_name = "depth"
        minmax_active = get_active_cell_range(xyzcorn_df)
        zloc = np.arange(
            minmax_active[4] // 1 - buffer, minmax_active[5] // 1 + buffer, srate
        )
        template = volume.copy()
        template = template.expand_dims({depth_dim_name: zloc})
    else:
        template = volume
        depth_dim_name = depth_dim

    template["gi"] = (
        tuple(key for key in template.dims),
        np.empty(tuple(sz for sz in template.dims.values()), dtype=np.int64),
    )

    def _trace_gilogs_mapper(trace, xyzcorn_df):
        # extract over a limited range
        try:
            z, i = extract_xy_gilog(
                xyzcorn_df, trace[xkey].values, trace[ykey].values, srate=srate
            )
        except ValueError:
            # there was no intersection of the cp grid
            z = np.r_[0, 10_000]
            i = np.r_[-1, -1]
        # filter and sample at desired outptu
        i2 = median_down_interpolation(z, i, zloc, fill_value=-1)
        # build output trace
        out_trace = (
            xr.Dataset(coords=trace.coords)
            .drop(depth_dim_name)
            .expand_dims({depth_dim_name: zloc})
        )
        out_trace["gi"] = ((depth_dim_name,), i2.astype(np.int64))
        return out_trace

    def _block_gilogs_mapper(ds, xyzcorn_df, **kwargs):
        # localise xyzcorn_df to block
        try:
            xyzcorn_df = xyzcorn_df.result
        except AttributeError:
            # not scattered
            pass

        block_localised_df_mask = filter_index_xyzcorn2loc(
            xyzcorn_df, ds[xkey].values, ds[ykey].values
        )
        block_localised_df = xyzcorn_df[block_localised_df_mask]

        preserve_dim_order = tuple(key for key in ds.dims)
        stack = ds.stack({"trace": mapping_dims})
        givol = (
            stack.groupby("trace")
            .map(_trace_gilogs_mapper, args=(block_localised_df,))
            .unstack("trace")
        )
        return givol.transpose(*preserve_dim_order)

    chunks = {key: len(val) for key, val in template.chunks.items()}
    if not chunks:
        chunks = {"1 chunk"}

    _block_gilogs_mapper.__name__ = f"Gilog over chunks: {chunks}"

    if client is not None:
        xyzcorn_scat = client.scatter(xyzcorn_df)
        return xr.map_blocks(
            _block_gilogs_mapper, template, args=(xyzcorn_scat,), template=template
        )
    else:
        return xr.map_blocks(
            _block_gilogs_mapper, template, args=(xyzcorn_df,), template=template
        )
