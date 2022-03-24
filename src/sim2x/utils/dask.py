import xarray as xr
from dask import array as da

from .tools import _optional_import_


def dask_gaussian_filter(ds: xr.Dataset, sigma: float = 0.5) -> None:
    """Applies filer to variables in place.

    Args:
        ds:
        sigma:
    """
    dndf = _optional_import_("dask_image.ndfilters", package="dask_image")
    for var in ds.data_vars:
        _ta = da.array(ds[var])
        ds[var] = (ds[var].dims, dndf.gaussian_filter(_ta, sigma).compute())
