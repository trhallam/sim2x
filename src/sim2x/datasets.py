"""
Example data sets for sim2x
"""
import os
import pooch
from ._version import version as __version__
import zipfile
import pathlib

GOODBOY = pooch.create(
    path=os.curdir,
    base_url="https://github.com/trhallam/sim2x/raw/{version}/resources/",
    # Always get the main branch if dev in version. Thick package doesn't use dev releases.
    version=__version__ + "+dirty" if "dev" in __version__ else __version__,
    # If this is a development version, get the data from the master branch
    version_dev="main",
    # The registry specifies the files that can be fetched from the local storage
    registry={
        "t1a.zip": "1206afc11eec7edf4d9ec41b6495ec7dba4a6653390c10a453d6ba7d36b968b0",
    },
)


def unzip_the_data(fname, action, pooch):
    fpath = pathlib.Path(fname)
    if fpath.suffix != ".zip":
        return fname

    unzipped_path = fpath.parent

    if not unzipped_path.exists():
        action = "update"

    if action in ("update", "download"):
        with zipfile.ZipFile(fpath, "r") as zip_file:
            zip_file.extractall(unzipped_path)

    return unzipped_path


def fetch_example_data():
    return {
        key: GOODBOY.fetch(key, processor=unzip_the_data) for key in GOODBOY.registry
    }
