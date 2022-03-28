"""
Example data sets for sim2x
"""
import os
import pooch
from ._version import version as __version__
import zipfile
import pathlib

T1A = pooch.create(
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

VOLVE = pooch.create(
    path=os.curdir,
    base_url="https://github.com/trhallam/sim2x/raw/{version}/resources/",
    # Always get the main branch if dev in version. Thick package doesn't use dev releases.
    version=__version__ + "+dirty" if "dev" in __version__ else __version__,
    # If this is a development version, get the data from the master branch
    version_dev="main",
    # The registry specifies the files that can be fetched from the local storage
    registry={
        "VOLVE_2020ZZ_OCT_PCAP.zip": "8dbc75ade766734c92fd60fbe6b0032998d6ebff74effd8dff38290770d1f52c",
        "volve_sim2seis_inputs.zip": "2f2ace866e9b140ee8ecd2e608239fb47060953d6fd95cf0862e6a3deab264cc",
        "volve10-migvel-twt-sub3d.sgy": "f2194eaef8ad675f2efe26845ed61e37451472cc23a770d0ec337778b61ae9a5",
        "volve10r12-full-twt-sub3d.sgy": "782a5c2ae952c47aedcc35d979ca8aef0259e6e77c0eaf0f19c4a4e222d8403b",
        "volve10r12-full-z-sub3d.sgy": "3c6c47ae6cc009002ce930b81551da4dd0c83fee788a538bb7b129d87aab911c",
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

    return unzipped_path / fpath.stem


def fetch_t1a_example_data():
    return {key: T1A.fetch(key, processor=unzip_the_data) for key in T1A.registry}


def fetch_volve_example_data():
    return {key: VOLVE.fetch(key, processor=unzip_the_data) for key in T1A.registry}
