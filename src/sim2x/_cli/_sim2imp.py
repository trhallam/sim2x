from typing import Sequence, Tuple, Dict, Union
import click
from loguru import logger
from pydantic import FilePath, Field, BaseModel

from more_itertools import chunked

import pandas as pd

from dataicer import DirectoryHandler, plugins

from ..utils.tools import module_loader
from .._sim2imp import transform_units, sim2imp
from .._version import version as __version__
from .config import Sim2Config

NAME = "Sim2x Imp"

IMP_CONFIG_DOC = """======================== {NAME} v{version} ========================
    Default sim2x imp configuration file:

        - The '#' sign denotes the start of comments.
        - Section titles should not be modified. These are the values without any indent.
        - Options for each variable are detailed in the comments.

    To run `sim2x imp` you must first convert your Eclipse model to a dataiced exlx deck using `eclx export`.
    This configuration file can be passed to `imp` in the command line using

    ```
    sim2x imp default_sim2x_imp_config.yml
    ```

    Note that any parameters in this config file will be overwritten if passed
    explicitly to `sim2x imp` in the command line.

    Full documentation in available on the sim2x docs page
    """.format(
    version=__version__, NAME=NAME
)


class _Sim2impConfigIO(BaseModel):
    """--------------------------------------
    Parameters for controlling Sim2x Imp Input/Ouput"""

    modelpath: str = Field(
        "", description="The iced model path (output of eclx export)"
    )
    outpath: str = Field(
        None,
        description="The output path of the iced imp model, else use same as modelpath",
    )
    reports: Sequence[int] = Field(
        [0], description="The simulation report numbers to output"
    )
    report_diffs: Sequence[int] = Field(
        [],
        description="List of report pairs to difference e.g. `[a, b, d, c]` will output a-b and d-c",
    )
    report_diffs_pc: Sequence[int] = Field(
        [],
        description="List of report pairs to difference e.g. `[a, b, d, c]` will output \n"
        "`(a-b)/b*100` and `(d-c)/c*100`, diffs are as percentages relative to the second value",
    )


class _Sim2impConfigModel(BaseModel):
    """--------------------------------------
    Parameters for controlling Sim2x Imp PEM Model"""

    pem_file: str = Field(
        "", description="Python file with the digirock PEM model defined"
    )
    pem_name: str = Field(
        "", description="Variable name of digirock RockModel from pem_file"
    )
    nan_cell: Tuple[float, float, float] = Field(
        None,
        description="Elastic properties to fill inactive cells [velp vels rhob],\n "
        "if missing the default shale properties from the impedance modelling will be used",
    )
    property_mapping: Dict[str, str] = Field(
        None,
        description="The properties from eclipse mapped to the props names of the digirock model.\n "
        "For example:\n"
        "\tSWAT:sw",
    )
    extra_props: Dict[str, Union[int, float]] = Field(
        None,
        description="Extra properties needed by the digirock model not provided by the simualtor. e.g `temp`",
    )


class Sim2impConfig(Sim2Config):

    IO: _Sim2impConfigIO = _Sim2impConfigIO()
    MODEL: _Sim2impConfigModel = _Sim2impConfigModel()


Sim2impConfig.__doc__ = IMP_CONFIG_DOC


def calc_diffs(a, b, key, pc=False):
    a = a[key].copy().values
    b = b[key].copy().values
    if not pc:
        return b - a
    else:
        return ((b - a) / a) * 100


@click.command()
@click.argument("config_file", type=click.File(), required=True)
@click.pass_context
def imp(ctx, config_file):
    """Convert a simulation corner point grid to a regular grid with a glocal cell index (GI)"""
    click.secho(f"{NAME} - v{__version__}\n - {__name__}")

    logger.debug("Loading Config")
    config = Sim2impConfig.parse_raw(config_file.read())
    logger.debug(config)

    try:
        pem_module = module_loader("pem_module", config.MODEL.pem_file)
        rock_model = getattr(pem_module, config.MODEL.pem_name, None)
        if rock_model is None:
            raise ValueError(
                f"Could not find rock model with name {config.MODEL.pem_name}"
            )
        logger.info("Loaded digirock model")
    except (KeyError, AttributeError):
        logger.error("The pem file could not be loaded")
        raise SystemExit
    except ValueError as err:
        logger.error(err.args[0])
        raise SystemExit

    try:
        with DirectoryHandler(
            config.IO.modelpath,
            plugins.get_pandas_handlers(mode="h5", array_mode="npz"),
            "r",
        ) as dh:
            sim = dh.deice()["eclxsim"]
        logger.info("Loaded sim")
        logger.debug(f"Found loaded reports: {sim.loaded_reports}")
        logger.debug(f"Has data colums: {sim.data.columns.to_list()}")
    except:
        logger.error("The sim model could not be loaded")
        raise SystemExit

    try:
        transform_units(sim)
    except:
        logger.error("The sim model unit transform failed")
        raise SystemExit

    try:
        imps = sim2imp(
            sim,
            rock_model,
            config.MODEL.property_mapping,
            config.IO.reports,
            config.MODEL.extra_props,
        )
    except Exception as err:
        logger.debug(err)
        logger.debug(f"Sim has reports {sim.loaded_reports}")
        logger.error("The simulation to impedance process failed")
        raise SystemExit

    if config.IO.outpath is None:
        outpath = "sim2imp"
    else:
        outpath = config.IO.outpath
    logger.debug(f"Writing output to {outpath}")

    imp_props = ["bulk_modulus", "shear_modulus", "density", "vp", "vs"]

    diffs = dict()
    if config.IO.report_diffs:
        for a, b in chunked(config.IO.report_diffs, 2):
            name = f"{b}-{a}"
            logger.debug(f"Calculating diff {name}")
            _tdf = imps[a][["i", "j", "k"]].copy()
            for ip in imp_props:
                _tdf.loc[:, ip] = calc_diffs(imps[a], imps[b], ip)
            diffs[name] = _tdf

    pc_diffs = dict()
    if config.IO.report_diffs:
        for a, b in chunked(config.IO.report_diffs_pc, 2):
            name = f"{b}-{a}"
            logger.debug(f"Calculating pc diff {name}")
            _tdf = imps[a][["i", "j", "k"]].copy()
            for ip in imp_props:
                _tdf.loc[:, ip] = calc_diffs(imps[a], imps[b], ip, pc=True)
            pc_diffs[name] = _tdf

    with DirectoryHandler(
        outpath, plugins.get_pandas_handlers(mode="h5", array_mode="npz"), "w"
    ) as dh:
        dh.ice(rock_model=rock_model, sim2imp=imps, diffs=diffs, pc_diffs=pc_diffs)
    logger.info(f"Wrote output to {outpath}")


def load_sim2x_imp(filepath, classes=None):
    """Load the ouput f sim2x imp"""
    with DirectoryHandler(
        filepath, plugins.get_pandas_handlers(mode="h5", array_mode="npz"), "r"
    ) as dh:
        return dh.deice(classes=classes)
