from typing import Sequence, Tuple, Dict
import click
from loguru import logger
from pydantic import FilePath, Field, BaseModel


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


class Sim2impConfig(Sim2Config):

    IO: _Sim2impConfigIO = _Sim2impConfigIO()
    MODEL: _Sim2impConfigModel = _Sim2impConfigModel()


Sim2impConfig.__doc__ = IMP_CONFIG_DOC


@click.command()
@click.argument("config_file", type=click.File(), required=True)
@click.pass_context
def imp(ctx, config_file):
    """Convert a simulation corner point grid to a regular grid with a glocal cell index (GI)"""
    click.secho(f"{NAME} - v{__version__}\n - {__name__}")

    logger.debug("Loading Config")
    config = Sim2impConfig.parse_raw(config_file.read())
    logger.debug(config)

    # print(config)
    pass
