import click
from loguru import logger
from pydantic import FilePath, Field, BaseModel


from .._version import version as __version__
from .config import Sim2Config

NAME = "Sim2x Rg"

RG_CONFIG_DOC = """======================== Sim2x RG v{version} ========================
    Default sim2x rg configuration file:

        - The '#' sign denotes the start of comments.
        - Section titles should not be modified. These are the values without any indent.
        - Options for each variable are detailed in the comments.

    To run `sim2x rg` you must first convert your Eclipse model to a dataiced exlx deck using `eclx export`.
    This configuration file can be passed to `rg` in the command line using

    ```
    sim2x rg default_sim2x_rg_config.yml
    ```

    Note that any parameters in this config file will be overwritten if passed
    explicitly to `sim2x rg` in the command line.

    Full documentation in available on the sim2x docs page
    """.format(
    version=__version__
)


class _Sim2rgConfigIO(BaseModel):
    """--------------------------------------
    Parameters for controlling Sim2Rg Input/Ouput"""

    modelpath: str = Field(
        "", description="The iced model path (output of eclx export)"
    )
    segypath: str = Field(
        "", description="The segy file whose geometry will be used for the regular grid"
    )
    outpath: str = Field("", description="The output path of the regular grid")
    jobs: int = Field(
        1,
        description="The number of jobs to split the task into",
    )


class _Sim2rgConfigSampling(BaseModel):
    """--------------------------------------
    Parameters for controlling Sim2Rg Sampling"""

    sample_rate: float = Field(0.1, description="The sample rate in depth domain")
    min_x: float = Field(None, description="The minimum x sampling location")
    max_x: float = Field(None, description="The maximum x sampling location")
    min_y: float = Field(None, description="The minimum y sampling location")
    max_y: float = Field(None, description="The minimum y sampling location")
    min_iline: int = Field(None, description="The minimum inline sample")
    max_iline: int = Field(None, description="The maximum inline sample")
    min_xline: int = Field(None, description="The minimum crossline sample")
    max_xline: int = Field(None, description="The maximum crossline sample")
    iline_step: int = Field(None, description="The inline sample step")
    xline_step: int = Field(None, description="The crossline sample step")


class Sim2rgConfig(Sim2Config):

    IO: _Sim2rgConfigIO = _Sim2rgConfigIO()
    SAMPLING: _Sim2rgConfigSampling = _Sim2rgConfigSampling()


Sim2rgConfig.__doc__ = RG_CONFIG_DOC


@click.command()
@click.argument("config_file", type=click.File(), required=True)
@click.option(
    "-j",
    "--jobs",
    help="Number of parallel processors to use for running model evaluations.",
    type=click.IntRange(1, 4),
)
@click.option("-d", "--debug", help="Activate debugging output.", default=False)
def rg(config_file, jobs=1, debug=False):
    """Convert a simulation corner point grid to a regular grid with a glocal cell index (GI)"""
    click.secho(f"{NAME} - v{__version__}\n - {__name__}")

    logger.debug("Loading Config")
    config = Sim2rgConfig.parse_raw(config_file.read())
    logger.debug(config)

    print(config)
    pass
