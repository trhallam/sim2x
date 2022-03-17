import click
import pathlib
import sys
from loguru import logger

from .._version import version as __version__
from ._sim2rg import rg, Sim2rgConfig
from ._sim2imp import imp, Sim2impConfig
from ._imp2seis import seis, Sim2seisConfig

CONFIGS = {"rg": Sim2rgConfig, "imp": Sim2impConfig, "seis": Sim2seisConfig}


@click.group()
@click.version_option(__version__)
@click.option(
    "-d", "--debug", help="Activate debugging output.", default=False, is_flag=True
)
@click.pass_context
def main(ctx, debug=False):

    ctx.ensure_object(dict)
    ctx.obj["DEBUG"] = debug

    logger.remove()
    if debug:
        logger.add(
            sys.stderr,
            level="DEBUG",
        )
        logger.info("Set logging to DEBUG level")
    else:
        logger.add(sys.stderr, level="INFO")
    pass


@main.command(name="gc")
@click.argument(
    "command",
    type=click.STRING,
)
@click.option(
    "-f",
    "--fname",
    type=click.STRING,
    default="sim2x_{cmd}_config.yml",
)
def generate_config(command, fname):
    "Generate a default yaml config file."

    try:
        cmd = globals()[command]
        config = CONFIGS[command]()
    except KeyError:
        logger.error(f"Unkown command: {command}")
        raise SystemExit

    yml = config.yaml_desc()

    if fname == "sim2x_{cmd}_config.yml":
        fname = fname.format(cmd=command)

    with open(pathlib.Path(fname), "w") as config_file:
        config_file.write(yml)


main.add_command(rg)
main.add_command(imp)
main.add_command(seis)
