from typing import Sequence, Tuple, Dict, Optional, List
import click
from enum import Enum
from loguru import logger
from pydantic import FilePath, Field, BaseModel, validator


from .._version import version as __version__
from .config import Sim2Config

NAME = "Sim2x Seis"

SEIS_CONFIG_DOC = """======================== {NAME} v{version} ========================
    Default sim2x seis configuration file:

        - The '#' sign denotes the start of comments.
        - Section titles should not be modified. These are the values without any indent.
        - Options for each variable are detailed in the comments.

    To run `sim2x seis` you must first convert your Eclipse model to a dataiced exlx deck using `eclx export`.
    Then run `sim2x rg` to get a regular grid.
    Then convert your `eclx` output to the elastic domain using `sim2x imp`.

    This configuration file can be passed to `seis` in the command line using

    ```
    sim2x seis default_sim2x_seis_config.yml
    ```

    Note that any parameters in this config file will be overwritten if passed
    explicitly to `sim2x seis` in the command line.

    Full documentation in available on the sim2x docs page
    """.format(
    version=__version__, NAME=NAME
)


class _Sim2seisConfigIO(BaseModel):
    """--------------------------------------
    Parameters for controlling Sim2x Seis Input/Ouput"""

    modelpath: str = Field(
        None, description="The iced model path (output of sim2x imp)"
    )
    regular_grid: str = Field("", description="A regular grid output from sim2x rg")
    outpath: str = Field(
        None,
        description="The output path of the synthetic seismic volumes, else use same as modelpath",
    )
    seisnc_prefix: str = Field(
        None, description="Prefix to prepend to every output file."
    )
    seisnc: bool = Field(True, description="Output seisnc volumes")
    segy: bool = Field(False, description="Ouput segy volumes")
    reports: Sequence[int] = Field(
        [0], description="The simulation report numbers to output"
    )
    report_diffs: Sequence[int] = Field(
        [],
        description="List of report pairs to difference e.g. `[a, b, d, c]` will output a-b and d-c",
    )
    earth_model: str = Field(
        None, description="The output path of the earth model. Not saved if `~`"
    )
    tshift: bool = Field(
        None,
        description="Output the exact timeshift volumes from the integrated velocity",
    )
    as_depth: bool = Field(True, description="Output all volumes in Depth")
    as_twt: bool = Field(False, description="Output all volumes in TWT")

    @validator("as_twt")
    def check_one_output(cls, v, values):
        v_depth = values["as_depth"]
        if not v and not v_depth:
            raise ValueError("One of as_twt or as_depth must be `true`")


class _TWTAlignmentEnum(str, Enum):
    twt_grid = "twt_grid"
    avg_vel = "avg_vel"
    twt_volume = "twt_volume"


class _Sim2seisConfigTWT(BaseModel):
    """--------------------------------------
    Parameters for controlling Sim2x Seis TWT alignment and output"""

    twt_alignment: Optional[_TWTAlignmentEnum] = Field(
        _TWTAlignmentEnum.twt_grid.value,
        description="The TWT alignment method to use. One of {}".format(
            tuple(v.value for v in _TWTAlignmentEnum)
        ),
    )
    twt_grid: str = Field(
        "",
        description="The seisworks ascii file containing the twt alignment horizon to match twt_alignment_kind.",
    )
    twt_grid_kind: int = Field(
        1, description=" The layer in the simulation model to align to the twt grid."
    )
    avg_velocity: float = Field(
        3000.0,
        description=" The overburden average velocity. twt_alignment_kind will be shifted to match this velocity.",
    )
    twt_volume: str = Field(
        "",
        description=" Provide a depth domain seisnc volume with TWT values at which to align the output.",
    )
    zrate: int = Field(
        1, description=" Vertical sample rate, m unless `as_twt = True` else ms."
    )
    vert_buff: int = Field(
        100, description=" Vertical buffer above and below model in samples."
    )


class _WaveletEnum(str, Enum):
    analytic = "analytic"
    petrel = "petrel"


class _WaveletTypeEnum(str, Enum):
    ricker = "ricker"


class _Sim2seisConfigWAVELET(BaseModel):
    """--------------------------------------
    Parameters for controlling Sim2x Seis Wavelet"""

    wavelet_input: Optional[_WaveletEnum] = Field(
        _WaveletEnum.analytic.value,
        description="The wavelet input. One of {}".format(
            tuple(v.value for v in _WaveletEnum)
        ),
    )
    wavelet_type: Optional[_WaveletTypeEnum] = Field(
        _WaveletTypeEnum.ricker.value,
        description="Analytic wavelet to use if `wavelet_input: analytic`. One of {}".format(
            tuple(v.value for v in _WaveletTypeEnum)
        ),
    )
    freq: float = Field(30, description="Dominant frequency (Hz) of analytic wavelet")
    nsamp: int = Field(256, description="Number of samples in analytic wavelet")
    srate: int = Field(1, description="The sample rate (ms) of the analytic wavelet")
    wavelet_file: str = Field("", description="Path to a wavelet file")
    psf: bool = Field(False, description="Use a 3D PSF wavelet in convolution")
    psf_max_offset: Tuple[float, float] = Field(
        (4500, 4500),
        description="Specify the acquisition patch size to limit the offset of the psf\n"
        " filter in the in-line and cross-line directions.",
    )


class _ReflectivityEnum(str, Enum):
    zoeppritz = "zoeppritz"
    aki_richards = "aki-richards"


class _Sim2seisConfigCONVOLUTION(BaseModel):
    """--------------------------------------
    Parameters for controlling Sim2x Seis Convolution"""

    angles: Dict[str, List[float]] = Field(
        {
            "near": [1, 3, 5, 7, 9],
            "mid": [11, 13, 15, 17, 19],
            "far": [21, 23, 25, 27, 29],
        },
        description="Angles to model and stack with name as key",
    )
    sum_angles: Dict[str, Sequence[str]] = Field(
        {"full": ["near", "mid", "far"]},
        description="Sum angle ranges from `angles` to stacked output by name",
    )
    # filter_ilxl:
    # filter_xy:
    refl_method: Optional[_ReflectivityEnum] = Field(
        _ReflectivityEnum.zoeppritz.value,
        description="The reflectivity calculation method, One of {}".format(
            tuple(v.value for v in _ReflectivityEnum)
        ),
    )
    quad: bool = Field(
        False, description="Output seismic and differences in quadrature phase"
    )
    run: bool = Field(True, description="Turn convolutional modelling on/off")


class _FillEnum(str, Enum):
    const = "const"
    pem = "pem"
    volume = "volume"


class _FillEnumR(str, Enum):
    const = "const"
    pem = "pem"
    volume = "volume"
    ratio = "ratio"


class _Sim2seisConfigEarthModel(BaseModel):
    """--------------------------------------
    Parameters for controlling Sim2x Seis PEM Model"""

    ob_vp_fill_method: Optional[_FillEnum] = Field(
        _FillEnum.const.value,
        description="Select a method to fill the compressional velocity in the overburden, one of {}".format(
            tuple(v.value for v in _FillEnum)
        ),
    )

    ob_vs_fill_method: Optional[_FillEnumR] = Field(
        _FillEnum.const.value,
        description="Select a method to fill the shear velocity in the overburden, one of {}".format(
            tuple(v.value for v in _FillEnumR)
        ),
    )

    ob_den_fill_method: Optional[_FillEnum] = Field(
        _FillEnum.const.value,
        description="Select a method to fill the density in the overburden, one of {}".format(
            tuple(v.value for v in _FillEnum)
        ),
    )

    ub_vp_fill_method: Optional[_FillEnum] = Field(
        _FillEnum.const.value,
        description="Select a method to fill the compressional velocity in the underburden, one of {}".format(
            tuple(v.value for v in _FillEnum)
        ),
    )

    ub_vs_fill_method: Optional[_FillEnumR] = Field(
        _FillEnum.const.value,
        description="Select a method to fill the shear velocity in the underburden, one of {}".format(
            tuple(v.value for v in _FillEnumR)
        ),
    )

    ub_den_fill_method: Optional[_FillEnum] = Field(
        _FillEnum.const.value,
        description="Select a method to fill the density in the underburden, one of {}".format(
            tuple(v.value for v in _FillEnum)
        ),
    )

    res_vp_fill_method: Optional[_FillEnum] = Field(
        _FillEnum.const.value,
        description="Select a method to fill the inactive cells with compressional velocity in the reservoir, one of {}".format(
            tuple(v.value for v in _FillEnum)
        ),
    )

    res_vs_fill_method: Optional[_FillEnumR] = Field(
        _FillEnum.const.value,
        description="Select a method to fill the inactive cells with shear velocity in the reservoir, one of {}".format(
            tuple(v.value for v in _FillEnumR)
        ),
    )

    res_den_fill_method: Optional[_FillEnum] = Field(
        _FillEnum.const.value,
        description="Select a method to fill the inactive cells with density in the reservoir, one of {}".format(
            tuple(v.value for v in _FillEnum)
        ),
    )

    vp_ob_const: float = Field(
        3500, description="Constant backgroud P-velocity (m/s) for overburden"
    )
    vp_ob_pem: str = Field(
        None,
        description="Name of the digirock model to use for the nan-fill in the model",
    )
    vp_ob_vol: str = Field(
        None, description="Path to a seisnc volume to fill the blank cells"
    )

    vs_ob_const: float = Field(
        3500, description="Constant backgroud S-velocity (m/s) for overburden"
    )
    vs_ob_pem: str = Field(
        None,
        description="Name of the digirock model to use for the nan-fill in the model",
    )
    vs_ob_vol: str = Field(
        None, description="Path to a seisnc volume to fill the blank cells"
    )
    vs_ob_ratio: float = Field(0.5, description="The vs/vp ratio to calculate vs")

    den_ob_const: float = Field(
        3500, description="Constant backgroud density (g/cc) for overburden"
    )
    den_ob_pem: str = Field(
        None,
        description="Name of the digirock model to use for the nan-fill in the model",
    )
    den_ob_vol: str = Field(
        None, description="Path to a seisnc volume to fill the blank cells"
    )

    vp_ub_const: float = Field(
        3500, description="Constant backgroud P-velocity (m/s) for underburden"
    )
    vp_ub_pem: str = Field(
        None,
        description="Name of the digirock model to use for the nan-fill in the model",
    )
    vp_ub_vol: str = Field(
        None, description="Path to a seisnc volume to fill the blank cells"
    )

    vs_ub_const: float = Field(
        3500, description="Constant backgroud S-velocity (m/s) for underburden"
    )
    vs_ub_pem: str = Field(
        None,
        description="Name of the digirock model to use for the nan-fill in the model",
    )
    vs_ub_vol: str = Field(
        None, description="Path to a seisnc volume to fill the blank cells"
    )
    vs_ub_ratio: float = Field(0.5, description="The vs/vp ratio to calculate vs")

    den_ub_const: float = Field(
        3500, description="Constant backgroud density (g/cc) for underburden"
    )
    den_ub_pem: str = Field(
        None,
        description="Name of the digirock model to use for the nan-fill in the model",
    )
    den_ub_vol: str = Field(
        None, description="Path to a seisnc volume to fill the blank cells"
    )

    vp_res_const: float = Field(
        3500, description="Constant backgroud P-velocity (m/s) for reservoir"
    )
    vp_res_pem: str = Field(
        None,
        description="Name of the digirock model to use for the nan-fill in the model",
    )
    vp_res_vol: str = Field(
        None, description="Path to a seisnc volume to fill the blank cells"
    )

    vs_res_const: float = Field(
        3500, description="Constant backgroud S-velocity (m/s) for reservoir"
    )
    vs_res_pem: str = Field(
        None,
        description="Name of the digirock model to use for the nan-fill in the model",
    )
    vs_res_vol: str = Field(
        None, description="Path to a seisnc volume to fill the blank cells"
    )
    vs_res_ratio: float = Field(0.5, description="The vs/vp ratio to calculate vs")

    den_res_const: float = Field(
        3500, description="Constant backgroud density (g/cc) for reservoir"
    )
    den_res_pem: str = Field(
        None,
        description="Name of the digirock model to use for the nan-fill in the model",
    )
    den_res_vol: str = Field(
        None, description="Path to a seisnc volume to fill the blank cells"
    )

    # vol_zaxis:
    #   req: opt
    #   type: string
    #   default: TWT
    #   docs: The axis of the input volumes used to fill the data gaps. Defaults to TWT
    #   options:
    #     - TWT
    #     - DEPTH

    gaussian_spatial_smoother: Tuple[float, float, float] = Field(
        (1, 1, 0.4),
        description="Filter dimensions in samples of a gaussian spatial smoother. [iline, xline, z]",
    )


class Sim2seisConfig(Sim2Config):

    IO: _Sim2seisConfigIO = _Sim2seisConfigIO()
    TWT: _Sim2seisConfigTWT = _Sim2seisConfigTWT()
    WAVELET: _Sim2seisConfigWAVELET = _Sim2seisConfigWAVELET()
    CONVOLUTION: _Sim2seisConfigCONVOLUTION = _Sim2seisConfigCONVOLUTION()
    EARTHMODEL: _Sim2seisConfigEarthModel = _Sim2seisConfigEarthModel()


Sim2seisConfig.__doc__ = SEIS_CONFIG_DOC


@click.command()
@click.argument("config_file", type=click.File(), required=True)
@click.pass_context
def seis(ctx, config_file):
    """Convert the outputs of rg and imp to synthetic seismic"""
    click.secho(f"{NAME} - v{__version__}\n - {__name__}")

    logger.debug("Loading Config")
    config = Sim2seisConfig.parse_raw(config_file.read())
    logger.debug(config)

    # print(config)
    pass
