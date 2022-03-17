from typing import Dict, Sequence, Union, Type
import re

import pandas as pd

from eclx import EclDeck
from eclx._eclmaps import EclUnitMap, EclUnitScaler
from eclx._grid import _xcorn_names, _ycorn_names, _zcorn_names

from digirock.typing import PropsDict
from digirock import Element


def transform_units(sim: EclDeck) -> None:
    """Convert from simulator units to digirock compatible units.
    This function reads the units from the init_intehead attribute and
    performs the necessary transforms.
    """
    units = sim.init_intehead["UNITS"]

    if units != 1:
        sim_units = EclUnitMap(units).name
        unit_scalar = EclUnitScaler[sim_units].value

        # transform propreties
        for col in sim.data.columns:
            if "PRESSURE" in col:
                sim.data[col] = sim.data[col] * unit_scalar["pressure"]
            if re.match(r"^D(X|Y|Z)$", col):
                sim.data[col] = sim.data[col] * unit_scalar["length"]
            if re.match(r"TOPS", col):
                sim.data[col] = sim.data[col] * unit_scalar["length"]

        # transform coordinates as well
        corns = _xcorn_names() + _ycorn_names() + _zcorn_names()
        for col in sim.xyzcorn.columns:
            if col in corns:
                sim.xyzcorn[col] = sim.xyzcorn[col] * unit_scalar["length"]

        sim.data["centerx"] = sim.xyzcorn[_xcorn_names()].mean(axis=1)
        sim.data["centery"] = sim.xyzcorn[_ycorn_names()].mean(axis=1)
        sim.data["centerz"] = sim.xyzcorn[_zcorn_names()].mean(axis=1)

        sim.init_intehead["UNITS"] = 1


def get_props_from_sim(sim: EclDeck, report: int) -> PropsDict:
    """Get the properties dictionary for a given report from sim instance

    Args:
        sim: The simulation export from eclx
        report: the report number to return

    Returns:
        dictionary of properties for digirock
    """
    cols = sim.data.columns.to_list()
    stat_cols = [v for v in cols if not re.match(r".+_\d+$", v)]
    dyn_cols = [v for v in cols if re.match(r".+_\d+$", v)]
    dyn_cols = [v for v in cols if re.match(f".+_{report}$", v)]

    if not dyn_cols:
        raise ValueError(f"No dynamic columns in sim for report {report}")

    out = {col: sim.data[col].values for col in stat_cols}
    for col in dyn_cols:
        new_col_name = re.split(r"^(.+)(_\d+)", col)[1]
        out[new_col_name] = sim.data[col].values
    return out


def popnfresh(props: PropsDict, mapping: Dict[str, str]) -> None:
    """Pop old keys and reassign them to props with a new key from mapping"""
    for key, new_key in mapping.items():
        try:
            props[new_key] = props.pop(key)
        except KeyError:
            pass


def sim2imp(
    sim: EclDeck,
    pem_model: Type[Element],
    mapping: Dict[str, str],
    reports: Sequence[int],
    extra_props: PropsDict = None,
) -> Dict[int, pd.DataFrame]:
    """Calculate elastic properties for a sim output from eclx using a digirock model.

    Args:
        sim: Eclipse simulation deck from eclx
        pem_model: digirock petroelastic model with methods for bulk_modulus, shear_modulus, density, vp and vs
        mapping: A mapping of ecl keyword names to pem_model props names.
        reports: The report numbers to process. All reports is `sim.loaded_reports`.
        extra_props: Extra properties not supplied by the sim model which are need by pem_model are supplied here.
            extra_props will overide values from sim

    Returns:
        report index to elastic property dataframe pairs
    """
    out = dict()

    for report in reports:
        props = get_props_from_sim(sim, report)
        popnfresh(props, mapping)
        if extra_props:
            props.update(extra_props)

        out[report] = pd.DataFrame(
            data=dict(
                i=sim.data["i"],
                j=sim.data["j"],
                k=sim.data["k"],
                bulk_modulus=pem_model.bulk_modulus(props),
                shear_modulus=pem_model.shear_modulus(props),
                density=pem_model.density(props),
                vp=pem_model.vp(props),
                vs=pem_model.vs(props),
            ),
            index=sim.data.index,
        )
    return out
