# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Full *sim2seis* workflow

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
import xarray as xr

rc("font", size=15)

# %% [markdown] tags=[]
# ## Fetch the simulation data

# %%
from sim2x.datasets import fetch_t1a_example_data

# %%
t1a = fetch_t1a_example_data()["t1a.zip"]

# %% [markdown]
# ## Extract the Simulation Data

# %%
from eclx import EclDeck

sim = EclDeck()
sim.load_grid(t1a / "TUT1A.EGRID")
sim.load_init(t1a / "TUT1A.INIT")
sim.set_rst(t1a / "TUT1A.UNRST")
sim.load_rst(reports=[0, 5, 10], keys=["PRESSURE", "SWAT", "BO"])

# %% [markdown]
# ## Run sim2imp

# %%
from digirock import (
    WaterECL,
    WoodsFluid,
    Mineral,
    VRHAvg,
    NurCriticalPoroAdjust,
    GassmannRock,
    DeadOil,
    Transform,
)
from digirock.utils.ecl import EclStandardConditions
from digirock.fluids.bw92 import wat_salinity_brine
from digirock.typing import NDArrayOrFloat, NDArrayOrInt

sal = wat_salinity_brine(
    EclStandardConditions.TEMP.value, EclStandardConditions.PRES.value, 1.00916
)
wat = WaterECL(31.02641, 1.02, 3e-6, 0.8, sal)
oil = DeadOil(std_density=0.784905)
oil.set_pvt([1.25, 1.2, 1.15], [2.06843, 5.51581, 41.36854])
fl = WoodsFluid(["swat", "soil"], [wat, oil])

sand = Mineral(2.75, 32, 14)
clay = Mineral(2.55, 25, 8)
vrh = VRHAvg(["vsand", "vclay"], [sand, clay])
ncp = NurCriticalPoroAdjust(["poro"], vrh, 0.39)

grock = GassmannRock(ncp, vrh, fl)

# we need to create a VSH transform to turn NTG into VSH
class VShaleMult(Transform):

    _methods = ["vp", "vs", "density", "bulk_modulus", "shear_modulus"]

    def __init__(self, element, mult):
        super().__init__(["NTG"], element, self._methods)
        self._mult = mult

    def density(self, props, **kwargs):
        props["vclay"] = props["NTG"] * self._mult
        return self.element.density(props, **kwargs)

    def vp(self, props, **kwargs):
        props["vclay"] = props["NTG"] * self._mult
        return self.element.vp(props, **kwargs)

    def vs(self, props, **kwargs):
        props["vclay"] = props["NTG"] * self._mult
        return self.element.vs(props, **kwargs)

    def bulk_modulus(self, props, **kwargs):
        props["vclay"] = props["NTG"] * self._mult
        return self.element.bulk_modulus(props, **kwargs)

    def shear_modulus(self, props, **kwargs):
        props["vclay"] = props["NTG"] * self._mult
        return self.element.shear_modulus(props, **kwargs)


rock = VShaleMult(grock, 0.5)

# %%
from sim2x import transform_units, sim2imp

# %%
sim.data

# %%
# convert sim untis from field to sim2x compatible
transform_units(sim)
imps = sim2imp(
    sim,  # the EclDeck
    rock,  # the full rock model
    {
        "PRESSURE": "pres",
        "SWAT": "swat",
        "PORO": "poro",
    },  # simulator props in sim.data mapped to rock model props
    sim.loaded_reports,  # the reports to run sim2imp on
    extra_props={"temp": 100},  # any extra props needed by `rock`
)
imps.keys()

# %% [markdown] tags=[]
# ## Extract a regular grid GI index

# %%
from sim2x import cpgrid_to_rg
from segysak import create2d_dataset, create3d_dataset

# %% [markdown]
# Create a 2D data volume with cdp geometry we want to model with sim2seis

# %%
arb_line = create2d_dataset((25, 1))
arb_line["cdp_x"] = (("cdp",), np.linspace(-50, 600, 25))
arb_line["cdp_y"] = (("cdp",), np.linspace(600, -50, 25))
arb_line = arb_line.set_coords(
    ("cdp_x", "cdp_y")
)  # make coordinates so they get inheritted by child volumes (e.g. gi_vol)
# arb_line["depth"] = (("depth", ), np.linspace(7980, 8220, 200))
# arb_line.expand_dims({"depth":np.linspace(7980, 8220, 200)})

# %%
vol_3d = create3d_dataset(dims=(25, 25, 1))
cdp_x_, cdp_y_ = np.meshgrid(arb_line.cdp_x.values, arb_line.cdp_y.values)
vol_3d["cdp_x"] = (("iline", "xline"), cdp_x_)
vol_3d["cdp_y"] = (("iline", "xline"), cdp_y_)

vol_3d

# %%
arb_line

# %%
plt.scatter(vol_3d.cdp_x, vol_3d.cdp_y, label="vol3d")
plt.scatter(arb_line.cdp_x, arb_line.cdp_y, label="arb_line")
plt.scatter(sim.xyzcorn["x0"], sim.xyzcorn["y0"], label="cp")
plt.legend()

# %% [markdown]
# Use the arb-line and the corner point geometry to great a new volume at the arb-line locations (cdp_x, and cdp_y) that maps the sim cell index to a regular grid sample.

# %%
arb_line

# %%
# gi_vol = cpgrid_to_rg(arb_line.drop("twt").chunk({"cdp":10}), sim.xyzcorn, mapping_dims=("cdp", ), srate=1, buffer=10).compute()
gi_vol = cpgrid_to_rg(
    arb_line.drop("twt"), sim.xyzcorn, mapping_dims=("cdp",), srate=1, buffer=50
)
display(gi_vol)
gi_vol.gi.T.plot(yincrease=False, vmin=-1, vmax=75)

# %%
gi_vol3d = cpgrid_to_rg(
    vol_3d.drop("twt"), sim.xyzcorn, mapping_dims=("xline", "iline"), srate=1, buffer=50
)
display(gi_vol3d)
gi_vol3d.sel(iline=10).gi.T.plot(yincrease=False, vmin=-1, vmax=75)

# %% [markdown] tags=[]
# ## Convert sim2imp output to regular grids
#
#

# %%
from sim2x import map_to_gi_vol

report_rgs = {key: map_to_gi_vol(val, gi_vol) for key, val in imps.items()}

# %%
report_rgs3d = {key: map_to_gi_vol(val, gi_vol3d) for key, val in imps.items()}

# %%
report_rgs3d[5]

# %%
# we will continue from here with just report 5
rem = report_rgs[5].copy()

fig, axs = plt.subplots(ncols=5, figsize=(20, 4))
rem.vp.plot(ax=axs[0], yincrease=False)
rem.vs.plot(ax=axs[1], yincrease=False)
rem.density.plot(ax=axs[2], yincrease=False)
rem.bulk_modulus.plot(ax=axs[3], yincrease=False)
rem.shear_modulus.plot(ax=axs[4], yincrease=False)
fig.tight_layout()

# %% [markdown] tags=[]
# ## Filling nan values

# %%
for key, fill_value in [
    ("vp", 3000),
    ("vs", 1550),
    ("density", 2.2),
    ("bulk_modulus", 14.5),
    ("shear_modulus", 5),
]:
    rem[key] = rem[key].where(np.invert(rem[key].isnull()), fill_value)

fig, axs = plt.subplots(ncols=5, figsize=(20, 4))
rem.vp.plot(ax=axs[0], yincrease=False)
rem.vs.plot(ax=axs[1], yincrease=False)
rem.density.plot(ax=axs[2], yincrease=False)
rem.bulk_modulus.plot(ax=axs[3], yincrease=False)
rem.shear_modulus.plot(ax=axs[4], yincrease=False)
fig.tight_layout()

# %%
rem3d = report_rgs3d[5].copy()

for key, fill_value in [
    ("vp", 3000),
    ("vs", 1550),
    ("density", 2.2),
    ("bulk_modulus", 14.5),
    ("shear_modulus", 5),
]:
    rem3d[key] = rem3d[key].where(np.invert(rem3d[key].isnull()), fill_value)

# %% [markdown]
# ## Smoothing the Earth Model

# %%
from sim2x.utils.dask import dask_gaussian_filter

# %%
dask_gaussian_filter(rem, 2.0)

fig, axs = plt.subplots(ncols=5, figsize=(20, 4))
rem.vp.plot(ax=axs[0], yincrease=False)
rem.vs.plot(ax=axs[1], yincrease=False)
rem.density.plot(ax=axs[2], yincrease=False)
rem.bulk_modulus.plot(ax=axs[3], yincrease=False)
rem.shear_modulus.plot(ax=axs[4], yincrease=False)
fig.tight_layout()

# %%
dask_gaussian_filter(rem3d, 1.5)

# %% [markdown] tags=[]
# ## Extracting surface from the simulation model

# %%
from eclx._grid import get_sim_surface

b = get_sim_surface(sim.xyzcorn, 1, face="top", slice_dir="k")
surf = arb_line.seis.surface_from_points(b[:, :2], b[:, 2])

surf.data.plot(yincrease=False)

# %%
surf = vol_3d.seis.surface_from_points(b[:, :2], b[:, 2])
surf.data.plot()

# %% [markdown]
# ## Create the TWT/DEPTH MODEL
#
#

# %%
from sim2x import twt_conv

twt_int = twt_conv.time_integral(rem.vp)
twt_int.plot(yincrease=False)
# twt

# %%
time_int3d = twt_conv.time_integral(rem3d.vp)
time_int3d.sel(iline=10).plot(yincrease=False)

# %% [markdown]
# Create a 3d variation we can model

# %%
depth_samples = xr.DataArray(
    np.linspace(2430, 2490, 25), dims="cdp", coords={"cdp": twt_int.cdp}, name="depth"
)
dsx_, dsy_ = np.meshgrid(np.linspace(2430, 2490, 25), np.linspace(2430, 2490, 25))
dsm_ = (dsx_ + dsy_) / 2
depth_samples3d = xr.DataArray(
    dsm_,
    dims=("iline", "xline"),
    coords={"iline": time_int3d.iline, "xline": time_int3d.xline},
    name="depth",
)

# %%
rem

# %%
twt_2500mps = depth_samples * 2 / 4000
peg_twt2500 = twt_conv.peg_time_interval(
    twt_int, depth_samples, twt_2500mps, mapping_dims=("cdp",)
)

# converting to twt
rem_twt = twt_conv.domain_convert_vol(
    peg_twt2500, rem, mapping_dims=("cdp",), output_samples=np.arange(1.2, 1.27, 0.001)
)

twt3d_2500mps = depth_samples3d * 2 / 2500
peg3d_twt2500 = twt_conv.peg_time_interval(time_int3d, depth_samples3d, twt3d_2500mps)

rem3d_twt = twt_conv.domain_convert_vol(peg3d_twt2500, rem3d)

# %%
rem_twt_z = twt_conv.domain_convert_vol(
    peg_twt2500,
    rem_twt,
    mapping_dims=("cdp",),
    domain_vol_direction="reverse",
    from_dim="twt",
    to_dim="depth",
)

# %%
fig, axs = plt.subplots(ncols=4, figsize=(20, 3))
rem.vp.plot(ax=axs[0])
rem_twt.vp.plot(ax=axs[1])
rem_twt_z.vp.plot(ax=axs[2])
((rem_twt_z.vp - rem.vp) / rem.vp * 100).plot(ax=axs[3])
fig.tight_layout()

# %%
# check that time conversion matches in both directions
fig, axs = plt.subplots()
twt_conv.time_convert_surface(peg_twt2500, depth_samples, mapping_dims=("cdp",)).plot(
    ax=axs
)
twt_2500mps.plot(ax=axs)


# %%
peg3d_z2500 = twt_conv.domain_convert_vol(
    peg3d_twt2500, xr.Dataset({"z": peg3d_twt2500.depth.broadcast_like(peg3d_twt2500)})
)

# %%
peg3d_z2500.sel(iline=10).z.plot()

# %%
peg3d_z2500_rtwt = twt_conv.domain_convert_vol(
    peg3d_twt2500,
    peg3d_z2500,
    from_dim="twt",
    to_dim="depth",
    domain_vol_direction="reverse",
)

# %% tags=[]
peg3d_twt2500.sel(xline=10).plot(yincrease=False)

# %% [markdown]
# ## calculate reflectivity volume

# %%
from sim2x import seis_synth, seis_interface

# %%
a = seis_synth.reflectivity(
    5,
    rem_twt.sel(cdp=10).vp.values,
    rem_twt.sel(cdp=10).vs.values,
    rem_twt.sel(cdp=10).density.values,
    method="zoep_pud",
)

# %%
plt.plot(a["Rp"])

# %%

rf = seis_synth.reflectivity_vol(rem_twt, 5, mapping_dims=("cdp",))

# %%
rf.refl_5.T.plot(yincrease=False)

# %%
rf.refl_5.T.plot(yincrease=False)

# %%
from sim2x import wavelets as wv

ricker = wv.RickerWavelet(
    25,
)
ricker.resample(dt=0.001)
ricker.as_seconds()

# %%
seis = seis_synth.convolution1d_vol(rf, "refl_5", ricker.amp, mapping_dims=("cdp",))

# %%
seis["amp"].T.plot(yincrease=False)

# %%
seis_depth = twt_conv.domain_convert_vol(
    peg_twt2500,
    seis,
    from_dim="twt",
    to_dim="depth",
    domain_vol_direction="reverse",
    mapping_dims=("cdp",),
    interpolation_kind="linear",
)
seis_depth_cubic = twt_conv.domain_convert_vol(
    peg_twt2500,
    seis,
    from_dim="twt",
    to_dim="depth",
    domain_vol_direction="reverse",
    mapping_dims=("cdp",),
    interpolation_kind="cubic",
)
seis_depth_sinc = twt_conv.domain_convert_vol(
    peg_twt2500,
    seis,
    from_dim="twt",
    to_dim="depth",
    domain_vol_direction="reverse",
    mapping_dims=("cdp",),
    interpolation_kind="sinc",
)

fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(15, 8))

seis_depth["amp"].plot(yincrease=False, ax=axs[0, 0])
seis_depth_cubic["amp"].plot(yincrease=False, ax=axs[1, 1])
seis_depth_sinc["amp"].plot(yincrease=False, ax=axs[2, 2])

(seis_depth["amp"] - seis_depth_cubic["amp"]).plot(yincrease=False, ax=axs[0, 1])
(seis_depth["amp"] - seis_depth_sinc["amp"]).plot(yincrease=False, ax=axs[0, 2])

(seis_depth_cubic["amp"] - seis_depth_sinc["amp"]).plot(yincrease=False, ax=axs[1, 2])

cdp = 12
seis_depth["amp"].sel(cdp=cdp).plot(ax=axs[1, 0], label="linear")
seis_depth_cubic["amp"].sel(cdp=cdp).plot(ax=axs[1, 0], label="cubic")

seis_depth["amp"].sel(cdp=cdp).plot(ax=axs[2, 0], label="linear")
seis_depth_sinc["amp"].sel(cdp=cdp).plot(ax=axs[2, 0], label="sinc")

seis_depth_cubic["amp"].sel(cdp=cdp).plot(ax=axs[2, 1], label="cubic")
seis_depth_sinc["amp"].sel(cdp=cdp).plot(ax=axs[2, 1], label="sinc")

fig.tight_layout()

# %%
