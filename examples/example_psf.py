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
# # Example PSF Wavelet Construction
#
# Sim2x uses an analytic form the PSF wavelet based upon the method of incident rays by ...

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from matplotlib import pyplot as plt

# %% [markdown]
# ## Create a base wavelet

# %%
from sim2x import wavelets as wv
ricker = wv.RickerWavelet(25, )
ricker.resample(dt=0.001)
ricker.as_seconds()

plt.plot(ricker.time, ricker.amp)

# %%
from sim2x._sim2seis._psf import *

# %%
import xarray as xr

# %%
fdom_ricker = analytic_illumination_kdom(ricker, 20, 20, 3500, (20, 25), (10, 15), size=(64, 64, 128))


# %%
fig, axs = plt.subplots(ncols=3, figsize=(15, 4))
fdom_ricker.sel(kx=0).plot(y="kz", ax=axs[0])
fdom_ricker.sel(ky=0).plot(y="kz", ax=axs[1])
fdom_ricker.sel(kz=-0.02, method="nearest").plot(y="ky", ax=axs[2])

# %%
psf_ricker = psf(ricker, 20, 20, 3500, [10, 25], [0, 15], size=(64, 64, 128), gaussian_sigma=0.5, twt=True)

# %%
fig, axs = plt.subplots(ncols=3, figsize=(15, 4))
psf_ricker.sel(x=0).plot(y="z", ax=axs[0])
psf_ricker.sel(y=0).plot(y="z", ax=axs[1])
psf_ricker.sel(z=0, method="nearest").plot(y="y", ax=axs[2])

# %%
psf_ricker

# %%
