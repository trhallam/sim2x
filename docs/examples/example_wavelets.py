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

# %%
from sim2x import wavelets as wv
import matplotlib.pyplot as plt

# %%
rwave = wv.RickerWavelet(25)

# %%
fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
axs[0].plot(rwave.time, rwave.amp)
axs[1].plot(rwave.freq(), rwave.fmag())

# %%
spectra = wv.wavelet_spectra(0.001, rwave["amp"])

# %%
