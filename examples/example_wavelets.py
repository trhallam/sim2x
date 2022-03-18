# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python [conda env:py39]
#     language: python
#     name: conda-env-py39-py
# ---

# %% [markdown] tags=[]
# `sim2x` has a module for handling wavelet functionality related to the convolutional modelling in `sim2x seis`. The `wavelets` module has both [methods](../api/wavelets.md#wavelet-functions) for manual creation of wavelets and [`Wavelet`](../api/wavelets.md#sim2xwavelet) derived classes which are used by the modelling package.
#
# A full listing of the available wavelet functionality is available in the [api](../api/wavelets.md).
#
# Some of the wavelet module functionality is detailed in this example.
#
# ## Imports
#
# The wavelet functionality for sim2x can be imported as a contained module. Here we import as `wv` to reduce the verbosity in the example.

# %%
from sim2x import wavelets as wv
import matplotlib.pyplot as plt
from matplotlib import rc

rc("font", size=15)

# %% [markdown]
# ## Wavelet Methods and data structures
#
# Let's create a standard ricker wavelet using the `ricker` method.

# %%
nsamp = 128 # the number of samples
dt = 0.001 # the sample rate in seconds
f = 25 # the dominant frequency Hz

rwave = wv.ricker(nsamp, dt, f)

# %% [markdown]
# `rwave` is returned as structured numpy array with labelled axes `'time'` and `'amp'`.

# %%
plt.plot(rwave['time'], rwave['amp'], label="rwave")
plt.legend()

# %% [markdown]
# Functionality is available to calculate the spectra of the wavelet using [`wavelet_spectra()`](../api/wavelets.md#sim2x._sim2seis._wavelets.wavelet_spectra).

# %%
rwave_spec = wv.wavelet_spectra(dt, rwave["amp"], df=1)

# %% [markdown]
# `rwave_spec` is also a numpy structured array which has the following dtypes

# %%
rwave_spec.dtype

# %%
fig, axs = plt.subplots(ncols=3, figsize=(15, 4), sharex=True)
axs[0].plot(rwave_spec['freq'], rwave_spec['amp'].real, label="amp")
axs[0].plot(rwave_spec['freq'], rwave_spec['mag'], label="mag")
axs[0].set_xlabel("Freq (Hz)")
axs[0].set_ylabel("Amplitude")
axs[1].plot(rwave_spec['freq'], rwave_spec['phase'], label="phase")
axs[1].set_xlabel("Freq (Hz)")
axs[1].set_ylabel("Phase (rad)")
axs[0].legend()

axs[2].scatter(rwave_spec["amp"].real, rwave_spec["amp"].imag)
fig.tight_layout()

# %% [markdown] tags=[]
# ## Wavelet Classes
#
# We can perform the same functionality using the built in classes for Wavelets. This example repeats the wavelet method with the `RickerWavelet` class.
#
# **Note:** *With wavelet classes, the attributes of the wavelet are accessed via the `.` operator not with keys as for structured numpy arrays.*

# %%
rwave_class = wv.RickerWavelet(f, nsamp=nsamp, dt=dt, name="ricker", df=1)

# %%
fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
axs[0].plot(rwave_class.time, rwave_class.amp)
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Amplitude")
axs[1].plot(rwave_class.freq, rwave_class.fmag)
axs[1].set_xlabel("Freq (Hz)")
axs[1].set_ylabel("Phase (rad)")

# %% [markdown]
# You can rotate the wavelet phase if needed.

# %%
rwave_class.shift_phase(90)

# %%
fig, axs = plt.subplots(ncols=3, figsize=(15, 4))
axs[0].plot(rwave_class.time, rwave_class.amp)
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Amplitude")
axs[1].plot(rwave_class.freq, rwave_class.fmag)
axs[1].set_xlabel("Freq (Hz)")
axs[1].set_ylabel("Phase (rad)")

sct = axs[2].scatter(rwave_class.famp.real, rwave_class.famp.imag, c=rwave_class.freq)
axs[2].set_xlabel("Freq Amp Real")
axs[2].set_ylabel("Freq Amp Imag")
plt.colorbar(sct, ax=axs[2], label="Freq (Hz)")
fig.tight_layout()

# %% [markdown]
# Or convert the time axis to milliseconds

# %%
rwave_class.as_milliseconds()

fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
axs[0].plot(rwave_class.time, rwave_class.amp)
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Amplitude")
axs[1].plot(rwave_class.freq, rwave_class.fmag)
axs[1].set_xlabel("Freq (Hz)")
axs[1].set_ylabel("Phase (rad)")
fig.tight_layout()

# %% [markdown]
# The wavelet can also be resampled to a different sample rate `dt`.

# %%
print("Current sample rate:", rwave_class.dt)
print("Current number of samples:", rwave_class.nsamp)
rwave_class.resample(0.5)
print("New sample rate:", rwave_class.dt)
print("New number of samples:", rwave_class.nsamp)

# %% [markdown]
# But the overall dimensions stay the same. The spectrum looks different because with a higher number of samples are FFT transforms up to higher frequencies.

# %%
fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
axs[0].plot(rwave_class.time, rwave_class.amp)
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Amplitude")
axs[1].plot(rwave_class.freq, rwave_class.fmag)
axs[1].set_xlabel("Freq (Hz)")
axs[1].set_ylabel("Phase (rad)")
fig.tight_layout()

# %%
