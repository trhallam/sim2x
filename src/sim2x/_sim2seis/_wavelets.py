"""Functions and classes necessary for handling wavelets.
"""
from typing import Literal, Union, Tuple
import numpy as np
import xarray as xr

from scipy import fftpack
from scipy.signal import hilbert
from scipy.interpolate import interp1d

from ..typing import Pathlike


def ricker(nsamp: int, dt: float, f: float) -> np.ndarray:
    """Create a Ricker Wavelet

    Create an analytical Ricker wavelet with a dominant frequency of f.
    Length in samples, dt im mSec, f in Hertz

    Args:
        nsamp: Number of samples in wavelet
        dt: Sample rate in s
        f: Dominant Frequency of the wavelet in Hz

    Returns:
        Structured array of length nsamp+1 tuples ('time', 'amp')
    """
    out = np.zeros((nsamp + 1), dtype=[("time", float), ("amp", float)])
    out[("time")] = (
        np.linspace(-nsamp / 2, nsamp / 2, nsamp + 1, dtype=float) * dt
    )  # zero phase symetrical
    out[("amp")] = (1 - 2 * np.square(np.pi * f * out[("time")])) * np.exp(
        -np.square(np.pi * f * out[("time")])
    )
    return out


def ormsby(
    nsamp: int, dt: float, f1: float, f2: float, f3: float, f4: float, norm: bool = True
) -> np.ndarray:
    """Create an Ormsby filtered wavelet"""
    out = np.zeros((nsamp + 1), dtype=[("time", float), ("amp", float)])
    out[("time")] = (
        np.linspace(-nsamp / 2, nsamp / 2, nsamp + 1) * dt
    )  # zero phase symetrical
    pi = np.pi
    c1 = np.square(pi * f4) / (pi * f4 - pi * f3)
    c2 = np.square(pi * f3) / (pi * f4 - pi * f3)
    c3 = np.square(pi * f2) / (pi * f2 - pi * f1)
    c4 = np.square(pi * f1) / (pi * f2 - pi * f1)

    out[("amp")] = (
        c1 * np.square(np.sinc(pi * f4 * out[("time")]))
        - c2 * np.square(np.sinc(pi * f3 * out[("time")]))
        - c3 * np.square(np.sinc(pi * f2 * out[("time")]))
        + c4 * np.square(np.sinc(pi * f1 * out[("time")]))
    )
    if norm:
        out[("amp")] = out[("amp")] / np.max(out[("amp")])
    return out


def bandpass(nsamp, dt, f1, f2, f3, f4, norm=True, precision=0.1):
    """Create a bandpass wavelet

    Args:
        nsamp (int): Number of samples in wavelet
        dt (float): Sample rate in seconds
        f1 (float): Lower frequency off
        f2 (float): Lower frequency on
        f3 (float): Upper frequency on
        f4 (float): Upper frequency off

    Returns:
        (numpy.ndarray): Structured array of length nsamp+1 tuples ('time', 'amp')

    Notes:
        Bandpass amplitude spectrum.

           1         f2________f3
                     /          \
                    /            \
           0  _____/              \______
                  f1              f4

    """
    out = np.zeros((nsamp + 1), dtype=[("time", float), ("amp", float)])
    out[("time")] = np.linspace(-nsamp / 2, nsamp / 2, nsamp + 1) * dt
    # frequency domain spectrum and filtering
    precision = int(1 / precision)
    spec_samp = nsamp * precision
    spec_time = np.linspace(-spec_samp, spec_samp, 2 * spec_samp + 1) * dt
    spec_freq = np.fft.fftfreq(2 * spec_samp + 1, dt)
    abs_spec_freq = np.abs(spec_freq)
    nf = spec_freq.shape[0]
    hnf = int((nf - 1) / 2)
    spec_amp = np.ones_like(spec_freq)
    blank = np.logical_or(abs_spec_freq <= f1, abs_spec_freq >= f4)
    spec_amp = np.where(blank, 0, spec_amp)
    rampup = np.logical_and(abs_spec_freq > f1, abs_spec_freq < f2)
    m1 = 1.0 / (f2 - f1)
    c1 = -m1 * f1
    spec_amp = np.where(rampup, m1 * abs_spec_freq + c1, spec_amp)
    rampdown = np.logical_and(abs_spec_freq > f3, abs_spec_freq < f4)
    m2 = 1.0 / (f3 - f4)
    c2 = -m2 * f4
    spec_amp = np.where(rampdown, m2 * abs_spec_freq + c2, spec_amp)
    # inverse fft and unwrap
    time_amp = np.fft.ifft(spec_amp)
    time_amp = np.r_[time_amp[hnf:], time_amp[:hnf]]
    # interpolation to out
    amp_func = interp1d(spec_time[:-1], time_amp.real[1:], kind="cubic")
    out[("amp")] = amp_func(out[("time")])
    if norm:
        out[("amp")] = out[("amp")] / np.max(out[("amp")])
    return out


def gabor(nsamp, dt, a, k0):
    """Gabor wavelet - Gausian modulated by an exponential

    Args:
        nsamp (int):
        dt (float):
        a (float): rate of exponential attenuation away from t=0
        k0 (float): rate of modulation
    """
    out = np.zeros((nsamp + 1), dtype=[("time", float), ("amp", float)])
    out[("time")] = np.linspace(-nsamp / 2, nsamp / 2, nsamp + 1) * dt
    i = np.complex(0, 1)
    out[("amp")] = np.real(
        np.exp(-1 * np.square(out[("time")]) / (a * a))
        * np.exp(-i * k0 * out[("time")])
    )
    return out


class Wavelet:
    """A class that encapsulates wavelets in sim2x

    Attributes:
        nsamp
        dt
        units
        df
        time
        amp
        name
        wtype
    """

    def __init__(
        self,
        nsamp: int = 128,
        dt: float = 0.002,
        name: Union[None, str] = None,
        units: Literal["s", "ms"] = "s",
    ):
        """Initialise a wavelet class

        nsamp (int, optional): Defaults to 128.
            Number of samples in the wavelet
        dt (float, optional): Defaults to 0.002.
            Sample rate of the wavelet (seconds)
        name (str, optional): Defaults to 'etlp_wavelet'.
            Name or descriptor of the wavelet
        units (str, optional): Defaults to 's'.
            The units of the time axis, either s or ms
        """
        self.nsamp = nsamp
        self.dt = dt
        self.units = units
        self.time = np.linspace(-nsamp / 2, nsamp / 2, nsamp + 1) * dt
        self.amp = np.zeros_like(self.time)
        self.name = name
        self.wtype = None

    @property
    def df(self):
        return 1 / (self.nsamp * self.dt)

    def _spectra(self, df=0.1, positive=True):
        """Calculate the spectra of the wavelet

        Performs a padded FFT of the wavelet to calculate the various
        spectral components of the complex signal.

        Args:
            df (float, optional): Defaults to 0.1. Desired frequency sample rate Hz
            positive (bool, optional): Defaults to True. Return only the positive
                frequency component.

        Attributes:
            self.famp (array-like): Frequency complex amplitude array.
            self.freq (array-like): Frequency values of self.famp (Hz)
            self.fmag (array-like): Magnitude of self.famp
            self.fpow (array-like): Power of self.famp
            self.phase (array-like): Phase of the wavelet (rad)

        Notes:
            DFTs are vulnerable to spectral leakage. This occurs when a non-interger
            number of periods exist within a signal. Spectral leakage allows a single-tone
            signal disperse across several frequencies after the DFT operation.

            Zero-padding a signal does not reveal more information about the spectrum, but it only
            interpolates between the frequency bings that would occur when no zero-padding is
            applied, i.e. zero-padding does not increase spectral resolution. More measured signal
            is required for this.

        """
        if df < self.df:
            psamp = self.nsamp * int(round(self.df / df))
        else:
            psamp = self.nsamp
        self.famp = fftpack.fft(self.amp, n=psamp)
        self.fmag = np.abs(self.famp)
        self.phase = np.arctan2(self.famp.imag, self.famp.real)
        self.freq = fftpack.fftfreq(psamp, self.dt)
        if positive:
            psamp2 = psamp // 2
            self.fmag = self.fmag[:psamp2]
            self.freq = self.freq[:psamp2]
            self.phase = self.phase[:psamp2]
        else:
            self.fmag = fftpack.fftshift(self.fmag)
            self.freq = fftpack.fftshift(self.freq)

    def resample(self, dt):
        time = np.arange(self.time.min(), self.time.max(), dt)
        wave = interp1d(self.time, self.amp, kind="cubic", fill_value="extrapolate")
        amp = wave(time)
        self.time = time
        self.amp = amp
        self.dt = dt
        self.nsamp = time.shape[0]
        self._spectra()

    def as_seconds(self):
        if self.units == "ms":
            self.time = self.time / 1000.0
            self.dt = self.dt / 1000.0
            self.units = "s"
        elif self.units == "s":
            pass  # already in seconds
        else:
            raise ValueError(f"Unknown units of current wavelet {self.units}.")

    def as_miliseconds(self):
        if self.units == "ms":
            pass  # already in seconds
        elif self.units == "s":
            self.time = self.time * 1000.0
            self.dt = self.dt * 1000.0
            self.units = "ms"
        else:
            raise ValueError(f"Unknown units of current wavelet {self.units}.")

    def shift_phase(self, shift: float) -> None:
        """Rotate the phase of the wavelet.

        Args:
            shift: Phase shift in degrees

        """
        # Get the shift as radians and mod by maximum of 360.
        sign = np.sign(shift)
        shift = np.abs(shift)
        shift = np.radians(sign * (shift % 360))

        ht = hilbert(self.amp)
        self.amp = ht.real * np.cos(shift) - ht.imag * np.sin(shift)
        self.amp = self.amp.real
        self._spectra()


class RickerWavelet(Wavelet):
    """
    Attributes:
        As for [`Wavelet`][sim2x.Wavelet]
        f: dominant frequency in Hz
    """

    def __init__(
        self,
        f: float,
        nsamp: int = 128,
        dt: float = 0.002,
        name: Union[None, str] = None,
        units: Literal["s", "ms"] = "s",
    ):
        """Initialise a wavelet class

        f: The dominant frequency of the wavelet
        nsamp: Number of samples in the wavelet
        dt: Sample rate of the wavelet (seconds)
        name: Name or descriptor of the wavelet
        units: The units of the time axis, either s or ms
        """
        self.f = f
        super().__init__(nsamp=nsamp, dt=dt, name=name, units=units)
        out = ricker(self.nsamp, self.dt, f)
        self.amp = out[("amp")]
        self._spectra()
        self.wtype = "ricker"


class BandpassWavelet(Wavelet):
    """A bandpass wavelet

    Attributes:
        As for [`Wavelet`][sim2x.Wavelet]
        fbounds: The bandpass boundaries tuple
    """

    def __init__(
        self,
        fbounds: Tuple[float, float, float, float],
        norm: bool = True,
        precision: float = 0.1,
        nsamp: int = 128,
        dt: float = 0.002,
        name: Union[None, str] = None,
        units: Literal["s", "ms"] = "s",
    ):
        """Initialise a wavelet class

        fbounds: The bandpass frequency boundaries tuple
        norm:
        precision:
        nsamp: Number of samples in the wavelet
        dt: Sample rate of the wavelet (seconds)
        name: Name or descriptor of the wavelet
        units: The units of the time axis, either s or ms
        """
        self.fbounds = fbounds
        super().__init__(nsamp=nsamp, dt=dt, name=name, units=units)

        out = bandpass(self.nsamp, self.dt, *fbounds, norm=norm, precision=precision)
        self.amp = out[("amp")]
        self._spectra()
        self.wtype = "bandpass"


class OrmsbyWavelet(Wavelet):
    """An Ormsby wavelet

    Attributes:
        As for [`Wavelet`][sim2x.Wavelet]
        fbounds: The bandpass boundaries tuple
    """

    def __init__(
        self,
        fbounds: Tuple[float, float, float, float],
        norm: bool = True,
        nsamp: int = 128,
        dt: float = 0.002,
        name: Union[None, str] = None,
        units: Literal["s", "ms"] = "s",
    ):
        """Initialise a wavelet class

        fbounds: The Ormsby frequency boundaries tuple
        norm:
        nsamp: Number of samples in the wavelet
        dt: Sample rate of the wavelet (seconds)
        name: Name or descriptor of the wavelet
        units: The units of the time axis, either s or ms
        """
        self.fbounds = fbounds
        super().__init__(nsamp=nsamp, dt=dt, name=name, units=units)
        out = ormsby(self.nsamp, self.dt, *fbounds, norm=norm)
        self.amp = out[("amp")]
        self._spectra()
        self.wtype = "bandpass"


class PetrelWavelet(Wavelet):
    """A wavelet loaded from a Petrel file.

    Attributes:

    """

    def __init__(self, filepath: Pathlike, norm: bool = False):
        """Load a Petrel format exported wavelet to this class.

        Args:
            filepath: The full filepath and name to the exported ascii wavelet file.
        """
        self.file = str(filepath)
        hflags = {"WAVELET-NAME": None, "WAVELET-TFS": None, "SAMPLE-RATE": None}

        with open(filepath) as wavelet_file:
            # WAVELET HEADER
            for line in wavelet_file:
                sline = line.split()
                try:
                    if sline[0] == "EOH":
                        break
                    if sline[0] in hflags.keys():
                        hflags[sline[0]] = sline[1]
                except IndexError:
                    pass

            name = hflags["WAVELET-NAME"]
            dt = float(hflags["SAMPLE-RATE"]) / 1000.0

            # WAVELET DATA BODY
            time = []
            amp = []
            for line in wavelet_file:
                sline = line.split()
                try:
                    if sline[0] == "EOD":
                        break
                    time.append(sline[0])
                    amp.append(sline[1])
                except IndexError:
                    pass

        super().__init__(name=name, dt=dt, nsamp=len(time), units="s")
        self.time = np.array(time, dtype=float)
        self.amp = np.array(amp, dtype=float)

        if norm:
            self.amp = self.amp / np.max(np.abs(self.amp))

        # if hflags["WAVELET-TFS"] is not None:
        #     self.time = self.time + float(hflags["WAVELET-TFS"])
        self.time = self.time / 1000.0

        self._spectra()
        self.wtype = "petrel"
