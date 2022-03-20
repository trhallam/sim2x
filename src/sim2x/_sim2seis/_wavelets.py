"""Functions and classes necessary for handling wavelets.
"""
from typing import Literal, Union, Tuple, Any
import numpy as np
import numpy.typing as npt
import xarray as xr

from scipy import fftpack
from scipy.signal import hilbert
from scipy.interpolate import interp1d

from ..typing import Pathlike


spectra_dtype = np.dtype(
    [("freq", "f8"), ("amp", "c8"), ("mag", "f8"), ("phase", "f8")]
)
SpectraDtype = npt.NDArray[spectra_dtype]
wavelet_dtype = np.dtype([("time", "f8"), ("amp", "f8")])
WaveletDtype = npt.NDArray[wavelet_dtype]


def zero_phase_time_axis(nsamp: int, dt: float) -> npt.NDArray[np.float_]:
    """Create a zero phase symetrical time axis with `nsamp`

    Args:
        nsamp: Number of samples
        dt: Sample rate

    Returns:
        time axis array
    """
    return np.linspace(-nsamp / 2, nsamp / 2, nsamp + 1, dtype=float) * dt


def ricker(nsamp: int, dt: float, f: float) -> WaveletDtype:
    """Create a Ricker Wavelet

    Create an analytical Ricker wavelet with a dominant frequency of `f`.
    Length in samples, `dt` im mSec, `f` in Hertz

    Args:
        nsamp: Number of samples in wavelet
        dt: Sample rate in (s)
        f: Dominant Frequency of the wavelet in Hz

    Returns:
        Structured array of length nsamp+1 tuples ('time', 'amp')
    """
    time = zero_phase_time_axis(nsamp, dt)
    amp = (1 - 2 * np.square(np.pi * f * time)) * np.exp(-np.square(np.pi * f * time))
    return np.array([zipped for zipped in zip(time, amp)], dtype=wavelet_dtype)


def ormsby(
    nsamp: int, dt: float, f1: float, f2: float, f3: float, f4: float, norm: bool = True
) -> WaveletDtype:
    """Create an Ormsby filtered wavelet

    Args:
        nsamp: Number of samples in wavelet
        dt: Sample rate in (s)
        f1: Low frequency cut off (Hz)
        f2: Low frequency pass (Hz)
        f3: High frequency pass (Hz)
        f4: High freuency cut off (Hz)
        norm: Normalise the wavelet

    Returns:
        Structured array of length nsamp+1 tuples ('time', 'amp')
    """
    time = zero_phase_time_axis(nsamp, dt)
    pi = np.pi
    c1 = np.square(pi * f4) / (pi * f4 - pi * f3)
    c2 = np.square(pi * f3) / (pi * f4 - pi * f3)
    c3 = np.square(pi * f2) / (pi * f2 - pi * f1)
    c4 = np.square(pi * f1) / (pi * f2 - pi * f1)

    amp = (
        c1 * np.square(np.sinc(pi * f4 * time))
        - c2 * np.square(np.sinc(pi * f3 * time))
        - c3 * np.square(np.sinc(pi * f2 * time))
        + c4 * np.square(np.sinc(pi * f1 * time))
    )
    if norm:
        amp = amp / np.max(amp)
    return np.array([zipped for zipped in zip(time, amp)], dtype=wavelet_dtype)


def bandpass(
    nsamp: int,
    dt: float,
    f1: float,
    f2: float,
    f3: float,
    f4: float,
    norm: bool = True,
    precision: float = 0.1,
) -> WaveletDtype:
    """Create a bandpass wavelet

    Args:
        nsamp: Number of samples in wavelet
        dt: Sample rate in seconds
        f1: Lower frequency off
        f2: Lower frequency on
        f3: Upper frequency on
        f4: Upper frequency off
        norm: Normalise the output
        precision: in frequency domain (1/Hz)

    Returns:
        Structured array of length nsamp+1 tuples ('time', 'amp')

    Notes:
        Bandpass amplitude spectrum.

        ```
        1         f2________f3
                  /          \\
                 /            \\
        0  _____/              \\______
               f1               f4
        ```

    """
    time = zero_phase_time_axis(nsamp, dt)
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
    amp = amp_func(time)
    if norm:
        amp = amp / np.max(amp)
    return np.array([zipped for zipped in zip(time, amp)], dtype=wavelet_dtype)


def gabor(nsamp: int, dt: float, a: float, k0: float) -> WaveletDtype:
    """Gabor wavelet - Gausian modulated by an exponential

    Args:
        nsamp: Number of samples in the wavelet
        dt: Sample rate of the wavelet (s)
        a: rate of exponential attenuation away from t=0
        k0: rate of modulation

    Returns:
        Structured array of length nsamp+1 tuples ('time', 'amp')
    """
    time = zero_phase_time_axis(nsamp, dt)
    i = complex(0, 1)
    amp = np.real(np.exp(-1 * np.square(time) / (a * a)) * np.exp(-i * k0 * time))
    return np.array([zipped for zipped in zip(time, amp)], dtype=wavelet_dtype)


def wavelet_spectra(dt: float, amp: npt.NDArray[Any], df: float = 10) -> SpectraDtype:
    """Calculate the spectra of the wavelet

    Performs a padded FFT of the wavelet to calculate the various
    spectral components of the complex signal.

      - `freq`: Frequency (Hz) at which the spectra has been evaluated
      - `amp`: The complex amplitude after the forwart FFT
      - `mag`: `np.abs(amp)` - magnitude of `amp`
      - `phase`: Phase angle of the complex amplitude signal

    Args:
        dt: The wavelet sample interval (s)
        amp: The wavelet amplitude array
        df: Desired frequency sample rate Hz

    Returns:
        Structured array ('freq', 'amp', 'mag', 'phase')

    Notes:
        DFTs are vulnerable to spectral leakage. This occurs when a non-interger
        number of periods exist within a signal. Spectral leakage allows a single-tone
        signal disperse across several frequencies after the DFT operation.

        Zero-padding a signal does not reveal more information about the spectrum, but it only
        interpolates between the frequency bins that would occur when no zero-padding is
        applied, i.e. zero-padding does not increase spectral resolution. More measured signal
        is required for this.

    """
    assert len(amp.shape) == 1
    nsamp = amp.size

    # increase the number of samples to get the desired spectral sampling
    _df = 1 / (nsamp * dt)
    if df < _df:
        psamp = nsamp * int(round(_df / df))
    else:
        psamp = nsamp

    famp = fftpack.fftshift(fftpack.fft(amp, n=psamp))
    freq = fftpack.fftshift(fftpack.fftfreq(psamp, dt))
    fmag = np.abs(famp)
    phase = np.angle(famp)
    phase[fmag < 1] = 0

    return np.array(
        [zipped for zipped in zip(freq, famp, fmag, phase)],
        dtype=spectra_dtype,
    )


class Wavelet:
    """A class that encapsulates wavelets in sim2x

    Attributes:
        nsamp: Number of samples
        dt: Sample rate
        units: time axis units ("s", "ms")
        df: Spectral sample rate
        time: Time array
        amp: Amp array
        name: Name or identifier
        wtype: Wavelet Type
    """

    def __init__(
        self,
        nsamp: int = 128,
        dt: float = 0.002,
        df: float = 10,
        positive_spectra: bool = True,
        name: Union[None, str] = None,
        units: Literal["s", "ms"] = "s",
    ):
        """Initialise a wavelet class

        Args:
            nsamp: Number of samples in the wavelet
            dt: Sample rate of the wavelet (seconds)
            df: The sample rate in the frequency domain
            positive_spectra: Return only the positive portion of the frequency spectrum
            name: Name or descriptor of the wavelet
            units: The units of the time axis, either s or ms
        """
        self.nsamp = nsamp
        self.dt = dt
        self.units = units
        self._wavelet = np.zeros((nsamp + 1), dtype=[("time", float), ("amp", float)])
        self._wavelet["time"] = np.linspace(-nsamp / 2, nsamp / 2, nsamp + 1) * dt
        self._wavelet["amp"] = np.zeros_like(self.time)
        self.name = name
        self.wtype = None
        self._update_spectra(df, positive_spectra)

    def _update_spectra(self, df: float, positive_spectra: bool) -> None:
        # Update the internal frequency spectra
        if self.units == "ms":
            dt = self.dt / 1000
        else:
            dt = self.dt

        self._df = df
        self._positive_spectra = positive_spectra
        self._spectra = wavelet_spectra(dt, self._wavelet["amp"], df=self._df)
        if positive_spectra:
            psamp2 = self._spectra.size // 2
            self._spectra = self._spectra[psamp2:]

    @property
    def df(self) -> float:
        return 1 / (self.nsamp * self.dt)

    @property
    def time(self) -> npt.NDArray[np.float_]:
        return self._wavelet["time"]

    @property
    def amp(self) -> npt.NDArray[np.float_]:
        return self._wavelet["amp"]

    @property
    def spectra(self) -> SpectraDtype:
        """the wavelet spectra"""
        return self._spectra

    @property
    def freq(self) -> npt.NDArray[np.float_]:
        """the spectra frequency"""
        return self._spectra["freq"]

    @property
    def fmag(self, df: float = 10) -> npt.NDArray[np.float_]:
        """the spectra magnitude"""
        return self._spectra["mag"]

    @property
    def famp(self, df: float = 10) -> npt.NDArray[np.complex_]:
        """the spectra amplitude"""
        return self._spectra["amp"]

    @property
    def fphase(self, df: float = 10) -> npt.NDArray[np.float_]:
        """the spectra phase"""
        return self._spectra["phase"]

    def set_wavelet(
        self,
        time: Union[None, npt.NDArray[np.float_]] = None,
        amp: Union[None, npt.NDArray[np.float_]] = None,
    ) -> None:
        """Set the wavelet manually

        `time` and `amp` are 1D arrays with the same size.

        Args:
            time: The time array
            amp: The amplitude array
        """
        if amp is None and time is None:
            raise ValueError("One of `time` or `amp` must be set.")
        elif amp is None:
            assert len(time.shape) == 1
            assert time.size == self.nsamp
            self._wavelet["time"] = time
        elif time is None:
            assert len(amp.shape) == 1
            assert amp.size == self.nsamp
            self._wavelet["amp"] = amp
        else:
            assert len(time.shape) == 1 and len(amp.shape) == 1
            assert time.size == amp.size
            self.nsamp = time.size
            self._wavelet = np.array(
                [zipped for zipped in zip(time, amp)], dtype=wavelet_dtype
            )
        self._update_spectra(self._df, self._positive_spectra)

    def resample(self, dt: float) -> None:
        """Resample the wavelet to a new `dt`

        Args:
            dt: new sample rate
        """
        time = np.arange(self.time.min(), self.time.max(), dt)
        wave = interp1d(self.time, self.amp, kind="cubic", fill_value="extrapolate")
        amp = wave(time)
        self.dt = dt
        self.set_wavelet(time, amp)

    def as_seconds(self) -> None:
        """Convert time axis to seconds"""
        if self.units == "ms":
            self._wavelet["time"] = self.time / 1000.0
            self.dt = self.dt / 1000.0
            self.units = "s"
        elif self.units == "s":
            pass  # already in seconds
        else:
            raise ValueError(f"Unknown units of current wavelet {self.units}.")

    def as_milliseconds(self) -> None:
        """Convert time axis to milliseconds"""
        if self.units == "ms":
            pass  # already in seconds
        elif self.units == "s":
            self._wavelet["time"] = self.time * 1000.0
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
        amp = ht.real * np.cos(shift) - ht.imag * np.sin(shift)
        self.set_wavelet(amp=amp.real)


class RickerWavelet(Wavelet):
    """Ricker wavelet class

    Extends the base [`Wavelet`][sim2xwavelet] class.

    Attributes:
        f (float): dominant frequency in Hz
    """

    def __init__(
        self,
        f: float,
        **kwargs,
    ):
        """Initialise a wavelet class

        Args:
            f: The dominant frequency of the wavelet
            kwargs: keyword arguments passed to [`Wavelet`][sim2xwavelet]
        """
        self.f = f
        super().__init__(**kwargs)
        wave = ricker(self.nsamp, self.dt, f)
        self.set_wavelet(wave["time"], wave["amp"])
        self.wtype = "ricker"


class BandpassWavelet(Wavelet):
    """A bandpass wavelet

    Extends the base [`Wavelet`][sim2xwavelet] class.

    Attributes:
        fbounds (tuple): The bandpass boundaries tuple
    """

    def __init__(
        self,
        fbounds: Tuple[float, float, float, float],
        norm: bool = True,
        precision: float = 0.1,
        **kwargs,
    ):
        """Initialise a wavelet class

        Args:
            fbounds: The bandpass frequency boundaries tuple
            norm:
            precision:
            kwargs: keyword arguments passed to [`Wavelet`][sim2xwavelet]
        """
        self.fbounds = fbounds
        super().__init__(**kwargs)
        wave = bandpass(self.nsamp, self.dt, *fbounds, norm=norm, precision=precision)
        self.set_wavelet(wave["time"], wave["amp"])
        self.wtype = "bandpass"


class OrmsbyWavelet(Wavelet):
    """An Ormsby wavelet

    Extends the base [`Wavelet`][sim2xwavelet] class.

    Attributes:
        fbounds (tuple): The bandpass boundaries tuple
    """

    def __init__(
        self, fbounds: Tuple[float, float, float, float], norm: bool = True, **kwargs
    ):
        """Initialise a wavelet class

        Args:
            fbounds: The Ormsby frequency boundaries tuple
            norm:
            kwargs: keyword arguments passed to [`Wavelet`][sim2xwavelet]
        """
        self.fbounds = fbounds
        super().__init__(**kwargs)
        wave = ormsby(self.nsamp, self.dt, *fbounds, norm=norm)
        self.set_wavelet(wave["time"], wave["amp"])
        self.wtype = "ormsby"


class PetrelWavelet(Wavelet):
    """A wavelet loaded from a Petrel file.

    Extends the base [`Wavelet`][sim2xwavelet] class.

    Attributes:
        file (str): The filepath of the loaded wavelet
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
        time = np.array(time, dtype=float)
        amp = np.array(amp, dtype=float)

        if norm:
            amp = amp / np.max(np.abs(amp))

        # if hflags["WAVELET-TFS"] is not None:
        #     self.time = self.time + float(hflags["WAVELET-TFS"])
        time = time / 1000.0

        self.set_wavelet(time, amp)
        self.wtype = "petrel"
