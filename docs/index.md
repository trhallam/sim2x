# Sim2x

`sim2x` is a Python framework for forward modelling reservoir simulation models. The forward workflow involves
three main processes, converting the simulation to impedance (`imp`), converting the simulation corner point grid to a regular grid (`rg`), and convolving a wavelet with a reflectivity model to generate synthetic seismic (`seis`).

`sim2x` builds upon the `eclx` and `digirock` libraries.

 - `eclx` is used for extracting simulation model data from Eclipse style output files.
 - `digirock` is used for the forward digital rock model.

If you write want to add to or extend `sim2x` please look at the project [contribution](contrib.md) guidelines.

## Quick Start

See the quick start example in the user guide.

## Installation

### Installing with `pip`

`sim2x` is available via `pip install`.

```
pip install sim2x
```

### Installing from source

Clone the repository

```
git clone http://github.com/trhallam/sim2x
```

and install using `pip`

```
cd sim2x
pip install .
```
