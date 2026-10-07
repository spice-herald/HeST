# HeST

**Helium Simulation Toolkit** — a fast, Python-based Monte Carlo simulation for modeling energy deposition and signal generation in superfluid helium detectors.

HeST simulates the full chain from a particle recoil in liquid helium to the signals observed in cryogenic sensors: energy partitioning into excitation channels, quasiparticle and photon propagation through a user-defined detector geometry, quantum evaporation of helium atoms, and collection by an array of virtual sensors.

## Installation

### From source (recommended)

```bash
git clone git@github.com:spice-herald/HeST.git
cd HeST
pip install .
```

### From pip

```bash
pip install HeST==0.2.0
```

### Dependencies

`numpy`, `scipy`, `numba`, `qetpy`, `scikit-image` (for 3D detector visualization)

## Quick start

```python
import HeST as H

# Build a detector geometry
detector = H.HeRALD_v1(fill_height=4.5)

# Simulate a 1 keV electronic recoil at position (0, 0, 2) cm
signal = H.Simulate(detector, "ER", 1000, 0, 0, 2)

# signal.energies[i]      — energy deposits (eV) at sensor i
# signal.arrivalTimes[i]  — arrival times (us) at sensor i
```

## Architecture

### Core modules

| Module | Purpose |
|---|---|
| `HeST_Core.py` | Energy channel partitioning (singlet, triplet, QP, IR yields for ER and NR), quasiparticle dispersion/velocity curves, momentum sampling from Bose-Einstein distributions |
| `Detection.py` | Particle propagation engines, detector/sensor geometry classes, and top-level simulation drivers |
| `Geometry.py` | Pre-defined detector geometries (HeRALD v1, LBNL, UMass variants) |
| `WIMP_Generation.py` | WIMP recoil spectrum generator (differential rate, rejection sampling) |

### Analysis

| Module | Purpose |
|---|---|
| `analysis_functions.py` | Histogram plotting by bounce count and QP flavor, waveform generation via template convolution, batch event extraction |

## Key capabilities

### Energy partitioning

`GetQuanta(energy, interaction)` converts a recoil energy (eV) into counts of singlet photons, triplet molecules, IR photons, and quasiparticles. Fractional yields for electronic and nuclear recoils are interpolated from Hertel et al. data. Quasiparticle counts assume a Bose-Einstein momentum distribution at an effective temperature (default 2 K).

### Particle propagation

Four independent propagation engines track particles through the detector geometry via ray marching:

- **`QP_propagation`** — Quasiparticles in superfluid helium. Handles specular, diffuse, and Andreev reflection off walls; quantum evaporation at the liquid surface (critical angle calculation, evaporation probability, atom kinematics); detection by wet (submerged) and dry sensors.
- **`photon_propagation`** — UV singlet and IR photons. Tracks reflections off walls and sensors with configurable probabilities.
- **`triplet_propagation`** — Triplet molecules that diffuse to surfaces. Models fluorescent quenching on inert walls (converting to a 16 eV photon) followed by photon tracking.
- **`gamma_propagation`** — Simplified gamma ray transport using exponential path lengths and a mean free path.

### Signal generation

High-level functions wrap the propagation engines and return `HestSignal` objects (per-sensor energy deposits and arrival times):

- `GetEvaporationSignal()` — quasiparticle channel
- `GetSingletSignal()` — singlet photon channel
- `GetTripletSignal()` — triplet molecule channel
- `GetIRSignal()` — IR photon channel
- `Simulate()` — runs all four channels end-to-end for a given recoil

### Detector geometry

Detectors are built from two classes:

- **`VSensor`** — defines a single sensor's geometry (boundary condition function), phonon collection efficiency, and adsorption gain.
- **`VDetector`** — assembles a full detector from boundary conditions (top, bottom, wall, liquid surface), a list of sensors, and reflection/absorption probabilities for each particle type and surface.

Pre-built geometries in `Geometry.py`:

| Geometry | Description |
|---|---|
| `HeRALD_v1` | 6x6 hex-packed 1 cm sensor array on top, 3.5 cm radius cell |
| `HeRALD_v1_monolithic` | Same cell with a single monolithic top sensor |
| `HeRALD_LBNL` | LBNL design with 2x2 sensor arrays on top and bottom |
| `HeRALD_UMass_splitCPD` | UMass design with a split (2-channel) top sensor |
| `HeRALD_UMass_monolithic` | UMass design with a single top sensor |

All boundary conditions use `numba.njit` for performance.

### WIMP spectrum

`WIMP_dRate(recoilEnergy_eV, mass_MeV)` computes the differential WIMP-helium scattering rate using standard halo model parameters (ported from NEST). `WIMP_spectrum()` performs rejection sampling to draw recoil energies from this distribution.

### Quasiparticle physics

- **Dispersion and velocity curves** — cubic spline interpolations of Donnelly et al. data (`QP_dispersion`, `QP_velocity`)
- **Momentum sampling** — `Random_QPmomentum()` draws from a Bose-Einstein distribution at a given effective temperature
- **Evaporation probability** — `evap_prob()` implements a momentum-dependent evaporation probability with separate phonon and R+ scales
- **Critical angle** — `critical_angle()` computes the maximum angle of incidence for quantum evaporation based on energy/momentum conservation
- **Andreev reflection** — `Andreev_reflection()` models quasiparticle flavor conversion (phonon, R-, R+) at surfaces

## Running simulations

### `simulate_spectrum.py` — batch event simulation

The main simulation driver. Generates per-channel (singlet, triplet, IR, quasiparticle) signals for many events across an energy spectrum, saving results to `.npz` (summary) and optionally `.h5` (raw per-quanta data) files.

**Runtime:** ~10,000 events takes approximately 1 hour on a single CPU core.

#### Basic usage

```bash
# Flat ER or NR spectrum, 10k events, 10 eV to 10 keV
python simulate_spectrum.py --spectrum flat --recoil_type ER --n_events 10000 --flat_min 10 --flat_max 10000

# Mono-energetic NR at specific energies
python simulate_spectrum.py --spectrum mono --recoil_type NR --n_events 100 --energies 100,1000,5000

# WIMP spectrum (forces NR)
python simulate_spectrum.py --spectrum wimp --recoil_type NR --n_events 200 --wimp_mass 1000

# Custom spectrum from a two-column text file (energy_eV, rate)
python simulate_spectrum.py --spectrum custom --recoil_type NR --spectrum_file my_spectrum.txt --n_events 300
```

#### Key options

| Flag | Description |
|---|---|
| `--spectrum` | Energy spectrum mode: `mono`, `flat`, `wimp`, `custom` |
| `--recoil_type` | `ER` (electronic) or `NR` (nuclear) |
| `--n_events` | Number of events per energy point (mono) or total (other modes) |
| `--per_sensor` | Save per-sensor hit energy, count, and timing arrays. Good to save! |
| `--save_raw` | Save full per-quanta arrival times and energies to HDF5. Needed for waveform simulation! |
| `--max_qp` | Cap on QPs propagated per event (default 100,000; excess are downsampled and reweighted) |
| `--fano_singlet`, `--fano_triplet`, `--fano_ir` | Per-channel Fano factors (default 1.0 = Poisson) |
| `--output` | Output file path (auto-generated if omitted) |

#### Configuration

Detector geometry, fill height, reflection probabilities, and other physics parameters are set in the configuration block at the top of the script. Edit these before running.

#### Output format

The `.npz` file contains per-event arrays: `recoil_energies`, `n_quanta_generated` (shape `n_events × 4`), `detected_energy`, `detected_count`, `mean_arrival_time`, and the full `detector_config` dict. With `--per_sensor`, additional `sensor_det_energy/count/time` arrays of shape `n_events × 4 × n_sensors` are included. With `--save_raw`, a companion `.h5` file stores the full per-quanta arrival time and energy arrays organized by event and channel.

### `Example_notebooks/Channel_Signal_Distributions.ipynb` — analysis

Jupyter notebook for visualizing simulation output. Loads `.npz` files produced by `simulate_spectrum.py` and generates:

- Theoretical yield curves (ER and NR energy fractions vs. recoil energy)
- Detector geometry diagram (cross-section and top-down sensor layout)
- Per-channel detected energy distributions and 2D scatter plots (singlet vs. QP, with recoil energy contours)
- Channel detection efficiency vs. recoil energy
- Arrival time distributions (per-channel and per-event from `.h5` raw data)
- ER vs. NR discrimination plots with calorimeter resolution smearing

Set `DATA_FILE` at the top of the notebook to point to your `.npz` output.

## Example notebooks

| Notebook | Description |
|---|---|
| `Example_notebooks/Channel_Signal_Distributions.ipynb` | Full analysis of `simulate_spectrum.py` output — yield curves, detector diagrams, 2D discrimination plots, arrival times |
| `Example_notebooks/Hest_basic.ipynb` | QP dispersion curves, critical angles, ER/NR yield fractions, momentum sampling distributions |
| `Example_notebooks/HeST_QP_Sim.ipynb` | Evaporation signal simulation — single-QP trajectories, million-QP statistics, per-sensor energy maps, wall reflection comparisons, speed profiling |
| `Example_notebooks/HeST_Singlet_Sim.ipynb` | Singlet photon propagation — single-photon paths, hit-position scatter plots, reflection probability comparisons |
| `Example_notebooks/Potential_Curves.ipynb` | Helium potential energy curve analysis |

## Map generation

Detector geometries can use pre-computed light collection efficiency (LCE) and quasiparticle evaporation (QPE) maps to accelerate simulations. These are 3D arrays mapping spatial positions to per-sensor detection probabilities.

To generate maps:

1. Configure detector geometry in `LCEmap_2DmapFromZ.py` / `QPEmap_2DmapFromZ.py`
2. Set Z binning in `processMapsInParallel.py`
3. Run: `python processMapsInParallel.py LCEmap_2DmapFromZ`
4. Merge outputs: `python Merge2DMaps.py`

Map generation is compute-intensive and best run on a cluster (e.g., NERSC).

