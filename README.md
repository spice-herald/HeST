# HeST

**Helium Simulation Toolkit** — Monte Carlo simulation of energy deposition and signal generation in superfluid helium detectors. Simulates the full chain from particle recoil to sensor signals: energy partitioning, photon and quasiparticle propagation, quantum evaporation, and collection by cryogenic sensors.

## Installation

```bash
git clone git@github.com:spice-herald/HeST.git
cd HeST
pip install .
```

Dependencies: `numpy`, `scipy`, `numba`, `qetpy`, `scikit-image`

## Quick start

```python
import HeST as H

detector = H.HeRALD_v1(fill_height=4.5)
signal = H.Simulate(detector, "ER", 1000, 0, 0, 2)  # 1 keV ER at (0,0,2) cm
# signal.energies[i]     — energy (eV) at sensor i
# signal.arrivalTimes[i] — arrival times (µs) at sensor i
```

## Physics model

### Energy partitioning

`GetQuanta(energy, interaction)` converts recoil energy into singlet photons, triplet molecules, IR photons, and quasiparticles. Yields are interpolated from digitized Hertel et al. (1810.06283) data for ER and NR interactions. Per-channel Fano factors control variance (`fano_singlet`, `fano_triplet`, `fano_IR`). QP energy is computed as the residual to conserve total energy.

IR quanta follow the Hertel/Seidel model: N_i = E/W (W = 43 eV) ionizations and N_ex = 0.45·N_i excitations, carrying 4 eV and 0.5 eV respectively.

The singlet:triplet branching ratio defaults to the energy-dependent Hertel curves but can be overridden via `singlet_fraction` to redistribute total UV energy between channels.

### Photon propagation

Four propagation engines track particles via ray marching through user-defined detector geometries:

**VUV photons** (singlet ~80 nm, triplet fluorescence) —
- Rayleigh scattering in LHe (configurable MFP, default 30 cm; Seidel et al. NIM A 489, 2002)
- Material-dependent reflection: separate probabilities for Cu walls and Si sensors (Palik 1985/1991)
- Diffuse vs. specular reflection per surface
- Snell's law and total internal reflection at the liquid/vacuum interface (n = 1.03), both upward and downward crossings
- Configurable VUV photon energy (default 16.0 eV, He₂* excimer emission)

**Triplet molecules** — diffuse to surfaces, then fluoresce with configurable yield (default 1.0). The resulting VUV photon is tracked with full photon physics (Rayleigh, refraction, material reflection).

**IR photons** — energy-dependent reflection: ionization channel (4 eV / 310 nm) and excitation channel (0.5 eV / 2.5 µm) use separate wall/sensor probabilities. Bulk absorption in LHe is negligible at both wavelengths.

**Gamma rays** — exponential path lengths with configurable mean free path.

### Quasiparticle propagation

Quasiparticles in superfluid helium with specular, diffuse, and Andreev reflection; quantum evaporation at the liquid surface (critical angle, evaporation probability, atom kinematics); detection by wet and dry sensors. Dispersion and velocity from Donnelly et al. data; momentum sampled from Bose-Einstein distributions.

## Modules

| Module | Purpose |
|---|---|
| `HeST_Core.py` | Energy partitioning, QP dispersion/velocity, momentum sampling |
| `Detection.py` | Propagation engines, detector/sensor geometry, simulation drivers |
| `Geometry.py` | Pre-built detector geometries (HeRALD v1, LBNL, UMass variants) |
| `WIMP_Generation.py` | WIMP recoil spectrum (differential rate, rejection sampling) |
| `analysis_functions.py` | Histograms, waveform generation, batch event extraction |

## Running simulations

### `simulate_spectrum.py`

Batch event simulation across an energy spectrum. Outputs `.npz` (summary) and optionally `.h5` (per-quanta raw data).

```bash
python simulate_spectrum.py --spectrum flat --recoil_type ER --n_events 10000 --flat_min 10 --flat_max 10000
python simulate_spectrum.py --spectrum mono --recoil_type NR --n_events 100 --energies 100,1000,5000
python simulate_spectrum.py --spectrum wimp --recoil_type NR --n_events 200 --wimp_mass 1000
```

Key flags: `--per_sensor` (per-sensor arrays), `--save_raw` (HDF5 per-quanta data for waveforms), `--fano_singlet/triplet/ir` (Fano factors), `--max_qp` (QP cap with reweighting).

Detector geometry, reflection probabilities, photon energies, Rayleigh MFP, fluorescence yield, and singlet fraction are set in the configuration block at the top of the script.

Runtime: ~10,000 events ≈ 1 hour on a single CPU core.

### Example notebooks

| Notebook | Description |
|---|---|
| `Channel_Signal_Distributions.ipynb` | Full analysis: yield curves, detector diagrams, 2D discrimination, arrival times |
| `Hest_basic.ipynb` | QP dispersion, critical angles, ER/NR yields, momentum sampling |
| `HeST_QP_Sim.ipynb` | Evaporation signal: trajectories, per-sensor maps, wall reflection comparisons |
| `HeST_Singlet_Sim.ipynb` | Singlet photon propagation: paths, hit positions, reflection comparisons |

## Map generation

Pre-computed light collection efficiency (LCE) and quasiparticle evaporation (QPE) maps accelerate simulations. Generate with `processMapsInParallel.py` and merge with `Merge2DMaps.py`. Compute-intensive — best run on a cluster.
