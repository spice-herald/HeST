"""
simulate_spectrum.py — Per-channel HeST signal simulation across energy spectra.

Simulates singlet, triplet, IR, and quasiparticle/evaporation signals
independently for each event, recording per-channel detected energy,
quanta counts, and arrival times. Supports mono-energetic, flat,
WIMP, and arbitrary user-defined energy spectra.

Usage examples:
    python simulate_spectrum.py --spectrum mono --recoil_type ER --n_events 100 --energies 100,1000,5000
    python simulate_spectrum.py --spectrum flat --recoil_type NR --n_events 500 --flat_min 50 --flat_max 5000
    python simulate_spectrum.py --spectrum wimp --recoil_type NR --n_events 200 --wimp_mass 1000
    python simulate_spectrum.py --spectrum custom --recoil_type NR --spectrum_file my_spectrum.txt --n_events 300

key flags

--per_sensor --> save per sensor hits and timing
--max_qp 100000 --> set max QPs to propagate before reweighitng (default is 100K)
--save_raw --> save full per-quanta arrival times and energies to HDF5 (.h5) file

structure of npz file (always saved):
    recoil_energies      — (n_events,) float64, recoil energy in eV
    recoil_types         — (n_events,) int, 0 = ER, 1 = NR
    n_quanta_generated   — (n_events, 4) int, quanta produced per channel [singlet, triplet, IR, QP]
    detected_energy      — (n_events, 4) float64, total detected energy per channel (eV)
    detected_count       — (n_events, 4) float64, detected quanta per channel (weighted if downsampled)
    mean_arrival_time    — (n_events, 4) float64, mean arrival time per channel (µs), NaN if no hits
    spectrum_mode        — string, one of mono/flat/wimp/custom
    detector_config      — dict, snapshot of all configuration constants

    with --per_sensor:
        sensor_det_energy — (n_events, 4, n_sensors) float64, detected energy per channel per sensor
        sensor_det_count  — (n_events, 4, n_sensors) float64, detected count per channel per sensor
        sensor_det_time   — (n_events, 4, n_sensors) float64, mean arrival time per channel per sensor

    with --spectrum wimp:
        wimp_mass        — float64, WIMP mass in MeV
        wimp_mass_labels — (n_events,) float64, WIMP mass label per event

structure of h5 file (with --save_raw):
events/{event_index}/
    attrs: recoil_energy, recoil_type
    {channel}/                          # singlet, triplet, ir, qp
        attrs: weight                   # 1.0 normally, >1.0 if QP downsampled
        {sensor_index}/                 # only sensors with hits
            arrival_times   — float64 array (µs)
            energies        — float64 array (eV)

"""

import argparse
import time
import numpy as np
from tqdm import tqdm

import HeST as H

# ============================================================
# DETECTOR CONFIGURATION — edit this section before running
# ============================================================
DETECTOR_GEOMETRY = "HeRALD_v1"     # Options: HeRALD_v1, HeRALD_v1_monolithic,
                                    #          HeRALD_LBNL, HeRALD_UMass_splitCPD,
                                    #          HeRALD_UMass_monolithic
FILL_HEIGHT = 4.5                   # Helium fill height in cm
CELL_RADIUS = 3.5                   # Cell radius in cm (used for position sampling)

# QP channel parameters
QP_WALL_REFLECTION_PROB = 0.3       # Probability of QP reflecting off walls
QP_WALL_DIFFUSE_PROB = 0.0          # Fraction of QP reflections that are diffuse
QP_SENSOR_REFLECTION_PROB = 0.0     # Probability of QP reflecting off sensors

# Photon channel parameters
UV_WALL_REFLECTION_PROB = 0.3       # Probability of UV photon reflecting off walls
UV_WALL_DIFFUSE_PROB = 0.0          # Fraction of UV reflections that are diffuse
IR_WALL_REFLECTION_PROB = 0.0       # Probability of IR photon reflecting off walls
IR_WALL_DIFFUSE_PROB = 0.0          # Fraction of IR reflections that are diffuse

# Simulation parameters
STEP_SIZE = 0.05                    # Ray-marching step size in cm
MAX_DIST = 10.0                     # Maximum propagation distance in cm
QP_TEMPERATURE = 2.0                # Effective temperature for QP momentum sampling (K)
MAX_QP_SIMULATED = 100000           # Max QPs to propagate per event; extras are downsampled
                                    # and results scaled by weight = n_qp / MAX_QP_SIMULATED
# ============================================================


def get_detector_config():
    """Return a dict snapshot of all detector configuration constants."""
    return dict(
        detector_geometry=DETECTOR_GEOMETRY,
        fill_height=FILL_HEIGHT,
        cell_radius=CELL_RADIUS,
        qp_wall_reflection_prob=QP_WALL_REFLECTION_PROB,
        qp_wall_diffuse_prob=QP_WALL_DIFFUSE_PROB,
        qp_sensor_reflection_prob=QP_SENSOR_REFLECTION_PROB,
        uv_wall_reflection_prob=UV_WALL_REFLECTION_PROB,
        uv_wall_diffuse_prob=UV_WALL_DIFFUSE_PROB,
        ir_wall_reflection_prob=IR_WALL_REFLECTION_PROB,
        ir_wall_diffuse_prob=IR_WALL_DIFFUSE_PROB,
        step_size=STEP_SIZE,
        max_dist=MAX_DIST,
        qp_temperature=QP_TEMPERATURE,
        max_qp_simulated=MAX_QP_SIMULATED,
    )


def build_detector():
    """Construct and configure the VDetector from the configuration block."""
    geometry_map = {
        "HeRALD_v1": H.HeRALD_v1,
        "HeRALD_v1_monolithic": H.HeRALD_v1_monolithic,
        "HeRALD_LBNL": H.HeRALD_LBNL,
        "HeRALD_UMass_splitCPD": H.HeRALD_UMass_splitCPD,
        "HeRALD_UMass_monolithic": H.HeRALD_UMass_monolithic,
    }
    if DETECTOR_GEOMETRY not in geometry_map:
        raise ValueError(f"Unknown geometry '{DETECTOR_GEOMETRY}'. "
                         f"Options: {list(geometry_map.keys())}")

    detector = geometry_map[DETECTOR_GEOMETRY](fill_height=FILL_HEIGHT)

    detector.set_QP_wall_reflection_prob(QP_WALL_REFLECTION_PROB)
    detector.set_QP_wall_diffuse_prob(QP_WALL_DIFFUSE_PROB)
    detector.set_QP_sensor_reflection_prob(QP_SENSOR_REFLECTION_PROB)
    detector.set_UV_wall_reflection_prob(UV_WALL_REFLECTION_PROB)
    detector.set_UV_wall_diffuse_prob(UV_WALL_DIFFUSE_PROB)
    detector.set_IR_wall_reflection_prob(IR_WALL_REFLECTION_PROB)
    detector.set_IR_wall_diffuse_prob(IR_WALL_DIFFUSE_PROB)

    return detector


def random_position_in_cylinder(cell_radius, fill_height):
    """Sample a uniform random position inside the cylindrical liquid volume."""
    r = cell_radius * np.sqrt(np.random.random())
    phi = 2 * np.pi * np.random.random()
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.random.uniform(0, fill_height)
    return x, y, z


def simulate_event(detector, energy, recoil_type, save_per_sensor=False,
                   save_raw=False):
    """
    Simulate a single recoil event and return per-channel results.

    Returns
    -------
    n_generated : (4,) array — [singlet, triplet, IR, QP] quanta counts
    det_energy  : (4,) array — total detected energy per channel (eV)
    det_count   : (4,) array — number of detected quanta per channel
    mean_time   : (4,) array — mean arrival time per channel (us), NaN if no hits
    sensor_energy : (4, nsensors) array or None — per-sensor detected energy
    sensor_count  : (4, nsensors) array or None — per-sensor detected count
    sensor_time   : (4, nsensors) array or None — per-sensor mean arrival time
    raw_signals : list of (HestSignal, weight) or None — raw signal objects per channel
    """
    quanta = H.GetQuanta(energy, recoil_type, T=QP_TEMPERATURE)
    n_singlet = quanta.get_nSingletPhotons()
    n_triplet = quanta.get_nTripletMolecules()
    n_ir = quanta.get_nIRPhotons()
    n_qp = quanta.get_nQuasiparticles()

    n_generated = np.array([n_singlet, n_triplet, n_ir, n_qp])
    det_energy = np.zeros(4)
    det_count = np.zeros(4)
    mean_time = np.full(4, np.nan)

    x, y, z = random_position_in_cylinder(CELL_RADIUS, FILL_HEIGHT)
    nsensors = detector.get_nsensors()

    if save_per_sensor:
        sensor_energy = np.zeros((4, nsensors))
        sensor_count = np.zeros((4, nsensors))
        sensor_time = np.full((4, nsensors), np.nan)
    else:
        sensor_energy = None
        sensor_count = None
        sensor_time = None

    if save_raw:
        raw_signals = [None, None, None, None]
    else:
        raw_signals = None

    sim_kwargs = dict(max_dist=MAX_DIST, step_size=STEP_SIZE)

    def collect_signal(sig, ch_index, weight=1.0):
        """Extract detected energy, count, and mean time from a HestSignal."""
        for s in range(nsensors):
            if len(sig.energies[s]) > 0:
                e_s = np.sum(sig.energies[s]) * weight
                c_s = len(sig.energies[s]) * weight
                det_energy[ch_index] += e_s
                det_count[ch_index] += c_s
                if save_per_sensor:
                    sensor_energy[ch_index, s] = e_s
                    sensor_count[ch_index, s] = c_s
            if save_per_sensor and len(sig.arrivalTimes[s]) > 0:
                sensor_time[ch_index, s] = np.mean(sig.arrivalTimes[s])
        time_arrays = [sig.arrivalTimes[s] for s in range(nsensors)
                       if len(sig.arrivalTimes[s]) > 0]
        if time_arrays:
            all_times = np.concatenate(time_arrays)
            if len(all_times) > 0:
                mean_time[ch_index] = np.mean(all_times)

    if n_singlet > 0:
        sig = H.GetSingletSignal(detector, n_singlet, x, y, z, **sim_kwargs)
        collect_signal(sig, 0)
        if save_raw:
            raw_signals[0] = (sig, 1.0)

    if n_triplet > 0:
        sig = H.GetTripletSignal(detector, n_triplet, x, y, z, **sim_kwargs)
        collect_signal(sig, 1)
        if save_raw:
            raw_signals[1] = (sig, 1.0)

    if n_ir > 0:
        sig = H.GetIRSignal(detector, n_ir, x, y, z, **sim_kwargs)
        collect_signal(sig, 2)
        if save_raw:
            raw_signals[2] = (sig, 1.0)

    if n_qp > 0:
        if n_qp > MAX_QP_SIMULATED:
            n_qp_sim = MAX_QP_SIMULATED
            qp_weight = n_qp / MAX_QP_SIMULATED
        else:
            n_qp_sim = n_qp
            qp_weight = 1.0
        sig = H.GetEvaporationSignal(detector, n_qp_sim, x, y, z,
                                     T=QP_TEMPERATURE, **sim_kwargs)
        collect_signal(sig, 3, weight=qp_weight)
        if save_raw:
            raw_signals[3] = (sig, qp_weight)

    return n_generated, det_energy, det_count, mean_time, sensor_energy, sensor_count, sensor_time, raw_signals


def generate_energies(args):
    """
    Generate (energy, recoil_type, wimp_mass_label) tuples for all events.

    Returns
    -------
    events : list of (energy_eV, recoil_type_str, wimp_mass_or_nan)
    """
    events = []

    rtype = args.recoil_type

    if args.spectrum == "mono":
        if args.energies is not None:
            energy_grid = np.array([float(e) for e in args.energies.split(",")])
        else:
            energy_grid = np.geomspace(20, 10000, 20)
        for e in energy_grid:
            for _ in range(args.n_events):
                events.append((e, rtype, np.nan))

    elif args.spectrum == "flat":
        energies = np.random.uniform(args.flat_min, args.flat_max,
                                     size=args.n_events)
        for e in energies:
            events.append((e, rtype, np.nan))

    elif args.spectrum == "wimp":
        mass = args.wimp_mass
        minE, maxE, maxY = H.WIMP_spectrum_prep(mass)
        for _ in range(args.n_events):
            e = H.WIMP_spectrum(mass, minE, maxE, maxY)
            events.append((e, rtype, mass))

    elif args.spectrum == "custom":
        if args.spectrum_file is None:
            raise ValueError("--spectrum_file required for custom mode")
        data = np.loadtxt(args.spectrum_file)
        energies_table = data[:, 0]
        rates_table = data[:, 1]
        cdf = np.cumsum(rates_table)
        cdf = cdf / cdf[-1]
        uniform_samples = np.random.random(args.n_events)
        sampled_energies = np.interp(uniform_samples, cdf, energies_table)
        for e in sampled_energies:
            events.append((e, rtype, np.nan))

    else:
        raise ValueError(f"Unknown spectrum mode: {args.spectrum}")

    return events


def generate_output_name(args):
    """Build a descriptive output filename from the run parameters."""
    geom_short = DETECTOR_GEOMETRY.replace("HeRALD_", "Herald")
    parts = [f"Spectrum_{args.spectrum.capitalize()}_{args.recoil_type}"]

    if args.spectrum == "mono":
        if args.energies is not None:
            e_list = [float(e) for e in args.energies.split(",")]
        else:
            e_list = list(np.geomspace(20, 10000, 20))
        parts.append(f"energyMin_{int(min(e_list))}_energyMax_{int(max(e_list))}")
    elif args.spectrum == "flat":
        parts.append(f"energyMin_{int(args.flat_min)}_energyMax_{int(args.flat_max)}")
    elif args.spectrum == "wimp":
        parts.append(f"wimpMass_{int(args.wimp_mass)}MeV")
    elif args.spectrum == "custom":
        import os
        base = os.path.splitext(os.path.basename(args.spectrum_file))[0]
        parts.append(f"file_{base}")

    parts.append(f"nEvents_{args.n_events}")
    parts.append(geom_short)

    return "_".join(parts) + ".npz"


def main():
    parser = argparse.ArgumentParser(
        description="Per-channel HeST signal simulation across energy spectra.")
    parser.add_argument("--n_events", type=int, default=100,
                        help="MC events per energy point (mono) or total (other modes)")
    parser.add_argument("--recoil_type", required=True,
                        choices=["ER", "NR"],
                        help="Recoil type: ER (electronic) or NR (nuclear)")
    parser.add_argument("--spectrum", default="mono",
                        choices=["mono", "flat", "wimp", "custom"],
                        help="Energy spectrum mode")
    parser.add_argument("--energies", default=None,
                        help="Comma-separated energies in eV (mono mode)")
    parser.add_argument("--flat_min", type=float, default=20,
                        help="Min energy in eV (flat mode)")
    parser.add_argument("--flat_max", type=float, default=10000,
                        help="Max energy in eV (flat mode)")
    parser.add_argument("--wimp_mass", type=float, default=1000,
                        help="WIMP mass in MeV (wimp mode)")
    parser.add_argument("--spectrum_file", default=None,
                        help="Two-column text file: energy_eV rate (custom mode)")
    parser.add_argument("--per_sensor", action="store_true",
                        help="Save per-sensor energy and count arrays (shape: n_events x 4 x n_sensors)")
    parser.add_argument("--save_raw", action="store_true",
                        help="Save full per-quanta arrival times and energies to HDF5 (.h5) file")
    parser.add_argument("--max_qp", type=int, default=None,
                        help="Max QPs to simulate per event (overrides config block)")
    parser.add_argument("--output", default=None,
                        help="Output file path (auto-generated from parameters if omitted)")
    args = parser.parse_args()

    global MAX_QP_SIMULATED
    if args.max_qp is not None:
        MAX_QP_SIMULATED = args.max_qp

    if args.spectrum == "wimp" and args.recoil_type != "NR":
        print("Warning: WIMP spectrum forces recoil_type=NR, ignoring --recoil_type")
        args.recoil_type = "NR"

    if args.output is None:
        args.output = generate_output_name(args)
    print(f"Output file: {args.output}")

    print(f"Building detector: {DETECTOR_GEOMETRY} (fill_height={FILL_HEIGHT} cm)")
    detector = build_detector()
    nsensors = detector.get_nsensors()
    print(f"  {nsensors} sensors configured")
    print(f"  Max QPs per event: {MAX_QP_SIMULATED} (downsample & weight if exceeded)")

    print(f"Generating event list (spectrum={args.spectrum})...")
    events = generate_energies(args)
    n_total = len(events)
    print(f"  {n_total} total events to simulate")

    # Warmup run for time estimate
    print("Running warmup event for time estimate...")
    t0 = time.time()
    simulate_event(detector, 1000.0, "ER")
    warmup_time = time.time() - t0
    est_minutes = warmup_time * n_total / 60
    print(f"  ~{warmup_time:.1f}s per event, estimated total: {est_minutes:.0f} minutes")

    # Allocate output arrays
    recoil_energies = np.zeros(n_total)
    recoil_types = np.zeros(n_total, dtype=int)
    n_quanta_generated = np.zeros((n_total, 4), dtype=int)
    detected_energy = np.zeros((n_total, 4))
    detected_count = np.zeros((n_total, 4))
    mean_arrival_time = np.full((n_total, 4), np.nan)
    wimp_mass_labels = np.full(n_total, np.nan)

    if args.per_sensor:
        sensor_det_energy = np.zeros((n_total, 4, nsensors))
        sensor_det_count = np.zeros((n_total, 4, nsensors))
        sensor_det_time = np.full((n_total, 4, nsensors), np.nan)

    # Set up HDF5 file if --save_raw
    h5_file = None
    h5_path = None
    channel_keys = ['singlet', 'triplet', 'ir', 'qp']
    if args.save_raw:
        import h5py
        h5_path = args.output.replace('.npz', '.h5')
        h5_file = h5py.File(h5_path, 'w')
        h5_file.attrs['n_events'] = n_total
        h5_file.attrs['spectrum_mode'] = args.spectrum
        h5_file.attrs['detector_geometry'] = DETECTOR_GEOMETRY
        h5_file.attrs['n_sensors'] = nsensors
        h5_events = h5_file.create_group('events')

    print("Running simulation...")
    for i, (energy, rtype, wmass) in enumerate(tqdm(events)):
        recoil_energies[i] = energy
        recoil_types[i] = 0 if rtype == "ER" else 1
        wimp_mass_labels[i] = wmass

        n_gen, det_e, det_c, m_time, s_e, s_c, s_t, raw = simulate_event(
            detector, energy, rtype, save_per_sensor=args.per_sensor,
            save_raw=args.save_raw)
        n_quanta_generated[i] = n_gen
        detected_energy[i] = det_e
        detected_count[i] = det_c
        mean_arrival_time[i] = m_time
        if args.per_sensor:
            sensor_det_energy[i] = s_e
            sensor_det_count[i] = s_c
            sensor_det_time[i] = s_t

        if args.save_raw and raw is not None:
            evt_grp = h5_events.create_group(str(i))
            evt_grp.attrs['recoil_energy'] = energy
            evt_grp.attrs['recoil_type'] = rtype
            for ch, ch_key in enumerate(channel_keys):
                if raw[ch] is None:
                    continue
                sig, weight = raw[ch]
                ch_grp = evt_grp.create_group(ch_key)
                ch_grp.attrs['weight'] = weight
                for s in range(nsensors):
                    if len(sig.arrivalTimes[s]) > 0:
                        s_grp = ch_grp.create_group(str(s))
                        s_grp.create_dataset('arrival_times', data=sig.arrivalTimes[s])
                        s_grp.create_dataset('energies', data=sig.energies[s])

    if h5_file is not None:
        h5_file.close()
        print(f"Raw per-quanta data saved to {h5_path}")

    # Build save dict
    save_dict = dict(
        recoil_energies=recoil_energies,
        recoil_types=recoil_types,
        n_quanta_generated=n_quanta_generated,
        detected_energy=detected_energy,
        detected_count=detected_count,
        mean_arrival_time=mean_arrival_time,
        spectrum_mode=np.array(args.spectrum),
        detector_config=np.array(get_detector_config()),
    )

    if args.per_sensor:
        save_dict["sensor_det_energy"] = sensor_det_energy
        save_dict["sensor_det_count"] = sensor_det_count
        save_dict["sensor_det_time"] = sensor_det_time

    if args.spectrum == "wimp":
        save_dict["wimp_mass"] = np.array(args.wimp_mass)
        save_dict["wimp_mass_labels"] = wimp_mass_labels

    np.savez(args.output, **save_dict)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
