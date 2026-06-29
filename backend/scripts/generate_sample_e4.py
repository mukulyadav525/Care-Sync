"""
Generate a realistic sample Empatica-E4 session for local development / demo.

Usage (from backend/ with the venv active):
    python scripts/generate_sample_e4.py [username] [session_name] [hours]

Creates:  Users/<username>/<session_name>/{ACC,BVP,EDA,HR,IBI,TEMP}.csv,
          tags.csv, info.txt
"""
import os
import sys
import csv
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
USERS_DIR = os.path.join(os.path.dirname(HERE), 'Users')


def _write_standard(path, start, fs, values):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([f'{start:.2f}'])
        w.writerow([f'{fs:.6f}'])
        for v in values:
            w.writerow([f'{v:.4f}'])


def main():
    username = sys.argv[1] if len(sys.argv) > 1 else 'mukul'
    session = sys.argv[2] if len(sys.argv) > 2 else 'session_2025-06-26'
    hours = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0

    rng = np.random.default_rng(42)
    start = time.time() - hours * 3600  # session began `hours` ago
    out_dir = os.path.join(USERS_DIR, username, session)
    os.makedirs(out_dir, exist_ok=True)

    dur = hours * 3600
    # slow circadian-ish drift shared across signals
    t_min = np.arange(0, dur, 60) / 60.0
    drift = np.sin(t_min / (len(t_min) / 3.0) * np.pi)

    # ---- HR.csv @ 1 Hz -----------------------------------------------------
    fs_hr = 1.0
    n = int(dur * fs_hr)
    base = 72 + 8 * np.interp(np.arange(n), np.linspace(0, n, len(drift)), drift)
    hr = base + rng.normal(0, 2.5, n) + 6 * (rng.random(n) > 0.985)  # occasional spikes
    _write_standard(os.path.join(out_dir, 'HR.csv'), start + 10, fs_hr, np.clip(hr, 50, 160))

    # ---- EDA.csv @ 4 Hz ----------------------------------------------------
    fs_eda = 4.0
    n = int(dur * fs_eda)
    tonic = 2.0 + 0.6 * np.interp(np.arange(n), np.linspace(0, n, len(drift)), drift)
    phasic = np.zeros(n)
    for _ in range(int(hours * 12)):  # skin-conductance responses
        p = rng.integers(0, n)
        w = int(fs_eda * 6)
        k = np.exp(-np.arange(w) / (fs_eda * 1.5)) * rng.uniform(0.3, 1.2)
        phasic[p:p + len(k)] += k[:max(0, n - p)]
    eda = np.clip(tonic + phasic + rng.normal(0, 0.02, n), 0.05, None)
    _write_standard(os.path.join(out_dir, 'EDA.csv'), start, fs_eda, eda)

    # ---- TEMP.csv @ 4 Hz ---------------------------------------------------
    fs_t = 4.0
    n = int(dur * fs_t)
    temp = 33.5 + 1.2 * np.interp(np.arange(n), np.linspace(0, n, len(drift)), drift) \
        + rng.normal(0, 0.05, n)
    _write_standard(os.path.join(out_dir, 'TEMP.csv'), start, fs_t, temp)

    # ---- ACC.csv @ 32 Hz (x, y, z in 1/64 g) ------------------------------
    fs_acc = 32.0
    n = int(dur * fs_acc)
    gravity = 64  # ~1g resting on one axis
    activity = (rng.random(n) > 0.8).astype(float)
    x = gravity + rng.normal(0, 4, n) + activity * rng.normal(0, 25, n)
    y = rng.normal(0, 4, n) + activity * rng.normal(0, 25, n)
    z = rng.normal(0, 4, n) + activity * rng.normal(0, 25, n)
    with open(os.path.join(out_dir, 'ACC.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([f'{start:.2f}', f'{start:.2f}', f'{start:.2f}'])
        w.writerow([f'{fs_acc:.6f}', f'{fs_acc:.6f}', f'{fs_acc:.6f}'])
        for xi, yi, zi in zip(x, y, z):
            w.writerow([int(xi), int(yi), int(zi)])

    # ---- BVP.csv @ 64 Hz ---------------------------------------------------
    fs_bvp = 64.0
    n = int(dur * fs_bvp)
    tt = np.arange(n) / fs_bvp
    hr_hz = 72 / 60.0
    bvp = 80 * np.sin(2 * np.pi * hr_hz * tt) + 25 * np.sin(4 * np.pi * hr_hz * tt) \
        + rng.normal(0, 6, n)
    _write_standard(os.path.join(out_dir, 'BVP.csv'), start, fs_bvp, bvp)

    # ---- IBI.csv -----------------------------------------------------------
    rr = []
    t = 0.0
    while t < dur:
        ibi = rng.normal(60.0 / 72, 0.04)  # ~0.83 s
        rr.append((t, ibi))
        t += ibi
    with open(os.path.join(out_dir, 'IBI.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([f'{start:.2f}', ' IBI'])
        for ts, ibi in rr:
            w.writerow([f'{ts:.3f}', f'{ibi:.3f}'])

    # ---- tags.csv (event markers) -----------------------------------------
    with open(os.path.join(out_dir, 'tags.csv'), 'w', newline='') as f:
        for frac in (0.15, 0.45, 0.75):
            f.write(f'{start + dur * frac:.2f}\n')

    # ---- info.txt ----------------------------------------------------------
    with open(os.path.join(out_dir, 'info.txt'), 'w') as f:
        f.write(
            "Empatica E4 sample session (synthetic)\n"
            f"Duration: {hours:.1f} h\n"
            "ACC.csv  - accelerometer x,y,z @ 32 Hz (1/64 g)\n"
            "BVP.csv  - blood volume pulse @ 64 Hz\n"
            "EDA.csv  - electrodermal activity @ 4 Hz (microsiemens)\n"
            "HR.csv   - heart rate @ 1 Hz (bpm), starts 10 s after others\n"
            "IBI.csv  - inter-beat intervals (s)\n"
            "TEMP.csv - skin temperature @ 4 Hz (Celsius)\n"
            "tags.csv - event marker timestamps (unix, UTC)\n"
        )

    print(f"Sample session written to {out_dir}")


if __name__ == '__main__':
    main()
