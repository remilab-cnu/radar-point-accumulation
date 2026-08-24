"""FMCW-MIMO detection chain for the Infineon BGT60TR13C 60 GHz gesture dataset.

Raw cube per recording: (F frames, 3 RX, 32 chirps, 64 ADC samples), real ADC.
Per frame: Range-FFT -> Doppler-FFT -> 2D CA-CFAR -> phase-monopulse angle from the
3-RX L-array -> per-detection (x, y, z, radial velocity, intensity).
Output: a canonical point-cloud DataFrame (frame,x,y,z,doppler,intensity) that plugs
straight into segment_instances / make_channels like the TI datasets.

Sensor params (Zenodo 15178095 / BGT60TR13C):
  fc≈60.5 GHz, BW=4 GHz -> range res 0.0375 m; 64 samples; 32 chirps; PRT=300 us;
  33 Hz frame rate; 3 RX in L-shape, lambda/2 spacing (lambda≈4.96 mm).
Antenna (L-array, 0-indexed rx0,rx1,rx2): azimuth baseline (rx0,rx2), elevation (rx1,rx2).
Signs are validated against a known lateral gesture and flipped via AZ_SIGN/EL_SIGN.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

C = 299792458.0
FC = 60.5e9
LAMBDA = C / FC
BW = 4e9
RANGE_RES = C / (2 * BW)            # 0.0375 m
N_SAMP, N_CHIRP, N_RX = 64, 32, 3
PRT = 300e-6
V_MAX = LAMBDA / (4 * PRT)          # ~4.14 m/s
V_RES = LAMBDA / (2 * N_CHIRP * PRT)
D_OVER_LAMBDA = 0.5                 # lambda/2 spacing -> 2*pi*d/lambda = pi
AZ_PAIR, EL_PAIR = (0, 2), (1, 2)
AZ_SIGN, EL_SIGN = +1.0, +1.0

# CA-CFAR params. alpha=2.5 chosen by spectra-quality inspection (infineon_viz):
# gesture-concentrated, clean background, ~8 pts/frame avg (higher on active frames)
# -> matches the sparse-array datasets' density without injecting clutter. A 3-RX
# 60 GHz sensor is physically sparser than the IWR1443/6843 sets (~46 pts/frame);
# forcing that density loosens CFAR into noise.
CFAR_TRAIN, CFAR_GUARD, CFAR_ALPHA = 9, 3, 2.5
RANGE_MIN_BIN, RANGE_MAX_BIN = 1, 33   # drop DC bin; keep near range


def _rd_cube(frame):
    """frame: (3,32,64) real -> complex range-Doppler per RX: (3, 32 doppler, R range)."""
    x = frame.astype(np.float64)
    x = x - x.mean(axis=2, keepdims=True)                 # remove per-chirp DC
    rwin = np.hanning(N_SAMP)[None, None, :]
    R = np.fft.rfft(x * rwin, axis=2)                     # (3,32,33) complex, positive ranges
    R = R - R.mean(axis=1, keepdims=True)                 # remove static (zero-Doppler clutter)
    dwin = np.hanning(N_CHIRP)[None, :, None]
    RD = np.fft.fftshift(np.fft.fft(R * dwin, axis=1), axes=1)   # (3,32,33) Doppler-centered
    return RD


def _ca_ring_noise(mag, train, guard):
    n_all, n_g = train ** 2, guard ** 2
    m_all = uniform_filter(mag, train, mode="nearest")
    m_g = uniform_filter(mag, guard, mode="nearest")
    return (m_all * n_all - m_g * n_g) / (n_all - n_g)


def _detect(mag, method="ca", alpha=4.0, topk=15, train=9, guard=3):
    """Return boolean detection mask over the (Doppler x Range) magnitude map.
    method: 'ca' CA-CFAR (mean ring x alpha), 'os' ordered-statistic CFAR
    (percentile ring x alpha), 'topk' per-frame K strongest cells above a CA floor,
    'pct' global percentile threshold (alpha = percentile, e.g. 99)."""
    if method == "ca":
        return mag > alpha * _ca_ring_noise(mag, train, guard)
    if method == "os":
        from scipy.ndimage import percentile_filter
        return mag > alpha * percentile_filter(mag, 75, size=train, mode="nearest")
    if method == "pct":
        return mag > np.percentile(mag, alpha)
    if method == "topk":
        floor = 2.0 * _ca_ring_noise(mag, train, guard)     # reject clear noise
        cand = mag * (mag > floor)
        if not np.any(cand):
            return np.zeros_like(mag, dtype=bool)
        thr = np.sort(cand.ravel())[::-1][min(topk, cand.size) - 1]
        return cand >= max(thr, 1e-9)
    raise ValueError(method)


def process_recording(cube, roi=None, method="ca", alpha=CFAR_ALPHA, topk=15, train=9, guard=3):
    """cube: (F,3,32,64) -> point-cloud DataFrame (frame,x,y,z,doppler,intensity)."""
    rows = []
    dopp_idx = np.arange(N_CHIRP) - N_CHIRP // 2          # centered Doppler bins
    for f in range(cube.shape[0]):
        RD = _rd_cube(cube[f])                            # (3,32,R)
        mag = np.abs(RD).sum(0)                           # (32,R) non-coherent RX sum
        noise = _ca_ring_noise(mag, train, guard)         # local noise floor (for CFAR-SNR)
        det = _detect(mag, method=method, alpha=alpha, topk=topk, train=train, guard=guard)
        det[:, :RANGE_MIN_BIN] = False
        det[:, RANGE_MAX_BIN:] = False
        dd, rr = np.where(det)
        for d, r in zip(dd, rr):
            rng = r * RANGE_RES
            v = dopp_idx[d] * (2 * V_MAX / N_CHIRP)
            c = RD[:, d, r]
            daz = np.angle(c[AZ_PAIR[0]] * np.conj(c[AZ_PAIR[1]]))
            dele = np.angle(c[EL_PAIR[0]] * np.conj(c[EL_PAIR[1]]))
            saz = np.clip(AZ_SIGN * daz / (2 * np.pi * D_OVER_LAMBDA), -1, 1)
            sel = np.clip(EL_SIGN * dele / (2 * np.pi * D_OVER_LAMBDA), -1, 1)
            az = np.arcsin(saz); el = np.arcsin(sel)
            x = rng * np.cos(el) * np.sin(az)
            y = rng * np.cos(el) * np.cos(az)
            z = rng * np.sin(el)
            snr = float(mag[d, r] / max(noise[d, r], 1e-12))   # CFAR-SNR (TI-like semantics)
            rows.append((f, x, y, z, v, float(mag[d, r]), snr))
    df = pd.DataFrame(rows, columns=["frame", "x", "y", "z", "doppler", "intensity", "intensity_snr"])
    if roi:
        for k, (lo, hi) in roi.items():
            df = df[(df[k] >= lo) & (df[k] <= hi)]
    return df.reset_index(drop=True)


if __name__ == "__main__":
    import os
    HERE = os.path.dirname(os.path.abspath(__file__))
    samp = os.path.join(HERE, "..", "data", "infineon_raw_sample", "user10_e1_recording0.npy")
    cube = np.load(samp)
    print("cube shape", cube.shape, "dtype", cube.dtype, "| range_res=%.4fm v_max=%.2f v_res=%.3f"
          % (RANGE_RES, V_MAX, V_RES))
    df = process_recording(cube)
    print("detected points:", len(df), "over", cube.shape[0], "frames  (%.1f pts/frame)" % (len(df)/cube.shape[0]))
    if len(df):
        for k in ("x", "y", "z", "doppler", "intensity"):
            a = df[k].values
            print(f"  {k:9s}: min={a.min():.3f} p50={np.percentile(a,50):.3f} max={a.max():.3f}")
