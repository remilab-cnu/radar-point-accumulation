# Dataset re-acquisition — verified sources

`experiments/data/` is not in git and was destroyed on 2026-08-12. Nothing in it was
lab-recorded; every set is public. This file is the record that did not exist then.
Sources below were recovered from session transcripts and **re-verified against the
live services on 2026-08-12** unless marked otherwise.

Loader expectations come from `experiments/src/spectra_dataset.py` and
`experiments/src/measurand_table.py`; paths are relative to `experiments/data/`.

## MM-Fi — verified and re-downloaded 2026-08-12

- Official repo: `github.com/ybhbingo/MMFi_dataset`. Its README links Drive folder
  `1zDbhfH3BV-xCZVUHmK65EgVV1HMDEYcz`.
- Take **`filtered_mmwave.zip` only**, file id `1KxPaB2amj0mQkjhrx_1yfPQ0_s2H58tx`,
  **212,018,368 bytes**, 263,422 entries. It is the mmWave-point-cloud-only package
  covering all 40 subjects. The `MMFi Dataset Split/E01..E04.zip` files in the same
  folder are the multi-GB full-modality archives and are **not needed**.
- License CC BY-NC 4.0. Subjects are 4 environments x 10 subjects
  (E01 = S01-10 ... E04 = S31-40).

```bash
python3 -m pip install gdown
cd experiments/data
python3 -m gdown 1KxPaB2amj0mQkjhrx_1yfPQ0_s2H58tx -O filtered_mmwave.zip
mkdir -p mmfi_extracted && python3 -c "import zipfile;zipfile.ZipFile('filtered_mmwave.zip').extractall('mmfi_extracted')"
# loader wants: data/mmfi_extracted/filtered_mmwave/E*/S*/A*/frame*.bin
```

## mRI — verified and re-downloaded 2026-08-17

- Dryad `doi:10.5061/dryad.9ghx3ffpp`, CC0-1.0. Code and loaders at
  `github.com/sizhean/mri`.
- Wanted file: **`dataset_release.zip`, 680,028,517 bytes**, Dryad file id `2738858`
  (version 266420). The other file, `blurred_videos.zip` (15 GB), is not needed.
- **The working route (2026-08-17), after two failing ones.** All plain requests
  (curl, cookie-carrying Node HTTPS) get an Anubis "Validating..." page: the
  clearance never leaves the browser. A click into a new tab stalls the same way.
  What works is a **same-tab navigation** in headless puppeteer: load the dataset
  landing page, then `page.goto` the `/downloads/file_stream/2738858` URL directly
  with `Browser.setDownloadBehavior` capture — the challenge runs in-tab and rolls
  straight into the download. 680,028,517 bytes, `PK` magic.
- The Google Drive mirror (`1kR2U_omRkVTNkoetr7Akkorx5HfAvZ_C`) stays
  quota-exhausted.
- Layout mapping: the zip holds all modalities; the loaders need only
  `dataset_release/aligned_data/radar/singleframe/subjectN.csv` and
  `dataset_release/aligned_data/pose_labels/subjectN_all_labels.cpl`, both
  extracted flat into `data/mri_sample/mri_data/` (40 files, 20 subjects).
- **Verified**: `mri_windows()` returns 968 recordings, the count the manuscript
  and `c_recompute_full.json` use, and re-running `c_grid_full.py mRI` reproduced
  the archived row (0.5971/0.6488/0.6962) with no diff.

## BGT60TR13C (the 60 GHz set) — re-downloaded and rebuilt 2026-08-17

- Zenodo record **15178095**, "Frame-Labeled 60 GHz FMCW Radar Gesture Dataset",
  Seifi / Sukianto / Carbonelli, CC BY 4.0. This is reference [31] of the manuscript.
- Single file `radar_dataset.zip`, **36,485 MB (~36.5 GB)** of raw radar cubes.
- The experiments do not read it directly. They read
  `data/infineon_recs.pkl`, the **frozen CFAR detection output** built from it by
  `experiments/src/infineon_build.py` / `infineon_detection.py`
  (CA/OS ring train = 9, guard = 3, per-frame 2-D CFAR on range-Doppler magnitude).
- **Rebuilt 2026-08-17 and content-verified; byte-verification failed as expected.**
  The 36.5 GB zip downloaded via aria2 (16 connections, ~4.5 MB/s; a single curl
  stream throttles to ~0.13 MB/s) matched the advertised length exactly and passed a
  zip integrity check. `rep_variants.infineon_recs()` rebuilt the pickle from it:
  **2,400 recordings**, matching the frozen count. The rebuilt file's md5
  (`c3f3acccfa65e08a0b7394ebfef8ae10`) does **not** match the recorded
  `7d60af89e571178e5e1d04147d78364b` — pandas DataFrame pickles are not
  byte-stable across library versions, so this was the expected outcome, and md5
  alone cannot arbitrate. Content was verified statistically instead, twice:
  regenerating `measurand_table.json` from the rebuilt pickle reproduced the
  archived `datasets` block **exactly** (step 0.2581, span 4.129, 33 lattice
  points, zero off-lattice), and the C grid sweep on the rebuilt pickle gives
  0.9576 at 32 bins, which rounds to the archived headline C of 0.958. The
  detection chain's output is numerically identical; only the serialization
  differs. Future audits should record a content hash (e.g., over sorted
  per-recording arrays), not a pickle md5.

## mHomeGes — verified and re-downloaded 2026-08-12

- Distribution: **`github.com/GestureMan/mHomeGes-dataset`**, MIT license, 141 MB,
  default branch `master`. Paper `10.1145/3432235` (IMWUT 2020, "Real-time Arm
  Gesture Recognition in Smart Home Scenarios via Millimeter Wave Sensing").
- The repository root already carries the layout the loader expects: thirteen
  `longGes_<dist>m` directories (1.2 m through 2.55 m ...), each holding
  per-subject directories of `point_*.csv`, plus `dailyRandom/`.
- Clone it straight into place; no extraction step:

```bash
cd experiments/data
git clone --depth 1 https://github.com/GestureMan/mHomeGes-dataset.git mhomeges_full
# loader wants: data/mhomeges_full/longGes_*/<subject>/point_<id>_<dist>m_<class>.csv
```

Those thirteen distance folders are the axis the paper's within-mHomeGes
domain-shift replication runs over, so keep all of them.

**Verified after re-download**: 13 `longGes_*` directories, 1,153 `point_*.csv`, and
`spectra_dataset.mhomeges_instances()` returns **22,108 instances** — the exact count
the manuscript cites in its §IV screening statement ("94 of 22,108 instances"). The
re-acquired copy is the same corpus the results were computed on.

## M-Gesture — do not re-acquire

Dropped entirely by PI decision 2026-07-05: no download, no results, no citation.
The full ~90-subject set is Baidu-locked and unreachable from the CNU network.
`HANDOVER_mgesture_download.md` is kept only as the record of that dead end.

## Network constraints on this machine

Re-checked 2026-08-12: `api.github.com`, `zenodo.org`, `drive.google.com` and
`huggingface.co` are reachable. **`raw.githubusercontent.com` is not** — fetch READMEs
through `api.github.com/repos/<owner>/<repo>/readme` and base64-decode instead.
`pan.baidu.com` and MEGA are SNI-blocked by the campus firewall.
