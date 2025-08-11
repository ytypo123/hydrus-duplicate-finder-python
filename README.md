# Hydrus Duplicate Finder (Python)

A Python port of a MATLAB duplicate-finder pipeline, designed to work with the **Hydrus Client API** to find potential duplicate video files based on:

- Duration
- Aspect ratio
- Perceptual hash (aHash)
- Image similarity (SSIM / MS-SSIM)

---

## Features

- **Duration & Aspect Ratio filtering** – quickly narrows down candidates.
- **aHash prefilter** – optional Hamming distance filter for faster comparisons.
- **SSIM / MS-SSIM** – configurable similarity threshold (default **0.6** recommended).
- **Progress bars** – via `tqdm` (can be disabled).
- **Verbose mode** – see exactly what’s happening at each stage.
- **Hydrus integration** – posts potential duplicates directly to the Hydrus client.

---

## Requirements

- **Python 3.8+**
- Hydrus client running with API enabled

### Dependencies

```bash
pip install requests pillow opencv-python scikit-image tqdm
```

**Optional** (for MS-SSIM):

```bash
pip install torch pytorch-msssim
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/hydrus-duplicate-finder-python.git
cd hydrus-duplicate-finder-python
```

### 2. Create and activate a virtual environment *(recommended)*
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create and configure `hydrus_config.json`
```json
{
  "api_key": "your-hydrus-api-key-here",
  "api_url": "http://127.0.0.1",
  "api_port": 42001,
  "tags": ["system:filetype is video", "meta:pd"],
  "verbose": false,
  "progress": {
    "enabled": true,
    "type": "tqdm"
  },
  "filters": {
    "use_ar_filter": true,
    "use_ahash_prefilter": true
  },
  "candidate_generation": {
    "tolSeconds": 1.0,
    "arTol": 0.05
  },
  "ahash": {
    "resize": [64, 64],
    "hammingMax": 32
  },
  "preprocessing": {
    "fixedResize": [256, 256],
    "toGray": true,
    "preBlurSigma": 0.6,
    "histEq": false,
    "useMSSSIM": true,
    "centerFrac": 0.90,
    "trimStrength": 0.15,
    "minKeepFrac": 0.70
  },
  "similarity_threshold": 0.60,
  "post": {
    "batch_size": 200,
    "max_retries": 3,
    "retry_pause_s": 1.0,
    "do_default_merge": true
  },
  "timeouts": {
    "api": 60,
    "thumb": 60,
    "post": 60
  }
}
```

---

## Usage

Run the script:

```bash
python hydrus_dupe_finder.py
```

---

## Key Parameters

- **`similarity_threshold`** – Increase to reduce false positives  
  (e.g., `0.6` recommended, `0.8` for stricter matches).
- **`verbose`** – Set to `true` to log per-pair SSIM scores and detailed stage info.
- **`progress.enabled`** – Show progress bars for each stage.

---

## How It Works

1. Search Hydrus for files matching tags.
2. Fetch metadata (duration, width, height).
3. Filter by duration ± tolerance.
4. Filter by aspect ratio ± tolerance *(optional)*.
5. **aHash prefilter** – fast perceptual hash filter *(optional)*.
6. Thumbnail comparison – SSIM / MS-SSIM similarity scoring.
7. Post results – potential duplicates sent to Hydrus for review.

---

## Notes

- The script is tuned for **video files**, but can work on still images with minor adjustments.
- For best performance:
  - Keep your Hydrus API on a local network.
  - For stricter matching, raise `similarity_threshold` and/or disable MS-SSIM.

---

## License

[MIT](LICENSE)
