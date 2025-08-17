# Hydrus Duplicate Finder (Python)

A Python duplicate-finder pipeline, designed to work with the **Hydrus Client API** to identify potential duplicate video files.  
It combines lightweight prefilters (duration, aspect ratio, perceptual hash) with structural image similarity (SSIM).

---

## Features

- **Tag-based search** – fetch files directly from Hydrus.
- **aHash prefilter** – fast perceptual hashing with Hamming distance filtering.
- **SSIM** – configurable similarity threshold (default **0.8**).
- **Border trimming & preprocessing** – reduce false positives from black bars or padding.
- **Parallel SSIM processing** – utilize all CPU cores automatically.
- **Progress bars** – via `tqdm` (optional).
- **Verbose logging** – see phase summaries and debug details.
- **Hydrus integration** – posts potential duplicates directly back into Hydrus.

---

## Requirements

- **Python 3.8+** (tested up to 3.13)
- A running Hydrus client with the **API enabled** and a key with  
  **“Edit File Relationships”** permission (perm id 8)

Dependencies are listed in [requirements.txt](requirements.txt):

```txt
numpy>=1.24
requests>=2.31
Pillow>=9.5
opencv-python>=4.8
scikit-image>=0.21
tqdm>=4.66
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/ytypo123/hydrus-duplicate-finder-python.git
cd hydrus-duplicate-finder-python
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### 3. Activate the environment
```bash
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure `hydrus_config.json`

Create a file called `hydrus_config.json` in the project root. Example:

```json
{
  "api_url": "http://127.0.0.1",
  "api_port": 42001,
  "api_key": "your-hydrus-api-key-here",

  "tags": [
    ["system:filetype=video"]
  ],

  "verbosity": 1,
  "progress": { "enabled": true, "type": "tqdm" },

  "network": { "max_connections": 4, "backoff_factor": 0.5 },
  "timeouts": { "api": 30, "thumb": 60, "post": 30 },

  "ahash": {
    "resize": [128, 128],
    "hammingMax": 16,
    "prefixBits": 0,
    "io_workers": 16
  },

  "preprocessing": {
    "trimStrength": 0.05,
    "minKeepFrac": 0.90,
    "fixedResize": [256, 256],
    "toGray": true,
    "preBlurSigma": 0.5,
    "histEq": false,
    "centerFrac": 1.0,
    "useMSSSIM": false
  },

  "similarity_threshold": 0.8,

  "ssim_parallel": {
    "enabled": true,
    "max_workers": 0,
    "chunksize": 1
  },

  "post": {
    "do_default_merge": true,
    "batch_size": 200,
    "max_retries": 3,
    "retry_pause_s": 1.0
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

- **`similarity_threshold`** – Increase for stricter matches  
  (`0.6` = lenient, `0.8` = recommended).
- **`preprocessing`** – Controls border trimming, resize, grayscale, blur, etc.
- **`ssim_parallel.enabled`** – Set `false` to disable multiprocessing.
- **`verbosity`** –  
  - `0` minimal (just results)  
  - `1` normal (phase summaries)  
  - `2` debug (detailed retries and failures)

---

## How It Works

1. Search Hydrus for files matching your tags.
2. Fetch all thumbnails and compute perceptual hashes (aHash).
3. Group and filter pairs by Hamming distance.
4. Compare remaining candidates using SSIM.
5. Check Hydrus for existing relationships.
6. Post new **potential duplicate** relationships back to Hydrus.

---

## Notes

- SSIM scores are sensitive to borders and letterboxing —  
  use `preprocessing.trimStrength` and `minKeepFrac` to reduce false positives.
- Hydrus decides which file to keep; this script only suggests *potential* duplicates.

---

## License

GPL-3
