# hydrus_dupe_finder.py
# One-file Python port of your MATLAB pipeline with:
# - All parameters in hydrus_config.json (now includes verbose + progress settings)
# - Progress bars for duration/AR (when verbose or progress.enabled) and aHash/SSIM
# - No CSV/tag ranking

import json
import math
from io import BytesIO
from itertools import combinations
from time import sleep

import numpy as np
import requests
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim

# Progress: tqdm optional
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# Optional MS-SSIM
try:
    import torch
    from pytorch_msssim import ms_ssim as torch_ms_ssim
    _HAS_MSSSIM = True
except Exception:
    _HAS_MSSSIM = False


# ---------- Utilities ----------
def _load_config(path="hydrus_config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _headers(api_key):
    return {"Hydrus-Client-API-Access-Key": api_key}

def _want_progress(C):
    """Progress shown if user enabled it or if verbose is on."""
    return bool(C.get("progress", {}).get("enabled", False)) or bool(C.get("verbose", False))

def _use_tqdm(C):
    """Use tqdm bars only if requested & available."""
    ptype = C.get("progress", {}).get("type", "tqdm").lower()
    return ptype == "tqdm" and _HAS_TQDM


# ---------- Hydrus API minimal helpers ----------
def api_version(api_url, api_key, timeout):
    r = requests.get(f"{api_url}/api_version", timeout=timeout)
    r.raise_for_status()
    return r.json().get("version")

def search_hashes(api_url, api_key, tags, timeout, return_hashes=True):
    params = {
        "tags": json.dumps(tags),
        "return_hashes": "true" if return_hashes else "false",
        "return_file_ids": "false"
    }
    r = requests.get(f"{api_url}/get_files/search_files", params=params, headers=_headers(api_key), timeout=timeout)
    r.raise_for_status()
    return r.json().get("hashes", [])

def get_metadata_batched(api_url, api_key, hashes, only_basic, timeout, batch_size=100):
    endpoint = f"{api_url}/get_files/file_metadata"
    meta = []
    for i in range(0, len(hashes), batch_size):
        batch = hashes[i:i+batch_size]
        params = {
            "only_return_basic_information": "true" if only_basic else "false",
            "hashes": json.dumps(batch)
        }
        r = requests.get(endpoint, params=params, headers=_headers(api_key), timeout=timeout)
        r.raise_for_status()
        meta.extend(r.json().get("metadata", []))
    return meta

def post_potential_duplicates(api_url, api_key, file_hashes, pairs, do_default_merge, timeout, batch_size=200, max_retries=3, retry_pause_s=1.0):
    endpoint = f"{api_url}/manage_file_relationships/set_file_relationships"
    attempted = 0
    posted = 0
    skipped = 0
    failed = 0
    fail_msgs = []

    uniq = sorted({tuple(sorted(p)) for p in pairs})

    for i in range(0, len(uniq), batch_size):
        chunk = uniq[i:i+batch_size]
        relationships = [
            {
                "hash_a": file_hashes[a],
                "hash_b": file_hashes[b],
                "relationship": 0,
                "do_default_content_merge": bool(do_default_merge),
            }
            for (a, b) in chunk
        ]
        body = {"relationships": relationships}
        attempts = 0
        while attempts < max_retries:
            attempts += 1
            try:
                r = requests.post(endpoint, headers=_headers(api_key), json=body, timeout=timeout)
                if r.status_code == 400:
                    skipped += len(chunk)
                    break
                r.raise_for_status()
                posted += len(chunk)
                break
            except Exception as e:
                if attempts < max_retries:
                    sleep(retry_pause_s)
                else:
                    failed += len(chunk)
                    fail_msgs.append(str(e))
        attempted += len(chunk)

    return {"attempted": attempted, "posted": posted, "skipped": skipped, "failed": failed, "fail_msgs": fail_msgs}


# ---------- Candidate generation ----------
def find_pairs_by_duration(durations_ms, tol_seconds, C):
    dsec = [(x or 0)/1000.0 for x in durations_ms]
    n = len(dsec)
    total = math.comb(n, 2) if n >= 2 else 0
    show_prog = _want_progress(C) and _use_tqdm(C)

    pairs = []
    if show_prog and total > 0:
        bar = tqdm(total=total, desc="Duration filter")
    else:
        bar = None

    for i in range(n - 1):
        di = dsec[i]
        for j in range(i + 1, n):
            if abs(di - dsec[j]) <= tol_seconds:
                pairs.append((i, j))
            if bar:
                bar.update(1)
    if bar:
        bar.close()
    return pairs

def find_pairs_by_ar(ars, ar_tol, candidate_pairs, C):
    show_prog = _want_progress(C) and _use_tqdm(C)
    iterable = candidate_pairs
    if show_prog:
        iterable = tqdm(candidate_pairs, desc="AR filter")

    out = []
    for i, j in iterable:
        a, b = ars[i], ars[j]
        if a == 0 or b == 0:
            continue
        if abs(a - b) <= ar_tol * max(a, b):
            out.append((i, j))
    return out


# ---------- aHash prefilter ----------
def compute_ahash_uint64(pil_img, pre_resize):
    if pre_resize:
        pil_img = pil_img.resize(tuple(pre_resize), Image.BILINEAR)
    if pil_img.mode != "L":
        pil_img = pil_img.convert("L")
    small = pil_img.resize((8, 8), Image.NEAREST)
    arr = np.asarray(small, dtype=np.uint8)
    m = float(arr.mean())
    bits = (arr > m).astype(np.uint8).flatten()
    h = 0
    for idx, bit in enumerate(bits, start=1):  # MSB-first pack
        if bit:
            shift = 64 - idx
            h |= (1 << shift)
    return h

def hamming64(a, b):
    return bin((a ^ b)).count("1")

def prefilter_pairs_with_ahash(api_url, api_key, file_hashes, pairs, resize, hamming_max, timeout, C):
    endpoint = f"{api_url}/get_files/thumbnail"

    needed = sorted(set(i for p in pairs for i in p))
    show_prog = _want_progress(C) and _use_tqdm(C)

    a_hashes = {}
    fetch_iter = needed if not show_prog else tqdm(needed, desc="aHash: fetch+hash")
    for idx in fetch_iter:
        h = file_hashes[idx]
        try:
            r = requests.get(endpoint, params={"hash": h}, headers=_headers(api_key), timeout=timeout)
            r.raise_for_status()
            pil = Image.open(BytesIO(r.content)).convert("RGB")
            a_hashes[idx] = compute_ahash_uint64(pil, resize)
        except Exception:
            a_hashes[idx] = None

    keep = []
    filt_iter = pairs if not show_prog else tqdm(pairs, desc="aHash: filter pairs")
    for i, j in filt_iter:
        hi = a_hashes.get(i)
        hj = a_hashes.get(j)
        if hi is None or hj is None:
            continue
        if hamming64(hi, hj) <= hamming_max:
            keep.append((i, j))
    return keep


# ---------- Image preprocessing & SSIM ----------
def trim_borders(im_rgb, strength, min_keep_frac):
    if im_rgb.ndim == 3 and im_rgb.shape[2] == 3:
        g = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)
    else:
        g = im_rgb
    g = g.astype(np.float32) / 255.0

    row_std = g.std(axis=1)
    col_std = g.std(axis=0)

    def smooth(v, w=5):
        if w <= 1:
            return v
        pad = w // 2
        vv = np.pad(v, (pad, pad), mode="edge")
        kernel = np.ones(w, dtype=np.float32) / w
        return np.convolve(vv, kernel, mode="valid")

    row_std = smooth(row_std, 5)
    col_std = smooth(col_std, 5)

    thr_r = float(strength * (row_std.max() if row_std.size else 0.0))
    thr_c = float(strength * (col_std.max() if col_std.size else 0.0))

    H, W = g.shape[:2]

    def first_above(a, thr, default):
        idx = np.argmax(a > thr)
        return int(idx) if a.size and (a > thr).any() else default

    def last_above(a, thr, default):
        if a.size and (a > thr).any():
            return int(len(a) - 1 - np.argmax((a > thr)[::-1]))
        return default

    top = first_above(row_std, thr_r, 0)
    bottom = last_above(row_std, thr_r, H - 1)
    left = first_above(col_std, thr_c, 0)
    right = last_above(col_std, thr_c, W - 1)

    if (bottom - top + 1) < int(min_keep_frac * H):
        top, bottom = 0, H - 1
    if (right - left + 1) < int(min_keep_frac * W):
        left, right = 0, W - 1

    return im_rgb[top:bottom + 1, left:right + 1, :]

def central_crop(im_rgb, frac):
    H, W = im_rgb.shape[:2]
    h = max(1, int(round(H * frac)))
    w = max(1, int(round(W * frac)))
    y0 = (H - h) // 2
    x0 = (W - w) // 2
    return im_rgb[y0:y0 + h, x0:x0 + w, :]

def preprocess_thumb(im_rgb, P):
    im = trim_borders(im_rgb, float(P["trimStrength"]), float(P["minKeepFrac"]))
    if P["fixedResize"]:
        w, h = int(P["fixedResize"][0]), int(P["fixedResize"][1])
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    if P["toGray"]:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    sigma = float(P["preBlurSigma"])
    if sigma > 0:
        im = cv2.GaussianBlur(im, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if P["histEq"]:
        if im.ndim == 2:
            im = cv2.equalizeHist(im.astype(np.uint8))
        else:
            im = np.stack([cv2.equalizeHist(im[:, :, c].astype(np.uint8)) for c in range(3)], axis=2)
    return im

def _fetch_thumb_rgb(api_url, api_key, file_hash, timeout):
    r = requests.get(f"{api_url}/get_files/thumbnail", params={"hash": file_hash}, headers=_headers(api_key), timeout=timeout)
    r.raise_for_status()
    pil = Image.open(BytesIO(r.content)).convert("RGB")
    return np.asarray(pil, dtype=np.uint8)

def _score_pair(im1_proc, im2_proc, P):
    center_frac = float(P["centerFrac"])
    if center_frac < 1.0:
        if im1_proc.ndim == 2:
            im1_proc = central_crop(np.stack([im1_proc]*3, axis=-1), center_frac)[:, :, 0]
            im2_proc = central_crop(np.stack([im2_proc]*3, axis=-1), center_frac)[:, :, 0]
        else:
            im1_proc = central_crop(im1_proc, center_frac)
            im2_proc = central_crop(im2_proc, center_frac)

    if im1_proc.ndim == 3 and im1_proc.shape[2] == 3:
        im1_g = cv2.cvtColor(im1_proc, cv2.COLOR_RGB2GRAY)
        im2_g = cv2.cvtColor(im2_proc, cv2.COLOR_RGB2GRAY)
    else:
        im1_g, im2_g = im1_proc, im2_proc

    if bool(P["useMSSSIM"]) and _HAS_MSSSIM:
        with torch.no_grad():
            t1 = torch.from_numpy(im1_g.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            t2 = torch.from_numpy(im2_g.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            return float(torch_ms_ssim(t1, t2, data_range=1.0).item())
    else:
        return float(ssim(im1_g, im2_g, data_range=255))

def compare_thumbnails(api_url, api_key, file_hashes, pairs, threshold, P, timeout, C):
    show_prog = _want_progress(C) and _use_tqdm(C)
    iterable = pairs if not show_prog else tqdm(pairs, desc="SSIM compare")

    kept = []
    for (i, j) in iterable:
        try:
            im1 = _fetch_thumb_rgb(api_url, api_key, file_hashes[i], timeout)
            im2 = _fetch_thumb_rgb(api_url, api_key, file_hashes[j], timeout)
            im1p = preprocess_thumb(im1, P)
            im2p = preprocess_thumb(im2, P)
            score = _score_pair(im1p, im2p, P)
            if score >= threshold:
                kept.append((i, j))
        except Exception:
            # skip on errors
            pass
    return kept


# ---------- Main ----------
def main():
    C = _load_config("hydrus_config.json")

    api_url = f"{C['api_url']}:{C['api_port']}"
    api_key = C["api_key"]

    VERBOSE = bool(C.get("verbose", False))
    if C.get("progress", {}).get("type", "tqdm").lower() == "tqdm" and not _HAS_TQDM and C.get("progress", {}).get("enabled", False):
        print("Note: tqdm not installed; progress bars disabled. Install with: pip install tqdm")

    # Step 0: Sanity / version
    ver = api_version(api_url, api_key, C["timeouts"]["api"])
    if VERBOSE:
        print(f"Hydrus API version: {ver}")

    # Step 1: Fetch hashes
    tags = C["tags"]
    hashes = search_hashes(api_url, api_key, tags, C["timeouts"]["api"])
    n = len(hashes)
    print(f"Got {n} hashes")
    if n == 0:
        return

    # Step 2: Metadata
    meta = get_metadata_batched(api_url, api_key, hashes, only_basic=True, timeout=C["timeouts"]["api"])
    if VERBOSE:
        print(f"Fetched metadata for {len(meta)} files")

    durations = [m.get("duration", 0) or 0 for m in meta]
    widths    = [m.get("width", 0) or 0 for m in meta]
    heights   = [m.get("height", 0) or 1 for m in meta]
    ars       = [(w / h if h else 0.0) for (w, h) in zip(widths, heights)]

    # Step 3: Candidate gen
    if VERBOSE:
        print("Generating duration-based candidate pairs...")
    dur_pairs = find_pairs_by_duration(durations, C["candidate_generation"]["tolSeconds"], C)
    print(f"Duration candidates: {len(dur_pairs)}")

    if C["filters"]["use_ar_filter"]:
        if VERBOSE:
            print("Applying AR filter...")
        ar_pairs = find_pairs_by_ar(ars, C["candidate_generation"]["arTol"], dur_pairs, C)
        print(f"AR-filtered candidates: {len(ar_pairs)}")
    else:
        ar_pairs = dur_pairs
        print(f"AR filter disabled: {len(ar_pairs)} pairs")

    if not ar_pairs:
        print("No candidates after filtering. Exiting.")
        return

    # Step 4: aHash prefilter (optional)
    if C["filters"]["use_ahash_prefilter"]:
        if VERBOSE:
            print("Running aHash prefilter...")
        pf = prefilter_pairs_with_ahash(
            api_url, api_key, hashes, ar_pairs,
            resize=C["ahash"]["resize"],
            hamming_max=C["ahash"]["hammingMax"],
            timeout=C["timeouts"]["thumb"],
            C=C
        )
        print(f"aHash kept: {len(pf)} / {len(ar_pairs)}")
        if not pf:
            print("No pairs survived the aHash prefilter. Exiting.")
            return
        ar_pairs = pf
    else:
        print(f"aHash prefilter disabled: passing {len(ar_pairs)} pairs to SSIM.")

    # Step 5: Thumbnail compare
    if VERBOSE:
        print("Comparing thumbnails with SSIM/MS-SSIM...")
    P = C["preprocessing"]
    dup_pairs = compare_thumbnails(
        api_url, api_key, hashes, ar_pairs,
        threshold=C["similarity_threshold"],
        P=P,
        timeout=C["timeouts"]["thumb"],
        C=C
    )
    print(f"Thumbnail compare found {len(dup_pairs)} duplicate pairs")

    if not dup_pairs:
        print(f"No duplicates >= {C['similarity_threshold']:.3f}. Nothing to post.")
        return

    # Step 6: Post to Hydrus
    if VERBOSE:
        print("Posting potential duplicates to Hydrus...")
    rep = post_potential_duplicates(
        api_url, api_key, hashes, dup_pairs,
        do_default_merge=C["post"]["do_default_merge"],
        timeout=C["timeouts"]["post"],
        batch_size=C["post"]["batch_size"],
        max_retries=C["post"]["max_retries"],
        retry_pause_s=C["post"]["retry_pause_s"]
    )
    print(f"Relationships -> Posted: {rep.get('posted', 0)} | Skipped: {rep.get('skipped', 0)} | Failed: {rep.get('failed', 0)} | Attempted: {rep.get('attempted', 0)}")


if __name__ == "__main__":
    main()
