#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example hydrus_config.json
This file controls how the duplicate detection script connects to Hydrus,
selects files, processes them, logs results, and posts relationships.

{
  // ---------------- Hydrus connection ----------------
  "api_url": "http://127.0.0.1",        // Base URL of the Hydrus Client API
  "api_port": 42001,                    // API port
  "api_key": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  // Your API access key

  // ---------------- Search tags ----------------
  "tags": [
    ["system:filetype=video"],           // Each inner array = AND group of tags
    ["system:filesize < 8MB"],           // Groups are ANDed together
    ["system:filesize > 7MB"]            // Example: find videos between 7MB and 8MB
  ],

  // ---------------- Logging & progress output ----------------
  // verbosity:
  //    0 = minimal (only final counts and key failures)
  //    1 = normal (phase summaries, counts)
  //    2 = extra (debug-level: retries, per-batch info, failures)
  "verbosity": 1,
  "verbose": true,                       // Legacy flag; ignored if "verbosity" is present
  "progress": { 
    "enabled": true,                     // Show progress bars?
    "type": "tqdm"                       // "tqdm" = pretty terminal bars; anything else disables
  },

  // ---------------- Network & retry settings ----------------
  "network": {
    "max_connections": 4,                // Max concurrent HTTP connections
    "backoff_factor": 0.5                // Retry delay growth factor
  },

  // ---------------- Timeout values (in seconds) ----------------
  "timeouts": { 
    "api": 30,                           // For Hydrus API calls (metadata/search)
    "thumb": 60,                         // For thumbnail downloads
    "post": 30                           // For posting duplicate relationships
  },

  // ---------------- aHash (average hash) settings ----------------
  "ahash": {
    "resize": [128, 128],                // Resize before hashing (speed/consistency)
    "hammingMax": 16,                    // Max Hamming distance allowed between hashes
    "prefixBits": 0,                     // Prefix length for bucketing (0 = all in one bucket)
    "io_workers": 16                     // Threads for downloading + hashing
  },

  // ---------------- Image preprocessing for SSIM ----------------
  "preprocessing": {
    "trimStrength": 0.05,                // Border-trim sensitivity
    "minKeepFrac": 0.90,                 // Keep at least this fraction after trimming
    "fixedResize": [256, 256],           // Resize all images before SSIM
    "toGray": true,                      // Convert to grayscale before SSIM
    "preBlurSigma": 0.5,                 // Gaussian blur (0 disables)
    "histEq": false,                     // Histogram equalization (contrast normalization)
    "centerFrac": 1.0,                   // Compare only central fraction (1.0 = full image)
    "useMSSSIM": false                   // (Reserved) Multi-scale SSIM
  },

  // ---------------- Similarity detection threshold ----------------
  "similarity_threshold": 0.8,           // SSIM score to consider pair a "duplicate" (0–1)

  // ---------------- Parallel SSIM processing ----------------
  "ssim_parallel": {
    "enabled": true,                     // Use multiple CPU processes for SSIM
    "max_workers": 0,                    // 0 = auto-detect based on CPU count
    "chunksize": 1                       // Workload size per process
  },

  // ---------------- Posting duplicate relationships ----------------
  "post": {
    "do_default_merge": true,            // Merge metadata (tags/ratings/etc.) by default
    "batch_size": 200,                   // Number of pairs per POST request
    "max_retries": 3,                    // Max retries per batch
    "retry_pause_s": 1.0                 // Pause between retries
  }
}
"""

import json, os, sys
from io import BytesIO
from time import sleep
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim

# tqdm is optional; we’ll detect if it’s available
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


# ------------------------------ Logging / Verbosity ------------------------------
class Logger:
    """
    Consistent console output + optional progress bars.

    Levels:
      0 = minimal (only key milestones & final results)
      1 = normal (phase summaries and counts)
      2 = extra (debug: retries, batch info, per-item failures)
    """
    def __init__(self, level: int, progress_enabled: bool, use_tqdm: bool):
        self.level = int(level)
        self.progress_enabled = bool(progress_enabled)
        self.use_tqdm = bool(use_tqdm and _HAS_TQDM)

    def log(self, level: int, msg: str):
        if self.level >= level:
            print(msg, flush=True)

    def phase(self, tag: str, msg: str, level: int = 1):
        self.log(level, f"[{tag}] {msg}")

    def tqdm_iter(self, iterable, total=None, desc="Progress"):
        if self.progress_enabled and self.use_tqdm:
            return tqdm(iterable, total=total, desc=desc)
        return iterable


# ------------------------------ Config / Utils ------------------------------
def _load_config(path="hydrus_config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _headers(api_key):
    return {"Hydrus-Client-API-Access-Key": api_key}

def _make_session(C):
    """
    Create a pooled HTTP session with retries for transient failures.
    """
    max_conns = int(C.get("network", {}).get("max_connections", 8))
    backoff = float(C.get("network", {}).get("backoff_factor", 0.5))
    status_forcelist = (500, 502, 503, 504)
    retry = Retry(
        total=5, connect=5, read=5, status=5,
        backoff_factor=backoff,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "POST"])
    )
    sess = requests.Session()
    adapter = HTTPAdapter(pool_connections=max_conns, pool_maxsize=max_conns, max_retries=retry)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


# ------------------------------ Hydrus API (basic) ------------------------------
def api_version(session, api_url, timeout):
    r = session.get(f"{api_url}/api_version", timeout=timeout)
    r.raise_for_status()
    return r.json().get("version")

def verify_access(session, api_url, api_key, timeout):
    """
    Returns Hydrus access info (includes key name, permissions).
    Raises on 419 (invalid/expired key) or HTTP errors.
    """
    r = session.get(f"{api_url}/verify_access_key", headers=_headers(api_key), timeout=timeout)
    if r.status_code == 419:
        raise RuntimeError("Access key invalid or expired (HTTP 419).")
    r.raise_for_status()
    return r.json()

def has_relationship_permission(access_info):
    """
    Check if the key has 'Edit File Relationships' (permission id 8).
    """
    if bool(access_info.get("permits_everything")):
        return True
    perms = access_info.get("basic_permissions") or []
    return 8 in set(perms) or "8" in {str(p) for p in perms}

def search_hashes(session, api_url, api_key, tags, timeout):
    """
    Query Hydrus by tags; returns list of file SHA256 hex hashes.
    """
    params = {"tags": json.dumps(tags), "return_hashes": "true", "return_file_ids": "false"}
    r = session.get(f"{api_url}/get_files/search_files", params=params, headers=_headers(api_key), timeout=timeout)
    r.raise_for_status()
    return r.json().get("hashes", [])


# ------------------------------ Hydrus API (relationships) ------------------------------
def get_relationships_for_hashes(session, api_url, api_key, hashes, timeout, chunk_size=500, log: Logger=None):
    """
    Fetch existing relationships for a set of hashes.
    Strategy:
      1) Try bulk POST (fast, new API).
      2) On 404/405, fall back to per-hash GET (legacy API).
    Returns: dict[hash] -> {
        "is_king": bool, "king": str, ...,
        "0": [hash, ...],  # potentials
        "1": [...],        # false positives (not duplicates)
        "3": [...],        # alternates
        "8": [...]         # duplicates
    }
    """
    endpoint = f"{api_url}/manage_file_relationships/get_file_relationships"
    all_map = {}

    if not hashes:
        return all_map

    # --- Attempt bulk POST first ---
    try:
        for start in range(0, len(hashes), chunk_size):
            chunk = hashes[start:start+chunk_size]
            body = {"files": {"hashes": chunk}}
            r = session.post(endpoint, headers=_headers(api_key), json=body, timeout=timeout)
            if r.status_code in (404, 405):
                raise FileNotFoundError("bulk-post-not-supported")
            r.raise_for_status()
            part = r.json().get("file_relationships", {})
            all_map.update(part)
        if log: log.phase("Hydrus", "Relationship lookup: bulk mode", level=2)
        return all_map
    except FileNotFoundError:
        # Fall through to GET-per-hash
        pass
    except requests.HTTPError as e:
        # If server explicitly rejects body shape with 400, try legacy GET.
        if e.response is None or e.response.status_code not in (400, 404, 405):
            raise

    # --- Legacy per-hash GET fallback ---
    if log: log.phase("Hydrus", "Relationship lookup: legacy per-hash mode", level=1)

    # Small helper to fetch one hash
    def _fetch_one(h):
        try:
            r = session.get(endpoint, headers=_headers(api_key),
                            params={"hash": h}, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            # Some builds return { "file_relationships": { <hash>: {...} } }
            # Others may return the object directly. Normalize here.
            fr = data.get("file_relationships")
            if isinstance(fr, dict) and h in fr:
                return h, fr[h]
            # If the server returns the relationships directly, accept that.
            if isinstance(data, dict) and any(k in data for k in ("0","1","3","8","king","is_king")):
                return h, data
        except Exception:
            return h, {}
        return h, {}

    # Iterate with optional progress
    it = hashes
    it = log.tqdm_iter(it, total=len(hashes), desc="Checking existing relationships") if log else it
    for h in it:
        key, rel = _fetch_one(h)
        if rel:
            all_map[key] = rel

    return all_map


def classify_pairs_against_hydrus(session, api_url, api_key, file_hashes, pairs, timeout, log: Logger=None):
    """
    For each (i,j) pair, check Hydrus's current relationship and bucket it.
    """
    if not pairs:
        return {
            "already_duplicates": [],
            "already_false_positives": [],
            "already_potentials": [],
            "already_alternates": [],
            "new_unknown": []
        }

    involved_idxs = sorted({k for ij in pairs for k in ij})
    involved_hashes = [file_hashes[k] for k in involved_idxs]
    rel_map = get_relationships_for_hashes(session, api_url, api_key, involved_hashes, timeout, log=log)

    rel = {h: rel_map.get(h, {}) for h in involved_hashes}
    buckets = {
        "already_duplicates": [],
        "already_false_positives": [],
        "already_potentials": [],
        "already_alternates": [],
        "new_unknown": []
    }

    def has_rel(h1, h2, key):
        a = set(rel.get(h1, {}).get(key, []))
        b = set(rel.get(h2, {}).get(key, []))
        return (h2 in a) or (h1 in b)

    for (i, j) in pairs:
        h1, h2 = file_hashes[i], file_hashes[j]
        if has_rel(h1, h2, "8"):
            buckets["already_duplicates"].append((i, j))
        elif has_rel(h1, h2, "1"):
            buckets["already_false_positives"].append((i, j))
        elif has_rel(h1, h2, "3"):
            buckets["already_alternates"].append((i, j))
        elif has_rel(h1, h2, "0"):
            buckets["already_potentials"].append((i, j))
        else:
            buckets["new_unknown"].append((i, j))

    return buckets



# ------------------------------ Posting relationships ------------------------------
def post_potential_duplicates(session, api_url, api_key, file_hashes, pairs, do_default_merge,
                              timeout, batch_size=200, max_retries=3, retry_pause_s=1.0, log: Logger=None):
    """
    Post candidate duplicate relationships (relationship=0) in batches.
    Returns counts and a sample of failure messages.
    """
    endpoint = f"{api_url}/manage_file_relationships/set_file_relationships"
    uniq = sorted({tuple(sorted(p)) for p in pairs})
    attempted = posted = skipped = failed = 0
    fail_msgs = []

    for i in range(0, len(uniq), batch_size):
        chunk = uniq[i:i+batch_size]
        body = {"relationships": [
            {"hash_a": file_hashes[a], "hash_b": file_hashes[b],
             "relationship": 0, "do_default_content_merge": bool(do_default_merge)}
            for (a, b) in chunk
        ]}

        attempts = 0
        while attempts < max_retries:
            attempts += 1
            try:
                r = session.post(endpoint, headers=_headers(api_key), json=body, timeout=timeout)
                if r.status_code == 200:
                    posted += len(chunk)
                    if log: log.phase("Post", f"Batch {i//batch_size+1}: posted {len(chunk)}", level=2)
                    break
                else:
                    msg = (r.text or "").strip()
                    fail_msgs.append(f"HTTP {r.status_code}: {msg[:300]}")
                    if log: log.phase("Post", f"Non-200: HTTP {r.status_code} :: {msg[:300]}", level=2)
                    r.raise_for_status()
            except Exception as e:
                if attempts < max_retries:
                    if log: log.phase("Post", f"Error (attempt {attempts}/{max_retries}): {e!r}; retry in {retry_pause_s}s", level=2)
                    sleep(retry_pause_s)
                else:
                    failed += len(chunk)
                    fail_msgs.append(repr(e))
        attempted += len(chunk)

    if fail_msgs and log:
        log.phase("Post", "Failures (sample up to 5):", level=2)
        for m in fail_msgs[:5]:
            log.phase("Post", f"  - {m}", level=2)

    return {"attempted": attempted, "posted": posted, "skipped": skipped, "failed": failed, "fail_msgs": fail_msgs}


# ------------------------------ aHash (thumbnail hashing) ------------------------------
def _fetch_thumb_rgb(session, api_url, api_key, file_hash, timeout):
    """
    Download the Hydrus thumbnail for a given file hash; return RGB uint8 array.
    """
    r = session.get(f"{api_url}/get_files/thumbnail", params={"hash": file_hash}, headers=_headers(api_key), timeout=timeout)
    r.raise_for_status()
    pil = Image.open(BytesIO(r.content)).convert("RGB")
    return np.asarray(pil, dtype=np.uint8)

def compute_ahash_uint64(pil_img, pre_resize):
    """
    Compute 64-bit average hash (aHash) for a PIL image (optional pre-resize).
    """
    if pre_resize:
        pil_img = pil_img.resize(tuple(pre_resize), Image.BILINEAR)
    if pil_img.mode != "L":
        pil_img = pil_img.convert("L")
    small = pil_img.resize((8, 8), Image.NEAREST)
    arr = np.asarray(small, dtype=np.uint8)
    m = float(arr.mean())
    bits = (arr > m).astype(np.uint8).flatten()
    h = 0
    for idx, bit in enumerate(bits, start=1):
        if bit:
            h |= (1 << (64 - idx))
    return h

def hamming64(a, b):
    return bin(a ^ b).count("1")

def fetch_all_ahashes(session, api_url, api_key, file_hashes, resize, timeout, workers, log: Logger):
    """
    Fetch thumbnails and compute aHash in parallel (threads).
    """
    ah = [None] * len(file_hashes)

    def _job(idx_hash):
        idx, h = idx_hash
        try:
            img = _fetch_thumb_rgb(session, api_url, api_key, h, timeout)
            ah[idx] = compute_ahash_uint64(Image.fromarray(img), resize)
        except Exception as e:
            ah[idx] = None
            if log: log.phase("Hash", f"Failed to hash index {idx}: {e}", level=2)

    it = enumerate(file_hashes)
    it = log.tqdm_iter(list(it), desc="Hashing thumbnails")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        ex.map(_job, it)
    return ah  # list of uint64 or None


def bucket_by_prefix(ah_list, prefix_bits):
    """
    Group items by the top N prefix bits of their 64-bit hash.
    Return dict: prefix -> list(indices).
    """
    mask = (0xFFFFFFFFFFFFFFFF >> (64 - prefix_bits)) << (64 - prefix_bits) if prefix_bits > 0 else 0
    buckets = defaultdict(list)
    for i, h in enumerate(ah_list):
        if h is None:
            continue
        key = (h & mask) if prefix_bits > 0 else 0
        buckets[key].append(i)
    return buckets

def candidate_pairs_from_buckets(buckets, log: Logger=None):
    """
    Yield pairs (i,j) within each bucket.
    Shows a pair-generation progress estimate when progress is enabled.
    """
    pbar = None
    if log and log.progress_enabled and log.use_tqdm:
        total = 0
        for idxs in buckets.values():
            n = len(idxs)
            total += n*(n-1)//2
        pbar = tqdm(total=total, desc="Generating candidate pairs")

    for idxs in buckets.values():
        n = len(idxs)
        if n < 2:
            continue
        for i in range(n-1):
            for j in range(i+1, n):
                if pbar: pbar.update(1)
                yield (idxs[i], idxs[j])

    if pbar: pbar.close()

def filter_pairs_by_hamming(ah_list, pairs, hamming_max):
    """
    Keep only pairs with Hamming distance <= threshold.
    """
    keep = []
    for i, j in pairs:
        hi, hj = ah_list[i], ah_list[j]
        if hi is None or hj is None:
            continue
        if hamming64(hi, hj) <= hamming_max:
            keep.append((i, j))
    return keep


# ------------------------------ Preprocess & SSIM ------------------------------
def trim_borders(im_rgb, strength, min_keep_frac):
    """
    Trim flat borders via running std; guarantee a minimum kept fraction.
    """
    g = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY) if im_rgb.ndim == 3 else im_rgb
    g = g.astype(np.float32) / 255.0
    row_std = np.convolve(np.pad(g.std(axis=1), 2, mode="edge"), np.ones(5)/5, mode="valid")
    col_std = np.convolve(np.pad(g.std(axis=0), 2, mode="edge"), np.ones(5)/5, mode="valid")
    thr_r = strength * row_std.max(); thr_c = strength * col_std.max()
    H, W = g.shape[:2]
    def first_above(a, thr, default): return int(np.argmax(a > thr)) if (a > thr).any() else default
    def last_above(a, thr, default):  return int(len(a) - 1 - np.argmax((a > thr)[::-1])) if (a > thr).any() else default
    top, bottom = first_above(row_std, thr_r, 0), last_above(row_std, thr_r, H-1)
    left, right = first_above(col_std, thr_c, 0), last_above(col_std, thr_c, W-1)
    if bottom - top + 1 < int(min_keep_frac * H): top, bottom = 0, H - 1
    if right - left + 1 < int(min_keep_frac * W): left, right = 0, W - 1
    return im_rgb[top:bottom+1, left:right+1, :]

def central_crop(im_rgb, frac):
    H, W = im_rgb.shape[:2]
    h, w = max(1, int(round(H*frac))), max(1, int(round(W*frac)))
    y0, x0 = (H-h)//2, (W-w)//2
    return im_rgb[y0:y0+h, x0:x0+w, :]

def preprocess_thumb(im_rgb, P):
    """
    Apply border trim, resize, grayscale, blur, and optional histogram equalization.
    """
    im = trim_borders(im_rgb, float(P["trimStrength"]), float(P["minKeepFrac"]))
    if P["fixedResize"]:
        im = cv2.resize(im, tuple(P["fixedResize"]), interpolation=cv2.INTER_AREA)
    if P["toGray"]:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    if float(P["preBlurSigma"]) > 0:
        im = cv2.GaussianBlur(im, (0, 0), sigmaX=float(P["preBlurSigma"]))
    if P["histEq"]:
        if im.ndim == 2:
            im = cv2.equalizeHist(im.astype(np.uint8))
        else:
            im = np.stack([cv2.equalizeHist(im[:, :, c].astype(np.uint8)) for c in range(3)], axis=2)
    return im

def _score_pair_ssim(im1_proc, im2_proc, P):
    """
    Compute SSIM score (0-1). Optionally restrict to image center.
    """
    if float(P.get("centerFrac", 1.0)) < 1.0:
        im1_proc = central_crop(im1_proc, float(P["centerFrac"]))
        im2_proc = central_crop(im2_proc, float(P["centerFrac"]))
    im1_g = cv2.cvtColor(im1_proc, cv2.COLOR_RGB2GRAY) if im1_proc.ndim == 3 else im1_proc
    im2_g = cv2.cvtColor(im2_proc, cv2.COLOR_RGB2GRAY) if im2_proc.ndim == 3 else im2_proc
    return float(ssim(im1_g, im2_g, data_range=255))

def _ssim_worker(args):
    (i, j, api_url, api_key, hash_i, hash_j, P, thr, timeout) = args
    try:
        sess = requests.Session()
        im1 = _fetch_thumb_rgb(sess, api_url, api_key, hash_i, timeout)
        im2 = _fetch_thumb_rgb(sess, api_url, api_key, hash_j, timeout)
        im1p = preprocess_thumb(im1, P); im2p = preprocess_thumb(im2, P)
        if _score_pair_ssim(im1p, im2p, P) >= thr:
            return (i, j)
    except Exception:
        pass
    return None

def ssim_parallel_compare(api_url, api_key, file_hashes, pairs, P, thr, timeout, par_cfg, log: Logger):
    """
    Compare candidate pairs with SSIM, in parallel (processes) or serial fallback.
    """
    if not pairs:
        return []

    # Serial fallback if disabled
    if not par_cfg.get("enabled", True):
        kept = []
        sess = requests.Session()
        it = log.tqdm_iter(pairs, total=len(pairs), desc="Comparing pairs (SSIM)")
        for (i, j) in it:
            try:
                im1 = _fetch_thumb_rgb(sess, api_url, api_key, file_hashes[i], timeout)
                im2 = _fetch_thumb_rgb(sess, api_url, api_key, file_hashes[j], timeout)
                if _score_pair_ssim(preprocess_thumb(im1, P), preprocess_thumb(im2, P), P) >= thr:
                    kept.append((i, j))
            except Exception as e:
                if log: log.phase("SSIM", f"Pair {(i,j)} failed: {e}", level=2)
        return kept

    # Parallel path
    max_workers = par_cfg.get("max_workers", 0) or os.cpu_count() or 1
    chunksize = int(par_cfg.get("chunksize", 1))
    tasks = [(i, j, api_url, api_key, file_hashes[i], file_hashes[j], P, thr, timeout) for (i, j) in pairs]
    kept = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        it = ex.map(_ssim_worker, tasks, chunksize=chunksize)
        it = log.tqdm_iter(it, total=len(tasks), desc="Comparing pairs (SSIM)")
        for res in it:
            if res is not None:
                kept.append(res)
    return kept


# ------------------------------ Main ------------------------------
def main():
    # Load config
    C = _load_config("hydrus_config.json")
    api_url = f"{C['api_url']}:{C['api_port']}"
    api_key = C["api_key"]

    # Verbosity / progress setup
    verbosity = C.get("verbosity", None)
    if verbosity is None:
        verbosity = 1 if C.get("verbose", False) else 0
    progress_enabled = bool(C.get("progress", {}).get("enabled", False)) or verbosity >= 1
    use_tqdm = (C.get("progress", {}).get("type", "tqdm").lower() == "tqdm")
    log = Logger(level=verbosity, progress_enabled=progress_enabled, use_tqdm=use_tqdm)

    # Networking
    session = _make_session(C)

    # 0) API + permissions preflight
    try:
        ver = api_version(session, api_url, C["timeouts"]["api"])
        log.phase("Hydrus", f"API version: {ver}", level=1)
    except Exception as e:
        log.phase("Hydrus", f"Failed to get API version: {e}", level=0)
        sys.exit(1)

    try:
        access = verify_access(session, api_url, api_key, C["timeouts"]["api"])
        if verbosity >= 1:
            log.phase("Auth", f"Access key name: {access.get('name')}", level=1)
            log.phase("Auth", f"Permissions: {access.get('human_description')}", level=1)
            log.phase("Auth", f"permits_everything: {access.get('permits_everything')} basic_permissions: {access.get('basic_permissions')}", level=2)
    except Exception as e:
        log.phase("Auth", f"Access check failed: {e}", level=0)
        sys.exit(1)

    if not has_relationship_permission(access):
        log.phase("Auth", "FATAL: Access key lacks 'Edit File Relationships' (perm id 8).", level=0)
        sys.exit(1)

    # 1) Pull files by tags
    try:
        hashes = search_hashes(session, api_url, api_key, C["tags"], C["timeouts"]["api"])
    except Exception as e:
        log.phase("Search", f"Failed to search hashes: {e}", level=0)
        sys.exit(1)

    n = len(hashes)
    log.phase("Search", f"Found {n} files matching tags.", level=0)
    if n == 0:
        log.phase("Result", "No files found. Nothing to do.", level=0)
        return

    # 2) aHash for every file (parallel)
    log.phase("Hash", "Computing thumbnail hashes for all files...", level=1)
    A = fetch_all_ahashes(
        session, api_url, api_key, hashes,
        resize=C["ahash"]["resize"],
        timeout=C["timeouts"]["thumb"],
        workers=int(C["ahash"]["io_workers"]),
        log=log
    )

    # 3) Bucket and Hamming filter to get candidate pairs
    buckets = bucket_by_prefix(A, int(C["ahash"]["prefixBits"]))
    cand_iter = candidate_pairs_from_buckets(buckets, log=log)
    ham_pairs = filter_pairs_by_hamming(A, cand_iter, int(C["ahash"]["hammingMax"]))
    log.phase("Pairs", f"Candidate pairs after Hamming filter: {len(ham_pairs)}", level=0)

    if not ham_pairs:
        log.phase("Result", "No aHash candidates; exiting.", level=0)
        return

    # 4) SSIM scoring (parallel or serial)
    log.phase("SSIM", "Scoring candidate pairs using SSIM…", level=1)
    dup_pairs = ssim_parallel_compare(
        api_url, api_key, hashes, ham_pairs,
        P=C["preprocessing"],
        thr=float(C["similarity_threshold"]),
        timeout=C["timeouts"]["thumb"],
        par_cfg=C["ssim_parallel"],
        log=log
    )
    log.phase("SSIM", f"SSIM indicated pairs (SSIM ≥ {C['similarity_threshold']}): {len(dup_pairs)}", level=0)

    if not dup_pairs:
        log.phase("Result", "Nothing to post (no SSIM-confirmed duplicates).", level=0)
        return

    # 5) Ask Hydrus what it already knows about these pairs
    try:
        buckets_known = classify_pairs_against_hydrus(
    session, api_url, api_key, hashes, dup_pairs,
    timeout=C["timeouts"]["api"], log=log
)

    except Exception as e:
        log.phase("Hydrus", f"Relationship lookup failed: {e}", level=0)
        # If this fails, we can still proceed to posting, but it's safer to stop:
        return

    # Clear, single-line status for easy scanning
    log.phase("Hydrus", (
        "Existing status -> "
        f"duplicates:{len(buckets_known['already_duplicates'])} | "
        f"not-duplicates:{len(buckets_known['already_false_positives'])} | "
        f"alternates:{len(buckets_known['already_alternates'])} | "
        f"potentials:{len(buckets_known['already_potentials'])} | "
        f"unknown:{len(buckets_known['new_unknown'])}"
    ), level=0)

    # Decide what to post: usually only truly new/unknown pairs
    pairs_to_post = buckets_known["new_unknown"]

    if not pairs_to_post:
        # Friendly phrasing that matches your ask
        if len(dup_pairs) > 0 and len(buckets_known["already_false_positives"]) == len(dup_pairs):
            log.phase("Result", f"{len(dup_pairs)} strong matches found, but Hydrus marks all as NOT duplicates — no posts made.", level=0)
        else:
            log.phase("Result", "Nothing to post (all pairs already known to Hydrus).", level=0)
        return

    # 6) Post new potential relationships to Hydrus
    log.phase("Post", f"Posting {len(pairs_to_post)} new potential relationships to Hydrus…", level=1)
    rep = post_potential_duplicates(
        session, api_url, api_key, hashes, pairs_to_post,
        do_default_merge=C["post"]["do_default_merge"],
        timeout=C["timeouts"]["post"],
        batch_size=C["post"]["batch_size"],
        max_retries=C["post"]["max_retries"],
        retry_pause_s=C["post"]["retry_pause_s"],
        log=log
    )

    # Final summary (consistent naming)
    log.phase(
        "Result",
        f"Relationships -> Posted: {rep.get('posted', 0)} | Skipped: {rep.get('skipped', 0)} | Failed: {rep.get('failed', 0)} | Attempted: {rep.get('attempted', 0)}",
        level=0
    )


if __name__ == "__main__":
    main()
