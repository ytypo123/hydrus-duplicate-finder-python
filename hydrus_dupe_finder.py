import json
import math
from io import BytesIO
from itertools import combinations
from time import sleep
from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np
import requests
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim

# Optional progress bar
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# Optional MS-SSIM
try:
    import torch
    from pytorch_msssim import ms_ssim as torch_ms_ssim
    _HAS_MSSSIM = True
except ImportError:
    _HAS_MSSSIM = False


# ---------- Utilities ----------
def _load_config(path="hydrus_config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _headers(api_key):
    return {"Hydrus-Client-API-Access-Key": api_key}

def _want_progress(C):
    return bool(C.get("progress", {}).get("enabled", False)) or bool(C.get("verbose", False))

def _use_tqdm(C):
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
    attempted = posted = skipped = failed = 0
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
    bar = tqdm(total=total, desc="Duration filter") if show_prog and total > 0 else None
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
    iterable = tqdm(candidate_pairs, desc="AR filter") if _want_progress(C) and _use_tqdm(C) else candidate_pairs
    return [
        (i, j) for i, j in iterable
        if ars[i] and ars[j] and abs(ars[i] - ars[j]) <= ar_tol * max(ars[i], ars[j])
    ]


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
    for idx, bit in enumerate(bits, start=1):
        if bit:
            h |= (1 << (64 - idx))
    return h

def hamming64(a, b):
    return bin(a ^ b).count("1")

def prefilter_pairs_with_ahash(api_url, api_key, file_hashes, pairs, resize, hamming_max, timeout, C):
    endpoint = f"{api_url}/get_files/thumbnail"
    needed = sorted(set(i for p in pairs for i in p))
    a_hashes = {}
    iterable = tqdm(needed, desc="aHash: fetch+hash") if _want_progress(C) and _use_tqdm(C) else needed
    for idx in iterable:
        try:
            r = requests.get(endpoint, params={"hash": file_hashes[idx]}, headers=_headers(api_key), timeout=timeout)
            r.raise_for_status()
            pil = Image.open(BytesIO(r.content)).convert("RGB")
            a_hashes[idx] = compute_ahash_uint64(pil, resize)
        except Exception:
            a_hashes[idx] = None

    keep = []
    iterable_pairs = tqdm(pairs, desc="aHash: filter pairs") if _want_progress(C) and _use_tqdm(C) else pairs
    for i, j in iterable_pairs:
        if a_hashes.get(i) is not None and a_hashes.get(j) is not None:
            if hamming64(a_hashes[i], a_hashes[j]) <= hamming_max:
                keep.append((i, j))
    return keep


# ---------- Image preprocessing & SSIM ----------
def trim_borders(im_rgb, strength, min_keep_frac):
    g = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY) if im_rgb.ndim == 3 and im_rgb.shape[2] == 3 else im_rgb
    g = g.astype(np.float32) / 255.0
    row_std = np.convolve(np.pad(g.std(axis=1), 2, mode="edge"), np.ones(5)/5, mode="valid")
    col_std = np.convolve(np.pad(g.std(axis=0), 2, mode="edge"), np.ones(5)/5, mode="valid")
    thr_r = strength * row_std.max()
    thr_c = strength * col_std.max()
    H, W = g.shape[:2]

    def first_above(a, thr, default): return int(np.argmax(a > thr)) if (a > thr).any() else default
    def last_above(a, thr, default): return int(len(a) - 1 - np.argmax((a > thr)[::-1])) if (a > thr).any() else default

    top, bottom = first_above(row_std, thr_r, 0), last_above(row_std, thr_r, H - 1)
    left, right = first_above(col_std, thr_c, 0), last_above(col_std, thr_c, W - 1)
    if bottom - top + 1 < int(min_keep_frac * H): top, bottom = 0, H - 1
    if right - left + 1 < int(min_keep_frac * W): left, right = 0, W - 1
    return im_rgb[top:bottom+1, left:right+1, :]

def central_crop(im_rgb, frac):
    H, W = im_rgb.shape[:2]
    h, w = max(1, int(round(H * frac))), max(1, int(round(W * frac)))
    y0, x0 = (H - h) // 2, (W - w) // 2
    return im_rgb[y0:y0 + h, x0:x0 + w, :]

def preprocess_thumb(im_rgb, P):
    im = trim_borders(im_rgb, P["trimStrength"], P["minKeepFrac"])
    if P["fixedResize"]:
        im = cv2.resize(im, tuple(P["fixedResize"]), interpolation=cv2.INTER_AREA)
    if P["toGray"]:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    if P["preBlurSigma"] > 0:
        im = cv2.GaussianBlur(im, (0, 0), sigmaX=P["preBlurSigma"])
    if P["histEq"]:
        if im.ndim == 2:
            im = cv2.equalizeHist(im.astype(np.uint8))
        else:
            im = np.stack([cv2.equalizeHist(im[:, :, c].astype(np.uint8)) for c in range(3)], axis=2)
    return im

def _fetch_thumb_rgb(api_url, api_key, file_hash, timeout):
    r = requests.get(f"{api_url}/get_files/thumbnail", params={"hash": file_hash}, headers=_headers(api_key), timeout=timeout)
    r.raise_for_status()
    return np.asarray(Image.open(BytesIO(r.content)).convert("RGB"), dtype=np.uint8)

def _score_pair(im1_proc, im2_proc, P):
    if P["centerFrac"] < 1.0:
        im1_proc, im2_proc = central_crop(im1_proc, P["centerFrac"]), central_crop(im2_proc, P["centerFrac"])
    im1_g = cv2.cvtColor(im1_proc, cv2.COLOR_RGB2GRAY) if im1_proc.ndim == 3 else im1_proc
    im2_g = cv2.cvtColor(im2_proc, cv2.COLOR_RGB2GRAY) if im2_proc.ndim == 3 else im2_proc
    if P["useMSSSIM"] and _HAS_MSSSIM:
        with torch.no_grad():
            t1 = torch.from_numpy(im1_g.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            t2 = torch.from_numpy(im2_g.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            return float(torch_ms_ssim(t1, t2, data_range=1.0).item())
    return float(ssim(im1_g, im2_g, data_range=255))

def _worker_compare_pair(args):
    (i, j, api_url, api_key, file_hash_i, file_hash_j, P, threshold, timeout) = args
    try:
        im1 = _fetch_thumb_rgb(api_url, api_key, file_hash_i, timeout)
        im2 = _fetch_thumb_rgb(api_url, api_key, file_hash_j, timeout)
        if _score_pair(preprocess_thumb(im1, P), preprocess_thumb(im2, P), P) >= threshold:
            return (i, j)
    except Exception:
        pass
    return None

def compare_thumbnails(api_url, api_key, file_hashes, pairs, threshold, P, timeout, C):
    par = C.get("ssim_parallel", {})
    if not par.get("enabled", False):
        iterable = tqdm(pairs, desc="SSIM compare") if _want_progress(C) and _use_tqdm(C) else pairs
        kept = []
        for i, j in iterable:
            try:
                if _score_pair(preprocess_thumb(_fetch_thumb_rgb(api_url, api_key, file_hashes[i], timeout),
                                                P),
                               preprocess_thumb(_fetch_thumb_rgb(api_url, api_key, file_hashes[j], timeout),
                                                P), P) >= threshold:
                    kept.append((i, j))
            except Exception:
                pass
        return kept

    max_workers = par.get("max_workers", 0) or os.cpu_count() or 1
    chunksize = int(par.get("chunksize", 1))
    tasks = [(i, j, api_url, api_key, file_hashes[i], file_hashes[j], P, threshold, timeout) for (i, j) in pairs]
    kept = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        iterable = ex.map(_worker_compare_pair, tasks, chunksize=chunksize)
        if _want_progress(C) and _use_tqdm(C):
            iterable = tqdm(iterable, total=len(tasks), desc="SSIM compare (parallel)")
        for res in iterable:
            if res is not None:
                kept.append(res)
    return kept


# ---------- Main ----------
# ... all your imports and function definitions stay the same ...

def main():
    C = _load_config("hydrus_config.json")

    api_url = f"{C['api_url']}:{C['api_port']}"
    api_key = C["api_key"]
    VERBOSE = bool(C.get("verbose", False))
    skip = C.get("skip_filters", {})

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

    # Step 3: Candidate generation
    if not skip.get("duration_filter", False):
        if VERBOSE:
            print("Generating duration-based candidate pairs...")
        dur_pairs = find_pairs_by_duration(durations, C["candidate_generation"]["tolSeconds"], C)
        print(f"Duration candidates: {len(dur_pairs)}")
    else:
        dur_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        print(f"[SKIP] Duration filter skipped: starting with {len(dur_pairs)} total pairs")

    if not skip.get("ar_filter", False):
        if VERBOSE:
            print("Applying AR filter...")
        ar_pairs = find_pairs_by_ar(ars, C["candidate_generation"]["arTol"], dur_pairs, C)
        print(f"AR-filtered candidates: {len(ar_pairs)}")
    else:
        ar_pairs = dur_pairs
        print(f"[SKIP] AR filter skipped: {len(ar_pairs)} pairs remain")

    if not skip.get("ahash_prefilter", False):
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
        print(f"[SKIP] aHash prefilter skipped: passing {len(ar_pairs)} pairs to SSIM.")

    # Step 4: SSIM/MS-SSIM compare
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

    # Step 5: Post results
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