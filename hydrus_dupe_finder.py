#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


# ---------- Config / Utils ----------
def _load_config(path="hydrus_config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _headers(api_key):
    return {"Hydrus-Client-API-Access-Key": api_key}

def _want_progress(C):
    return bool(C.get("progress", {}).get("enabled", False)) or bool(C.get("verbose", False))

def _use_tqdm(C):
    return C.get("progress", {}).get("type", "tqdm").lower() == "tqdm" and _HAS_TQDM

def _make_session(C):
    # Robust pooled session + retries for transient network/server issues
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


# ---------- Hydrus API ----------
def api_version(session, api_url, timeout):
    r = session.get(f"{api_url}/api_version", timeout=timeout)
    r.raise_for_status()
    return r.json().get("version")

def verify_access(session, api_url, api_key, timeout, verbose=True):
    r = session.get(f"{api_url}/verify_access_key", headers=_headers(api_key), timeout=timeout)
    if r.status_code == 419:
        raise RuntimeError("Access key invalid or expired (HTTP 419).")
    r.raise_for_status()
    info = r.json()
    if verbose:
        print("Access key name:", info.get("name"))
        print("Permissions:", info.get("human_description"))
        print("permits_everything:", info.get("permits_everything"),
              "basic_permissions:", info.get("basic_permissions"))
    return info

def has_relationship_permission(access_info):
    if bool(access_info.get("permits_everything")):
        return True
    perms = access_info.get("basic_permissions") or []
    str_perms = {str(p) for p in perms}
    return "8" in str_perms  # 8 == Edit File Relationships

def search_hashes(session, api_url, api_key, tags, timeout):
    params = {"tags": json.dumps(tags), "return_hashes": "true", "return_file_ids": "false"}
    r = session.get(f"{api_url}/get_files/search_files", params=params, headers=_headers(api_key), timeout=timeout)
    r.raise_for_status()
    return r.json().get("hashes", [])

def post_potential_duplicates(session, api_url, api_key, file_hashes, pairs, do_default_merge, timeout, batch_size=200, max_retries=3, retry_pause_s=1.0, verbose=True):
    """
    Posts candidate duplicate relationships (relationship=0) in batches.
    Prints server messages on non-200 responses and after final failure.
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
                    break
                else:
                    msg = (r.text or "").strip()
                    fail_msgs.append(f"HTTP {r.status_code}: {msg[:300]}")
                    if verbose:
                        print(f"[post] Non-200: HTTP {r.status_code} :: {msg[:300]}")
                    r.raise_for_status()
            except Exception as e:
                if attempts < max_retries:
                    if verbose:
                        print(f"[post] Error (attempt {attempts}/{max_retries}): {e!r}; retrying in {retry_pause_s}s...")
                    sleep(retry_pause_s)
                else:
                    failed += len(chunk)
                    fail_msgs.append(repr(e))
        attempted += len(chunk)

    if fail_msgs and verbose:
        print("Post failures (sample up to 5):")
        for m in fail_msgs[:5]:
            print("  -", m)

    return {"attempted": attempted, "posted": posted, "skipped": skipped, "failed": failed, "fail_msgs": fail_msgs}


# ---------- aHash ----------
def _fetch_thumb_rgb(session, api_url, api_key, file_hash, timeout):
    r = session.get(f"{api_url}/get_files/thumbnail", params={"hash": file_hash}, headers=_headers(api_key), timeout=timeout)
    r.raise_for_status()
    pil = Image.open(BytesIO(r.content)).convert("RGB")
    return np.asarray(pil, dtype=np.uint8)

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

def fetch_all_ahashes(session, api_url, api_key, file_hashes, resize, timeout, workers, C):
    ah = [None] * len(file_hashes)
    def _job(idx_hash):
        idx, h = idx_hash
        try:
            img = _fetch_thumb_rgb(session, api_url, api_key, h, timeout)
            ah[idx] = compute_ahash_uint64(Image.fromarray(img), resize)
        except Exception:
            ah[idx] = None
    it = enumerate(file_hashes)
    if _want_progress(C) and _use_tqdm(C):
        it = tqdm(list(it), desc="aHash: fetch+hash (all files)")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        ex.map(_job, it)
    return ah  # list of uint64 or None


def bucket_by_prefix(ah_list, prefix_bits):
    """Return dict: prefix -> list(indices)."""
    mask = (0xFFFFFFFFFFFFFFFF >> (64 - prefix_bits)) << (64 - prefix_bits) if prefix_bits > 0 else 0
    buckets = defaultdict(list)
    for i, h in enumerate(ah_list):
        if h is None:
            continue
        key = (h & mask) if prefix_bits > 0 else 0
        buckets[key].append(i)
    return buckets

def candidate_pairs_from_buckets(buckets, C=None):
    """Yield pairs (i,j) within each bucket; optionally progress-estimate."""
    pbar = None
    if _want_progress(C) and _use_tqdm(C):
        total = 0
        for idxs in buckets.values():
            n = len(idxs)
            total += n*(n-1)//2
        pbar = tqdm(total=total, desc="aHash: bucket pair gen")

    for idxs in buckets.values():
        n = len(idxs)
        if n < 2:
            continue
        for i in range(n-1):
            for j in range(i+1, n):
                if pbar: pbar.update(1)
                yield (idxs[i], idxs[j])

    if pbar: pbar.close()

def filter_pairs_by_hamming(ah_list, pairs, hamming_max, C):
    keep = []
    for i, j in pairs:
        hi, hj = ah_list[i], ah_list[j]
        if hi is None or hj is None:
            continue
        if hamming64(hi, hj) <= hamming_max:
            keep.append((i, j))
    return keep


# ---------- Preprocess & SSIM ----------
def trim_borders(im_rgb, strength, min_keep_frac):
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

def ssim_parallel_compare(api_url, api_key, file_hashes, pairs, P, thr, timeout, par_cfg, C):
    pairs = list(pairs)
    if not pairs:
        return []
    max_workers = par_cfg.get("max_workers", 0) or os.cpu_count() or 1
    chunksize = int(par_cfg.get("chunksize", 1))
    tasks = [(i, j, api_url, api_key, file_hashes[i], file_hashes[j], P, thr, timeout) for (i, j) in pairs]
    kept = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        it = ex.map(_ssim_worker, tasks, chunksize=chunksize)
        if _want_progress(C) and _use_tqdm(C):
            it = tqdm(it, total=len(tasks), desc="SSIM compare (parallel)")
        for res in it:
            if res is not None:
                kept.append(res)
    return kept


# ---------- Main ----------
def main():
    C = _load_config("hydrus_config.json")
    api_url = f"{C['api_url']}:{C['api_port']}"
    api_key = C["api_key"]
    VERBOSE = bool(C.get("verbose", False))

    # Create session FIRST (fixes the NameError you saw)
    session = _make_session(C)

    # 0) API + permissions preflight
    ver = api_version(session, api_url, C["timeouts"]["api"])
    if VERBOSE:
        print(f"Hydrus API version: {ver}")

    access = verify_access(session, api_url, api_key, C["timeouts"]["api"], verbose=VERBOSE)
    if not has_relationship_permission(access):
        print("\n[FATAL] Your access key lacks 'Edit File Relationships' (perm id 8).")
        sys.exit(1)

    # 1) Pull files by tags
    hashes = search_hashes(session, api_url, api_key, C["tags"], C["timeouts"]["api"])
    n = len(hashes)
    print(f"Got {n} hashes")
    if n == 0:
        return

    # 2) aHash for every file (parallel, threaded)
    A = fetch_all_ahashes(
        session, api_url, api_key, hashes,
        resize=C["ahash"]["resize"],
        timeout=C["timeouts"]["thumb"],
        workers=int(C["ahash"]["io_workers"]),
        C=C
    )

    # 3) Bucket by aHash prefix to avoid O(N^2), then filter by Hamming
    buckets = bucket_by_prefix(A, int(C["ahash"]["prefixBits"]))
    cand_iter = candidate_pairs_from_buckets(buckets, C=C)
    ham_pairs = filter_pairs_by_hamming(A, cand_iter, int(C["ahash"]["hammingMax"]), C)
    print(f"aHash kept: {len(ham_pairs)} candidate pairs")

    if not ham_pairs:
        print("No aHash candidates; exiting.")
        return

    # 4) SSIM (parallel, processes) — no MS-SSIM
    dup_pairs = ssim_parallel_compare(
        api_url, api_key, hashes, ham_pairs,
        P=C["preprocessing"],
        thr=float(C["similarity_threshold"]),
        timeout=C["timeouts"]["thumb"],
        par_cfg=C["ssim_parallel"],
        C=C
    )
    print(f"SSIM confirmed {len(dup_pairs)} duplicate pairs")

    if not dup_pairs:
        print("Nothing to post.")
        return

    # 5) Post candidates to Hydrus (robust + messages)
    rep = post_potential_duplicates(
        session, api_url, api_key, hashes, dup_pairs,
        do_default_merge=C["post"]["do_default_merge"],
        timeout=C["timeouts"]["post"],
        batch_size=C["post"]["batch_size"],
        max_retries=C["post"]["max_retries"],
        retry_pause_s=C["post"]["retry_pause_s"],
        verbose=VERBOSE
    )
    print(f"Relationships -> Posted: {rep.get('posted', 0)} | Skipped: {rep.get('skipped', 0)} | Failed: {rep.get('failed', 0)} | Attempted: {rep.get('attempted', 0)}")


if __name__ == "__main__":
    main()
