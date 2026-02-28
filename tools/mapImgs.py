#!/usr/bin/env python3
import os
import glob
import csv
import argparse
from dataclasses import dataclass

import cv2
import numpy as np


# -----------------------------
# Utils
# -----------------------------
def imread_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb_to_gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def clahe_gray(gray, clip=3.0, tile=(8, 8)):
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return c.apply(gray)

def resize_max_side(img, max_side):
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img, 1.0
    scale = max_side / float(s)
    new = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(img, new, interpolation=cv2.INTER_AREA), scale

def safe_mkdir(p):
    if p:
        os.makedirs(p, exist_ok=True)

@dataclass
class MatchResult:
    ok: bool
    x: float = np.nan
    y: float = np.nan
    inliers: int = 0
    matches: int = 0
    score: float = 0.0
    msg: str = ""


# -----------------------------
# Sticker isolation (optional but helpful)
# -----------------------------
def crop_largest_quad(gray):
    """
    Attempts to find the largest "paper-like" contour and returns a perspective-
    rectified crop. If it fails, returns the original gray and identity transform.
    """
    g = gray.copy()
    g = cv2.GaussianBlur(g, (5, 5), 0)
    # Edge-based since low contrast can break thresholding
    edges = cv2.Canny(g, 30, 120)

    # Close gaps
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gray, np.eye(3, dtype=np.float32)

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 0.05 * gray.shape[0] * gray.shape[1]:
        # too small, probably noise
        return gray, np.eye(3, dtype=np.float32)

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) < 4:
        return gray, np.eye(3, dtype=np.float32)

    # Use minAreaRect for robust quad
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)

    # Order points: tl, tr, br, bl
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1).ravel()
    tl = box[np.argmin(s)]
    br = box[np.argmax(s)]
    tr = box[np.argmin(diff)]
    bl = box[np.argmax(diff)]
    src = np.array([tl, tr, br, bl], dtype=np.float32)

    # Target size
    w1 = np.linalg.norm(br - bl)
    w2 = np.linalg.norm(tr - tl)
    h1 = np.linalg.norm(tr - br)
    h2 = np.linalg.norm(tl - bl)
    W = int(round(max(w1, w2)))
    H = int(round(max(h1, h2)))
    W = max(W, 200)
    H = max(H, 200)

    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(gray, M, (W, H), flags=cv2.INTER_CUBIC)
    return warped, M


# -----------------------------
# Feature extractor (SIFT if available, else AKAZE)
# -----------------------------
def make_detector():
    # Prefer SIFT (best for this), fallback to AKAZE
    if hasattr(cv2, "SIFT_create"):
        det = cv2.SIFT_create(nfeatures=6000)
        norm = cv2.NORM_L2
        is_binary = False
        return det, norm, is_binary
    det = cv2.AKAZE_create()
    norm = cv2.NORM_HAMMING
    is_binary = True
    return det, norm, is_binary

def make_matcher(norm, is_binary):
    if not is_binary:
        # FLANN for float descriptors
        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=64)
        return cv2.FlannBasedMatcher(index_params, search_params)
    # Brute force is fine for binary
    return cv2.BFMatcher(norm, crossCheck=False)


# -----------------------------
# Main matching
# -----------------------------
def compute_features(det, gray):
    kps, des = det.detectAndCompute(gray, None)
    if des is None or len(kps) < 10:
        return [], None
    return kps, des

def project_center(H, w, h):
    pt = np.array([[[w / 2.0, h / 2.0]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(out[0]), float(out[1])

def match_closeup_to_board(board_gray, board_kp, board_des,
                           close_rgb, det, matcher,
                           ratio=0.75, ransac=5.0,
                           try_crop=True):
    close_gray = rgb_to_gray(close_rgb)
    close_gray = clahe_gray(close_gray)

    # Optional: rectify sticker region to get cleaner features
    if try_crop:
        warped, M_crop = crop_largest_quad(close_gray)
        warped = clahe_gray(warped)
        close_for_feats = warped
        # We later estimate H_warped->board. For center mapping in warped coords, that's enough.
    else:
        close_for_feats = close_gray

    kp2, des2 = compute_features(det, close_for_feats)
    if des2 is None:
        return MatchResult(False, msg="too_few_features")

    # knnMatch + Lowe ratio
    knn = matcher.knnMatch(des2, board_des, k=2)
    good = []
    for m_n in knn:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 12:
        return MatchResult(False, matches=len(good), msg="too_few_good_matches")

    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([board_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac)
    if H is None or mask is None:
        return MatchResult(False, matches=len(good), msg="homography_failed")

    inliers = int(mask.ravel().sum())
    if inliers < 10:
        return MatchResult(False, inliers=inliers, matches=len(good), msg="too_few_inliers")

    h, w = close_for_feats.shape[:2]
    x, y = project_center(H, w, h)

    # Simple confidence score: inliers * inlier_ratio
    score = float(inliers) * (float(inliers) / float(len(good)))

    return MatchResult(True, x=x, y=y, inliers=inliers, matches=len(good), score=score, msg="ok"), H, close_for_feats


def draw_debug(board_rgb, close_gray, H, out_path, x, y):
    dbg = board_rgb.copy()
    # draw projected center
    cv2.circle(dbg, (int(round(x)), int(round(y))), 16, (255, 0, 0), 3)

    # draw projected closeup corners
    h, w = close_gray.shape[:2]
    corners = np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]], dtype=np.float32)
    proj = cv2.perspectiveTransform(corners, H).astype(int).reshape(-1, 2)
    cv2.polylines(dbg, [proj], True, (0, 255, 0), 3)

    cv2.imwrite(out_path, cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True, help="Path to big board image")
    ap.add_argument("--closeups", required=True, help="Glob or folder for closeup images, e.g. 'closeups/*.jpg'")
    ap.add_argument("--out_csv", default="matches.csv")
    ap.add_argument("--debug_dir", default="", help="If set, writes debug overlay images here")
    ap.add_argument("--board_max_side", type=int, default=4096, help="Downscale board for speed (keeps coords in downscaled space)")
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac", type=float, default=5.0)
    ap.add_argument("--no_crop", action="store_true", help="Disable closeup quad-crop/rectification")
    args = ap.parse_args()

    safe_mkdir(args.debug_dir)

    board_rgb = imread_rgb(args.board)
    board_rgb, s_board = resize_max_side(board_rgb, args.board_max_side)
    board_gray = clahe_gray(rgb_to_gray(board_rgb))

    det, norm, is_binary = make_detector()
    matcher = make_matcher(norm, is_binary)

    board_kp, board_des = compute_features(det, board_gray)
    if board_des is None:
        raise RuntimeError("Could not compute features on board image")

    # Resolve closeups list
    closeup_paths = []
    if os.path.isdir(args.closeups):
        closeup_paths = sorted(glob.glob(os.path.join(args.closeups, "*.*")))
    else:
        closeup_paths = sorted(glob.glob(args.closeups))

    if not closeup_paths:
        raise RuntimeError("No closeup images found")

    rows = []
    for p in closeup_paths:
        try:
            close_rgb = imread_rgb(p)
        except Exception as e:
            rows.append([p, "", "", 0, 0, 0.0, f"read_error:{e}"])
            continue

        res = match_closeup_to_board(
            board_gray, board_kp, board_des,
            close_rgb, det, matcher,
            ratio=args.ratio, ransac=args.ransac,
            try_crop=not args.no_crop
        )

        # match_closeup_to_board returns either MatchResult or (MatchResult, H, close_gray)
        if isinstance(res, tuple):
            mr, H, close_used = res
        else:
            mr, H, close_used = res, None, None

        rows.append([p, mr.x, mr.y, mr.inliers, mr.matches, mr.score, mr.msg])

        if args.debug_dir and mr.ok and H is not None:
            out_img = os.path.join(args.debug_dir, os.path.basename(p) + "_overlay.jpg")
            draw_debug(board_rgb, close_used, H, out_img, mr.x, mr.y)

    # Write CSV (coords are in the possibly-downscaled board coordinate system)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["closeup_path", "x_board", "y_board", "inliers", "good_matches", "score", "status"])
        w.writerows(rows)

    print(f"Wrote: {args.out_csv}")
    if args.debug_dir:
        print(f"Debug overlays in: {args.debug_dir}")
    if s_board != 1.0:
        print(f"NOTE: board was downscaled by {s_board:.4f}. Multiply x/y by {1.0/s_board:.4f} to map to original board pixels.")


if __name__ == "__main__":
    main()
