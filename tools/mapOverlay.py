#!/usr/bin/env python3
import csv
import argparse
import cv2
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True, help="Original board image path")
    ap.add_argument("--csv", required=True, help="matches.csv path")
    ap.add_argument("--out", required=True, help="Output overlay image path (png/jpg)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Board downscale factor used when matching. If matching used a downscaled board, pass that factor here.")
    ap.add_argument("--only_ok", action="store_true", help="Only plot rows with status == ok")
    ap.add_argument("--start_index", type=int, default=1, help="Marker numbering start")
    args = ap.parse_args()

    img = cv2.imread(args.board, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.board)

    # Draw settings (OpenCV uses BGR)
    circle_color = (0, 0, 255)
    text_color = (255, 255, 255)
    outline_color = (0, 0, 0)

    h, w = img.shape[:2]
    # Scale marker size with image
    base = max(h, w)
    radius = max(8, base // 250)
    thickness = max(2, base // 700)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, base / 2500.0)
    text_thickness = max(1, base // 1200)

    plotted = 0
    idx = args.start_index

    with open(args.csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            status = (row.get("status") or "").strip()
            if args.only_ok and status != "ok":
                idx += 1
                continue

            try:
                x = float(row["x_board"])
                y = float(row["y_board"])
            except Exception:
                idx += 1
                continue

            if not np.isfinite(x) or not np.isfinite(y):
                idx += 1
                continue

            # If matching was done on a downscaled board, coordinates are in that downscaled space.
            # Convert to original pixels:
            x0 = int(round(x / args.scale))
            y0 = int(round(y / args.scale))

            # Skip if out of bounds (allow small margin)
            if x0 < -50 or y0 < -50 or x0 > w + 50 or y0 > h + 50:
                idx += 1
                continue

            # Draw marker
            cv2.circle(img, (x0, y0), radius, circle_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (x0, y0), radius, outline_color, thickness, lineType=cv2.LINE_AA)

            label = str(idx)

            # Place text with outline for readability
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
            tx = x0 + radius + 4
            ty = y0 + (th // 2)

            # Outline
            cv2.putText(img, label, (tx, ty), font, font_scale, outline_color,
                        text_thickness + 2, lineType=cv2.LINE_AA)
            # Fill
            cv2.putText(img, label, (tx, ty), font, font_scale, text_color,
                        text_thickness, lineType=cv2.LINE_AA)

            plotted += 1
            idx += 1

    if not cv2.imwrite(args.out, img):
        raise RuntimeError(f"Failed to write {args.out}")

    print(f"Wrote overlay: {args.out}")
    print(f"Plotted markers: {plotted}")

if __name__ == "__main__":
    main()

