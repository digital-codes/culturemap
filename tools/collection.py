#!/usr/bin/env python3
import os, glob, random, math, argparse
from PIL import Image, ImageOps

def best_grid(n, target_aspect):
    """
    Choose cols/rows so (cols/rows) ~= target_aspect, with enough cells for n.
    Brute force over possible cols.
    """
    best = None
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        aspect = cols / rows
        diff = abs(aspect - target_aspect)
        area = cols * rows
        # tie-break: fewer unused cells, then smaller area
        unused = area - n
        key = (diff, unused, area)
        if best is None or key < best[0]:
            best = (key, cols, rows)
    return best[1], best[2]

def pad_to_aspect(img, target_aspect, color=(255, 255, 255)):
    w, h = img.size
    cur = w / h
    if abs(cur - target_aspect) < 1e-6:
        return img
    if cur > target_aspect:
        # too wide -> increase height
        new_h = int(round(w / target_aspect))
        pad_total = max(0, new_h - h)
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        return ImageOps.expand(img, border=(0, pad_top, 0, pad_bottom), fill=color)
    else:
        # too tall -> increase width
        new_w = int(round(h * target_aspect))
        pad_total = max(0, new_w - w)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return ImageOps.expand(img, border=(pad_left, 0, pad_right, 0), fill=color)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True, help="Board image path (for aspect ratio)")
    ap.add_argument("--closeups", required=True, help="Folder or glob, e.g. closeups/*.png")
    ap.add_argument("--out", required=True, help="Output collage image path (png/jpg)")
    ap.add_argument("--tile", type=int, default=256, help="Tile size (default 256)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (0 = random)")
    ap.add_argument("--exact_aspect", action="store_true",
                    help="Pad the final collage to EXACT board aspect ratio")
    ap.add_argument("--bg", default="255,255,255", help="Background as R,G,B (for padding/empty)")
    args = ap.parse_args()

    bg = tuple(int(x) for x in args.bg.split(","))

    board = Image.open(args.board)
    bw, bh = board.size
    target_aspect = bw / bh

    # collect closeup paths
    if os.path.isdir(args.closeups):
        paths = sorted(glob.glob(os.path.join(args.closeups, "*.*")))
    else:
        paths = sorted(glob.glob(args.closeups))
    if not paths:
        raise SystemExit("No closeups found.")

    if args.seed == 0:
        random.shuffle(paths)
    else:
        random.Random(args.seed).shuffle(paths)

    n = len(paths)
    cols, rows = best_grid(n, target_aspect)

    W = cols * args.tile
    H = rows * args.tile
    canvas = Image.new("RGB", (W, H), bg)

    # paste tiles
    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        # fit into tile (keeps aspect; pads inside tile if needed)
        im = ImageOps.pad(im, (args.tile, args.tile), color=bg, method=Image.Resampling.LANCZOS)
        x = (i % cols) * args.tile
        y = (i // cols) * args.tile
        canvas.paste(im, (x, y))

    if args.exact_aspect:
        canvas = pad_to_aspect(canvas, target_aspect, color=bg)

    # save
    canvas.save(args.out, quality=95)
    print(f"Closeups: {n}")
    print(f"Grid: {cols} x {rows} tiles, tile={args.tile}px")
    print(f"Output: {args.out} ({canvas.size[0]}x{canvas.size[1]})")
    print(f"Target aspect (board): {target_aspect:.6f}, output aspect: {(canvas.size[0]/canvas.size[1]):.6f}")

if __name__ == "__main__":
    main()
