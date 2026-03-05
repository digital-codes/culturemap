import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from scipy.spatial import distance
import os
import json
from matplotlib.patches import Rectangle
import argparse
import geopandas as gp
from shapely.geometry import Point

def get_map(map_path,square=False):
    with Image.open(map_path) as img:
        if square:
            # Crop to square
            min_dim = min(img.size)
            left = (img.width - min_dim) / 2
            top = (img.height - min_dim) / 2
            right = left + min_dim
            bottom = top + min_dim
            img = img.crop((left, top, right, bottom))
        else:
            img = img.convert("RGB")  # Ensure image is in RGB mode

    #map_width, map_height = img.size

    return img  # Returns (width, height)
    #map_width, map_height = map_img.size
    # return img.size  # Returns (width, height)

def distribute_stickers(num_stickers, map_width, map_height):
    # Generate random positions for stickers
    np.random.seed(42)  # For reproducibility
    # Calculate sticker size based on map dimensions and number of stickers
    sticker_size = min(map_width, map_height) // 14  # Adjust divisor to change sticker size
    min_distance = sticker_size * .9  # Minimum distance between stickers based on size
    # positions, shift down by half sticker size to center them
    sticker_positions = np.random.rand(num_stickers, 2) * [map_width - sticker_size, map_height - sticker_size] + sticker_size / 2

    # Function to ensure stickers are not overlapping more than 10%
    def adjust_positions(positions, min_distance):
        for i in range(len(positions)):
            while True:
                if i > 0:
                    distances = distance.cdist([positions[i]], positions[:i], 'euclidean')
                    if np.any(distances < 0.9 * min_distance):
                        positions[i] = np.random.rand(2) * [map_width, map_height]
                        continue
                if i < len(positions) - 1:
                    distances = distance.cdist([positions[i]], positions[i+1:], 'euclidean')
                    if np.any(distances < 0.9 * min_distance):
                        positions[i] = np.random.rand(2) * [map_width, map_height]
                        continue
                break
        return positions

    # Adjust positions
    sticker_positions = adjust_positions(sticker_positions, min_distance)

    return sticker_positions, sticker_size


def plot_stickers(sticker_positions, angles, sticker_paths, sticker_size, map_img, cardBox=False, output_path="sticker_distribution.png"):
    map_width, map_height = map_img.size
    result_img = map_img.copy()
    draw = ImageDraw.Draw(result_img)
    bboxes = []

    for i, pos in enumerate(sticker_positions):
        sticker_path = sticker_paths[i % len(sticker_paths)]
        try:
            sticker = Image.open(sticker_path)
            sticker.thumbnail((sticker_size, sticker_size))
            rotated_sticker = sticker.rotate(angles[i], expand=True)

            # Paste the rotated sticker onto the map
            result_img.paste(rotated_sticker, (int(pos[0]), int(pos[1])), rotated_sticker if rotated_sticker.mode == 'RGBA' else None)

            # Draw blue bounding box
            bbox = [
                (pos[0], pos[1]),
                (pos[0] + sticker_size, pos[1] + sticker_size)
            ]
            bboxes.append(bbox) 
            if cardBox:
                # Draw blue bounding box
                draw.rectangle(bbox, outline="blue", width=2)
        except Exception as e:
            print(f"Error processing sticker {sticker_path}: {e}")

    result_img.save(output_path)
    return bboxes


def main(args):
    # Example usage
    #map_path = "/home/kugel/daten/work/projekte/okLabs/zkm/useum/generated3.jpg"
    #sticker_dir = "squared_png2"
    parser = argparse.ArgumentParser(description="Distribute stickers on a map")
    parser.add_argument("-c", "--card-dir", required=True, help="Directory containing sticker images")
    parser.add_argument("-m", "--map-path", required=True, help="Path to the map image")
    parser.add_argument("-o", "--output-base", default="stickers", help="Base for generated files (without extension)")
    parser.add_argument("-b", "--bbox", action="store_true", help="card bounding box")
    parser.add_argument("-s", "--square", action="store_true", help="Generate square map")
    parser.add_argument("-n", "--num-stickers", type=int, default=-1, help="Number of stickers to distribute, -1 for all")
    parsed = parser.parse_args(args)

    sticker_dir = parsed.card_dir
    map_path = parsed.map_path
    output_base = parsed.output_base
    cardBox = parsed.bbox
    square = parsed.square
    sticker_paths = [os.path.join(sticker_dir, f) for f in os.listdir(sticker_dir)]
    if parsed.num_stickers != -1:
        sticker_paths = sticker_paths[:parsed.num_stickers]
    num_stickers = len(sticker_paths)
    print(f"Number of stickers: {num_stickers}")

    map_img = get_map(map_path, square=square)
    map_width, map_height = map_img.size
    sticker_positions, sticker_size = distribute_stickers(num_stickers, map_width, map_height)
    sticker_angles = [random.uniform(-30, 30) for _ in range(len(sticker_positions))]
    sticker_boxes = plot_stickers(sticker_positions, sticker_angles, sticker_paths, sticker_size, map_img, cardBox=cardBox, output_path=f"{output_base}.png")

    stickers = []
    for i,s in enumerate(sticker_paths):
        sticker_info = {
            "path": s,
            "name": os.path.basename(s),
            "position": sticker_positions[i].tolist(),
            "angle": sticker_angles[i],
            "size": int(sticker_size),
            "bbox": sticker_boxes[i]
        }
        stickers.append(sticker_info)

    with open(f"{output_base}.json", "w") as f:
        json.dump(stickers, f, indent=4)
        

    geo_bbox = [49.04091, 8.30584, 48.98788,8.49089]  # [min_lat, min_lon, max_lat, max_lon]
    def pixel_to_geo(pixel_pos, map_dimensions, bbox):
        """Convert pixel coordinates to geographic coordinates"""
        map_width, map_height = map_dimensions
        min_lat, min_lon, max_lat, max_lon = bbox
        
        # Normalize pixel coordinates to [0, 1]
        norm_x = pixel_pos[0] / map_width
        norm_y = 1 - (pixel_pos[1] / map_height)  # Flip y-axis
        
        # Map to geographic coordinates
        geo_lon = min_lon + norm_x * (max_lon - min_lon)
        geo_lat = min_lat + norm_y * (max_lat - min_lat)
        
        return [geo_lon, geo_lat]

    features = []
    for i, pos in enumerate(sticker_positions):
        geo_coords = pixel_to_geo(pos, (map_width, map_height), geo_bbox)
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": geo_coords
            },
            "properties": {
                "name": os.path.basename(sticker_paths[i]),
                "rotation": sticker_angles[i],
                "size": int(sticker_size)
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(f"{output_base}.geojson", "w") as f:
        json.dump(geojson, f, indent=2)
        
if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
