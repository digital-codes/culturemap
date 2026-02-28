import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import distance
import os
import json
from matplotlib.patches import Rectangle
import argparse
import geopandas as gp
from shapely.geometry import Point

def get_map_dimensions(map_path):
    with Image.open(map_path) as img:
        return img.size  # Returns (width, height)

def distribute_stickers(num_stickers, map_width, map_height, min_distance=20):
    # Generate random positions for stickers
    np.random.seed(42)  # For reproducibility
    # Calculate sticker size based on map dimensions and number of stickers
    sticker_size = min(map_width, map_height) // 16  # Adjust divisor to change sticker size
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


def plot_stickers(sticker_positions, angles, sticker_paths, sticker_size, map_path, cardBox=False, square=False, output_path="sticker_distribution.png"):
    map_img = Image.open(map_path)
    if square:
        # Crop to square
        min_dim = min(map_img.size)
        left = (map_img.width - min_dim) / 2
        top = (map_img.height - min_dim) / 2
        right = left + min_dim
        bottom = top + min_dim
        map_img = map_img.crop((left, top, right, bottom))
    map_width, map_height = map_img.size
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(map_img, extent=(0, map_width, 0, map_height))
    ax.set_xlim(0, map_width)
    ax.set_ylim(0, map_height)
    ax.set_aspect('equal')
    ax.axis('off')

    for i, pos in enumerate(sticker_positions):
        sticker_path = sticker_paths[i % len(sticker_paths)]
        try:
            img = Image.open(sticker_path)
            img.thumbnail((sticker_size, sticker_size))  # Resize sticker
            rotated_img = img.rotate(angles[i], expand=True)
            ax.imshow(rotated_img, extent=(pos[0], pos[0] + sticker_size, pos[1], pos[1] + sticker_size), alpha=0.9)
            if cardBox:
                # Draw blue bounding box around sticker
                rect = Rectangle((pos[0], pos[1]), sticker_size, sticker_size, linewidth=2, edgecolor='blue', facecolor='none')
                ax.add_patch(rect)
        except Exception as e:
            print(f"Error processing sticker {sticker_path}: {e}, index: {i}, position: {pos}")
            print(f"Error loading image {sticker_path}: {e}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


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
    parsed = parser.parse_args(args)

    sticker_dir = parsed.card_dir
    map_path = parsed.map_path
    output_base = parsed.output_base
    cardBox = parsed.bbox
    square = parsed.square
    sticker_paths = [os.path.join(sticker_dir, f) for f in os.listdir(sticker_dir)]
    num_stickers = len(sticker_paths)
    print(f"Number of stickers: {num_stickers}")

    map_width, map_height = get_map_dimensions(map_path)
    sticker_positions, sticker_size = distribute_stickers(num_stickers, map_width, map_height)
    sticker_angles = [random.uniform(-30, 30) for _ in range(len(sticker_positions))]
    plot_stickers(sticker_positions, sticker_angles, sticker_paths, sticker_size, map_path, cardBox=cardBox, square=square, output_path=f"{output_base}.png")

    with open(f"{output_base}.json", "w") as f:
        json.dump({
            "sticker_paths": sticker_paths,
            "sticker_positions": sticker_positions.tolist(),
            "sticker_angles": sticker_angles,
            "sticker_size": sticker_size
        }, f, indent=4)
        

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
