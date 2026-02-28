import cv2
from geopandas import overlay
import numpy as np
from PIL import Image
from skimage import io
import svgwrite
import matplotlib.pyplot as plt
import sys
import os
from PIL import Image, ImageDraw


def load_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to enhance contrast
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    image = np.array(thresh)
    return image
    
    # Invert the image to make the text more visible
    #inverted = 255 - thresh
    #return inverted


def load_image_1(image_path):
    # Load the image using Pillow to support WEBP format
    pil_image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = np.array(pil_image)
    return image

def threshold_image(image, threshold=150): # 128):
    _, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return thresh

def invert_image(image):
    return cv2.bitwise_not(image)

def vectorize_image(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def display_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def display_contours(image, contours, title):
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)
    plt.imshow(contour_image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def display_vectorized_image(contours, width, height):
    fig, ax = plt.subplots()
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert the y-axis to match image coordinates
    for contour in contours:
        points = np.array(contour).reshape(-1, 2)
        ax.plot(points[:, 0], points[:, 1], color='#000')
    plt.title('Vectorized Image')
    plt.axis('off')
    plt.show()

def export_svg(contours, svg_path, width, height, background_color='#fff', line_color='#000'):
    dwg = svgwrite.Drawing(svg_path, profile='tiny', size=(width, height))
    if background_color != 'none':
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill=background_color))
    for contour in contours:
        points = [(int(point[0][0]), int(point[0][1])) for point in contour]
        if len(points) > 2:  # Ensure there are enough points to form a polygon
            poly = dwg.polygon(points, fill=line_color, fill_opacity = .4, stroke=line_color, stroke_width=5)
            dwg.add(poly)
                
    dwg.save()

def process_image(image_path):
    image = load_image(image_path)

    image_base = image_path.split(".")[0]
    # Display the imported image
    #display_image(image, 'Imported Image')
    
    thresh_image = threshold_image(image)
    
    # Display the thresholded image
    #display_image(thresh_image, 'Thresholded Image')
    
    inverted_image = invert_image(thresh_image)
    
    # Display the inverted thresholded image
    #display_image(inverted_image, 'Inverted Thresholded Image')
    
    contours, hierarchy = vectorize_image(thresh_image)
    # ensure we have mutable lists (and handle None) because some OpenCV builds return tuples
    contours = list(contours) if contours is not None else []
    inverted_contours, _ = vectorize_image(inverted_image)
    inverted_contours = list(inverted_contours) if inverted_contours is not None else []
    
    height, width = image.shape

    # Display contours on the original thresholded image
    #display_contours(thresh_image, contours, 'Contours on Thresholded Image')
    
    # Display contours on the inverted thresholded image
    #display_contours(inverted_image, inverted_contours, 'Contours on Inverted Thresholded Image')

    # Display the vectorized image
    #display_vectorized_image(contours, width, height)

    # in many cases there is an object which covers the whole image. remove it
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 0.95 * width * height:
            print("Removing large contour")
            # remove by index using array comparison to avoid ambiguous truth-value for numpy arrays
            for i, c in enumerate(contours):
                if np.array_equal(c, max_contour):
                    del contours[i]
                    break
    if len(inverted_contours) > 0:
        max_contour = max(inverted_contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 0.95 * width * height:
            print("Removing large contour")
            for i, c in enumerate(inverted_contours):
                if np.array_equal(c, max_contour):
                    del inverted_contours[i]
                    break

    
    export_svg(contours, f'{image_base}_black_on_white.svg', width, height, background_color='#ffffff', line_color='#000000')
    #export_svg(inverted_contours, f'{image_base}_white_on_black.svg', width, height, background_color='#000000', line_color='#ffffff')
    export_svg(contours, f'{image_base}_black_on_transparent.svg', width, height, background_color='none', line_color='#000000')
    #export_svg(inverted_contours, f'{image_base}_white_on_transparent.svg', width, height, background_color='none', line_color='#ffffff')

    # load original color image and ensure RGBA
    color_img = Image.open(image_path).convert("RGBA")

    # read the exported svg _black_on<_transpearent.svg as a raster image
    svg_raster_path = f'{image_base}_black_on_transparent.png'
    os.system(f'rsvg-convert -w {width} -h {height} {image_base}_black_on_transparent.svg -o {svg_raster_path}')
    overlay = Image.open(svg_raster_path).convert("RGBA")

    # composite overlay onto original color image
    result = Image.alpha_composite(color_img, overlay)

    # save to file_ovl.ext (preserve original extension)
    ext = os.path.splitext(image_path)[1]
    overlay_path = f"{image_base}_ovl{ext}"
    # convert to RGB for formats that don't support alpha (e.g. JPEG)
    if ext.lower() in (".jpg", ".jpeg"):
        result.convert("RGB").save(overlay_path)
    else:
        result.save(overlay_path)

    # Ensure the saved overlay has a minimum dimension of 1024 (scale up or down preserving aspect ratio)
    target_min = 1024
    img = Image.open(overlay_path)
    w, h = img.size
    if min(w, h) != target_min:
        scale = target_min / min(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        # Convert for formats that don't support alpha if needed
        if ext.lower() in (".jpg", ".jpeg"):
            img = img.convert("RGB")
        img.save(overlay_path)

    print(f"Saved overlay to {overlay_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Provide image name")
        process_image("emissions.webp")
        sys.exit()
    process_image(sys.argv[1])
