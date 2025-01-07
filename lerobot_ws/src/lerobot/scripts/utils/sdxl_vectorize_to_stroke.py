import argparse
import re
import os
import cv2
import numpy as np
import vtracer
import json
import math
from PIL import Image

import torch
from diffusers import AutoPipelineForText2Image

from svgpathtools import svg2paths, Path, CubicBezier
import matplotlib.pyplot as plt

def pass_white(hex_color, threshold=(88, 88, 88)):
    """
    HEX 색상 코드가 RGB 임계값 미만인지 확인하는 함수.
    
    Parameters:
        hex_color (str): HEX 색상 코드 (예: "#FEFEFE").
        threshold (tuple): RGB 임계값 (예: (88, 88, 88)).
        
    Returns:
        bool: RGB 값이 threshold 미만이면 True, 아니면 False.
    """
    # HEX -> RGB 변환
    hex_color = hex_color.lstrip('#')  # '#' 제거
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))  # (R, G, B)로 변환

    # RGB 값이 threshold 미만인지 확인
    return not (rgb[0] < threshold[0] and rgb[1] < threshold[1] and rgb[2] < threshold[2])

def apply_transform(path, transform):
    """
    Apply a 2D translation transform to a path.

    Parameters:
    - path: An svgpathtools Path object.
    - transform: A string representing the transform (e.g., 'translate(x, y)').

    Returns:
    - Transformed Path object.
    """
    if not transform:
        return path

    # Parse the transform (assumes translate)
    translate_match = re.search(r'translate\(([^,]+),?([^)]+)?\)', transform)
    dx, dy = 0, 0
    if translate_match:
        dx = float(translate_match.group(1))
        dy = float(translate_match.group(2)) if translate_match.group(2) else 0

    # Apply translation to each segment in the path
    new_segments = []
    for segment in path:
        new_start = segment.start + complex(dx, dy)
        new_end = segment.end + complex(dx, dy)
        if isinstance(segment, CubicBezier):
            new_control1 = segment.control1 + complex(dx, dy)
            new_control2 = segment.control2 + complex(dx, dy)
            new_segments.append(CubicBezier(new_start, new_control1, new_control2, new_end))
        else:
            new_segments.append(type(segment)(new_start, new_end))
    return Path(*new_segments)

def visualize_strokes(input_svg):
    def cubic_bezier_point(p0, p1, p2, p3, t):
        """
        Compute a point on a cubic Bezier curve at parameter t.

        Parameters:
        - p0, p1, p2, p3: Control points of the cubic Bezier curve.
        - t: Parameter between 0 and 1.

        Returns:
        - (x, y): Point on the curve.
        """
        x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
        y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
        return x, y
    
    paths, attributes = svg2paths(input_svg)
    
    
    plt.figure(figsize=(10, 10))
    
    figure = []
    for path, attr in zip(paths, attributes):
        
        if pass_white(attr.get("fill", "#000000")):
            continue
        
        transform = attr.get("transform", None)
        transformed_path = apply_transform(path, transform)
        
        strokes = []
        for segment in transformed_path:

            p0 = np.array((segment.start.real, -segment.start.imag))
            p3 = np.array((segment.end.real, -segment.end.imag))

            num_line_segment = math.ceil(np.linalg.norm(p0-p3))
            
            if isinstance(segment, CubicBezier):
                p1 = np.array((segment.control1.real, -segment.control1.imag))
                p2 = np.array((segment.control2.real, -segment.control2.imag))

                # Sample points along the cubic Bezier curve
                sampled_points = [
                    cubic_bezier_point(p0, p1, p2, p3, t)
                    for t in np.linspace(0, 1, num_line_segment)
                ]
                strokes.append(sampled_points)

                x_vals, y_vals = zip(*sampled_points)
                plt.plot(x_vals, y_vals, color="black", linewidth=1.0)
            else:
                # For line segments
                x_vals, y_vals = [p0[0], p3[0]], [p0[1], p3[1]]
                strokes.append([(p0[0], p0[1]), (p3[0], p3[1])])

                plt.plot(x_vals, y_vals, color="black", linewidth=1.0)
        figure.append(strokes)
    
    plt.axis("off")
    plt.grid(False)
    plt.savefig(f"recon_from_{input_svg}.png")

    return figure
    # with open(f"strokes_{input_svg}.json", "w") as f:
    #     json.dump(figure, f)

def prompt_to_line_art_img(prompt, filename, pipeline_text2image):
    chat_template = f"Minimalistic line drawing of {prompt}, 'very few lines', 'no color', 'no shading', simple outlines, focused on essential shapes only, figures appearing in a clean and basic style."

    image_path = f"original_{filename}.png"

    image = pipeline_text2image(prompt=chat_template).images[0]
    image.save(image_path)
    
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    # image = cv2.imread(image_path)

    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_value = 127
    _, threshold_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    threshold_image_name = f"thresholded_{filename}.png"
    Image.fromarray(threshold_image).save(threshold_image_name)

    return threshold_image, threshold_image_name

def img_to_svg_to_stroke(filename, threshold_image_name):
    output_svg_path = f"output_{filename}.svg"

    vtracer.convert_image_to_svg_py(
        image_path=threshold_image_name,
        out_path=output_svg_path, 
        colormode="bw", 
        mode="spline", 
        color_precision=2, 
        path_precision=1, 
        splice_threshold=10,
        corner_threshold=1,
        hierarchical="cutout"
        )
        
    return visualize_strokes(output_svg_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="an apple on a stool")
    parser.add_argument("--filename", default="image")

    args = parser.parse_args()

    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")

    filename = args.filename
    prompt = args.prompt

    prompt_to_line_art_img(prompt, filename, pipeline_text2image)
    # chat_template = f"Minimalistic line drawing of {prompt}, 'very few lines', 'no color', 'no shading', simple outlines, focused on essential shapes only, figures appearing in a clean and basic style."

    # image_path = f"original_{filename}.png"

    # image = pipeline_text2image(prompt=chat_template).images[0]
    # image.save(image_path)

    # image = cv2.imread(image_path)

    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # threshold_value = 127
    # _, threshold_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    threshold_image_name = f"thresholded_{filename}.png"
    # Image.fromarray(threshold_image).save(threshold_image_name)

    output_svg_path = f"output_{filename}.svg"

    vtracer.convert_image_to_svg_py(
        image_path=threshold_image_name,
        out_path=output_svg_path, 
        colormode="bw", 
        mode="spline", 
        color_precision=2, 
        path_precision=1, 
        splice_threshold=10,
        corner_threshold=1,
        hierarchical="cutout"
        )

    visualize_strokes(output_svg_path)
