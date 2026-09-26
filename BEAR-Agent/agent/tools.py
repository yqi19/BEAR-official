from gradio_client import Client, file
from multimodal_conversable_agent import MultimodalConversableAgent
from PIL import Image, ImageDraw
import numpy as np
import tempfile
import time, os
import cv2, json, os, sys, time, random
import numpy as np
from PIL import Image
from matplotlib import colormaps
from matplotlib.colors import Normalize

import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
import ast

# Load all vision experts
from config import SOM_ADDRESS, GROUNDING_DINO_ADDRESS, DEPTH_ANYTHING_ADDRESS

# Set of Marks
som_client = Client(SOM_ADDRESS)

# Grounding DINO
gd_client = Client(GROUNDING_DINO_ADDRESS)

# DepthAnything
da_client = Client(DEPTH_ANYTHING_ADDRESS)

class AnnotatedImage:
    # A class to represent an annotated image. It contains the annotated image and the original image.
    
    def __init__(self, annotated_image: Image.Image, original_image: Image.Image=None):
        self.annotated_image = annotated_image
        self.original_image = original_image

# def segment_and_mark(image, granularity:float = 1.8, alpha:float = 0.1, anno_mode:list = ['Mask', 'Mark']):
#     """Use a segmentation model to segment the image, and add colorful masks on the segmented objects. Each segment is also labeled with a number.
#     The annotated image is returned along with the bounding boxes of the segmented objects.
#     This tool may help you to better reason about the relationship between objects, which can be useful for spatial reasoning etc.

#     Args:
#         image (PIL.Image.Image): the input image
#         granularity (float, optional): The granlarity of the segmentation. Rranges from 0 to 2.5. The higher the more fine-grained. Defaults to 1.8.
#         alpha (float, optional): The alpha of the added colorful masks. Defaults to 0.1.
#         anno_mode (list, optional): What annotation is added on the input image. Mask is the colorful masks. And mark is the number labels. Defaults to ['Mask', 'Mark'].

#     Returns:
#         output_image (AnnotatedImage): the original image annotated with colorful masks and number labels. Each mask is labeled with a number. The number label starts at 1.
#         bboxes (List): listthe bounding boxes of the masks.The order of the boxes is the same as the order of the number labels.
        
#     Example:
#         User request: I want to find a seat close to windows, where should I sit?
#         Code:
#         ```python
#         image = Image.open("sample_img.jpg")
#         output_image, bboxes = segment_and_mark(image)
#         display(output_image.annotated_image)
#         ```
#         Model reply: You can sit on the chair numbered as 5, which is close to the window.
#         User: Give me the bounding box of that chair.
#         Code:
#         ```python
#         print(bboxes[4]) # [0.24, 0.21, 0.3, 0.4]
#         ```
#         Model reply: The bounding box of the chair numbered as 5 is [0.24, 0.21, 0.3, 0.4].
#     """
    
    
#     with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
#         image.save(tmp_file.name, 'JPEG')
#         image = tmp_file.name

#         outputs = som_client.predict(file(image), granularity, alpha, "Number", anno_mode)

#         original_image = Image.open(image)
#         output_image = Image.open(outputs[0])
        
#         output_image = AnnotatedImage(output_image, original_image)
        
#         w,h = output_image.annotated_image.size
                
#         masks = outputs[1]
        
#         bboxes = []
        
#         for mask in masks:
#             bbox = mask['bbox']
#             bboxes.append((bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h))
        
#     return output_image, bboxes


# def detection(image, objects, box_threshold:float = 0.35, text_threshold:float = 0.25):
#     """Object detection using Grounding DINO model. It returns the annotated image and the bounding boxes of the detected objects.
#     The text can be simple noun, or simple phrase (e.g., 'bus', 'red car'). Cannot be too hard or the model will break.
#     The detector is not perfect, it may wrongly detect objects or miss some objects.
#     Also, notice that the bounding box label might be out of the image boundary.
#     You should use the output as a reference, not as a ground truth.

#     Args:
#         image (PIL.Image.Image): the input image
#         objects (List[str]): a list of objects to detect. Each object should be a simple noun or a simple phrase.

#     Returns:
#         output_image (AnnotatedImage): the original image, annotated with bounding boxes. Each box is labeled with the detected object, and an index.
#         processed boxes (List): list the bounding boxes of the detected objects
    
#     Example:
#         image = Image.open("sample_img.jpg")
#         output_image, boxes = detection(image, ["bus"])
#         display(output_image.annotated_image)
#         print(boxes) # [(0.24, 0.21, 0.3, 0.4), (0.6, 0.3, 0.2, 0.3)]
#     """
    
#     with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
#         image.save(tmp_file.name, 'JPEG')
#         image = tmp_file.name
    
#         outputs = gd_client.predict(file(image), ', '.join(objects), box_threshold, text_threshold)
        
#         # process images
#         original_image = Image.open(image)
#         output_image = Image.open(outputs[0])
#         output_image = AnnotatedImage(output_image, original_image)
        
#         # process boxes
#         boxes = outputs[1]['boxes']
#         processed_boxes = []
        
#         cleaned = boxes.replace('tensor(', '').rstrip(')')
#         box_list = ast.literal_eval(cleaned)
#         for box in box_list:
#             processed_boxes.append((box[0]-box[2]/2, box[1] - box[3]/2, box[2], box[3]))
#             print("possible bbox:", processed_boxes[-1])
        
#     return output_image, processed_boxes


def depth(image):
    """Depth estimation using DepthAnything model. It returns the depth map of the input image. 
    A colormap is used to represent the depth. It uses Inferno colormap. The closer the object, the warmer the color.
    This tool may help you to better reason about the spatial relationship, like which object is closer to the camera.
    Depth can also help you to understand the 3D structure of the scene, which can be useful for analyzing the motion between frames, etc.

    Args:
        image (PIL.Image.Image): the input image

    Returns:
        output_image (PIL.Image.Image): the depth map of the input image
        
    Example:
        image = Image.open("sample_img.jpg")S
        output_image = depth(image)
        display(output_image)
    """
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        image.save(tmp_file.name, 'JPEG')
        image = tmp_file.name
        outputs = da_client.predict(file(image))
    output_image = Image.open(outputs)
    
    return output_image


def crop_image(image, x:float, y:float, width:float, height:float):
    """Crop the image based on the normalized coordinates.
    Return the cropped image.
    This has the effect of zooming in on the image crop.

    Args:
        image (PIL.Image.Image): the input image
        x (float): the horizontal coordinate of the upper-left corner of the box
        y (float): the vertical coordinate of that corner
        width (float): the box width
        height (float): the box height

    Returns:
        cropped_img (PIL.Image.Image): the cropped image
        
    Example:
        image = Image.open("sample_img.jpg")
        cropped_img = crop_image(image, 0.2, 0.3, 0.5, 0.4)
        display(cropped_img)
    """
    
    # get height and width of image
    w, h = image.size
    
    # limit the range of x and y
    x = min(max(0, x), 1)
    y = min(max(0, y), 1)
    x2 = min(max(0, x+width), 1)
    y2 = min(max(0, y+height), 1)
    
    cropped_img = image.crop((x*w, y*h, x2*w, y2*h))
    return cropped_img


def zoom_in_image_by_bbox(image, box, padding=0.05):
    """A simple wrapper function to crop the image based on the bounding box.
    The zoom factor cannot be too small. Minimum is 0.1

    Args:
        image (PIL.Image.Image): the input image
        box (List[float]): the bounding box in the format of [x, y, w, h]
        padding (float, optional): The padding for the image crop, outside of the bounding box. Defaults to 0.05.

    Returns:
        cropped_img (PIL.Image.Image): the cropped image
        
    Example:
        image = Image.open("sample_img.jpg")
        annotated_img, boxes = detection(image, "bus")
        cropped_img = zoom_in_image_by_bbox(image, boxes[0], padding=0.1)
        display(cropped_img)
    """
    assert padding >= 0.05, "The padding should be at least 0.05"
    x, y, w, h = box
    x, y, w, h = x-padding, y-padding, w+2*padding, h+2*padding
    return crop_image(image, x, y, w, h)
        

# def sliding_window_detection(image: Image.Image, objects):
#     """Deal with the case when the user query is asking about objects that are not seen by the model.
#     In that case, the most common reason is that the object is too small such that both the vision-language model and the object detection model fail to detect it.
#     This function tries to detect the object by sliding window search.
#     With the help of the detection model, it tries to detect the object in the zoomed-in patches.
#     The function returns a list of annotated images that may contain at leas one of the objects, annotated with bounding boxes.
#     It also returns a list of a list of bounding boxes of the detected objects.

#     Args:
#         image (PIL.Image.Image): the input image
#         objects (List[str]): a list of objects to detect. Each object should be a simple noun or a simple phrase.
        
#     Returns:
#         possible_patches (List[AnnotatedImage]): a list of annotated zoomed-in images that may contain the object, annotated with bounding boxes.
#         possible_boxes (List[List[List[Float]]]): For each image in possible_patches, a list of bounding boxes of the detected objects. 
#             The coordinates are w.r.t. each zoomed-in image. The order of the boxes is the same as the order of the images in possible_patches.
            
#     Example:
#         image = Image.open("sample_img.jpg")
#         possible_patches, possible_boxes = search_object_and_zoom(image, ["bird", "sign"])
#         for i, patch in enumerate(possible_patches):
#             print(f"Patch {i}:")
#             display(patch.annotated_image)
        
#         # print the bounding boxes of the detected objects in the first patch
#         print(possible_boxes[0]) # [[0.24, 0.21, 0.3, 0.4], [0.6, 0.3, 0.2, 0.3]]
#     """
    
#     def check_if_box_margin(box, margin=0.005):
#         x_margin = min(box[0], 1-box[0]-box[2])
#         y_margin = min(box[1], 1-box[1]-box[3])
#         return x_margin < margin or y_margin < margin
    
#     # # first try to detect the object
#     # annotated_img, detection_boxes = detection(image, text)
    
#     # if len(detection_boxes) != 0:
#     #     return [annotated_img], [detection_boxes]
    
#     # if not detected, do sliding window search
#     box_width = 1/3
#     box_height = 1/3

#     possible_patches = []
#     possible_boxes = []
    
#     for x in np.arange(0, 7/9, 2/9):
#         for y in np.arange(0, 7/9, 2/9):
#             cropped_img = crop_image(image, x, y, box_width, box_height)
#             annotated_img, detection_boxes= detection(cropped_img, objects)
            
#             # if one of the boxes is not too close to the edge, save it
#             margin_flag = True
#             for box in detection_boxes:
#                 if not check_if_box_margin(box):
#                     margin_flag = False
#                     break
            
#             # if the object is detected and the box is not too close to the edge
#             if len(detection_boxes) != 0 and not margin_flag:
#                 possible_patches.append(annotated_img)
#                 possible_boxes.append(detection_boxes)

#     return possible_patches, possible_boxes


def overlay_images(background_img, overlay_img, alpha=0.3, bounding_box=[0, 0, 1, 1]):
    """
    Overlay an image onto another image with transparency.
    This is particularly useful visualizing heatmap while preserving some info from the original image.
    For example, you can overlay a segmented image on a heatmap to better understand the spatial relationship between objects.
    It will also help seeing the labels, circles on the original image that may not be visible on the heatmap.

    Args:
    background_img_pil (PIL.Image.Image): The background image in PIL format.
    overlay_img_pil (PIL.Image.Image): The image to overlay in PIL format.
    alpha (float): Transparency of the overlay image.
    bounding_box (List[float]): The bounding box of the overlay image. The format is [x, y, w, h]. The coordinates are normalized to the background image. Defaults to [0, 0, 1, 1].

    Returns:
    PIL.Image.Image: The resulting image after overlay, in PIL format.
    s
    Example:
        image = Image.open('original.jpg')
        depth_map = depth(image)
        overlayed_image = overlay_images(depth_map, image, alpha=0.3)
        display(overlayed_image)
    """
    # Calculate the actual pixel coordinates of the bounding box
    bg_width, bg_height = background_img.size
    x = int(bounding_box[0] * bg_width)
    y = int(bounding_box[1] * bg_height)
    w = int(bounding_box[2] * bg_width)
    h = int(bounding_box[3] * bg_height)

    # Resize overlay image to the bounding box size
    overlay_resized = overlay_img.resize((w, h), Image.Resampling.LANCZOS)

    # Adjust the overlay image's transparency
    overlay_with_alpha = overlay_resized.copy()
    overlay_with_alpha.putalpha(int(255 * alpha))  # Set the transparency level

    # Create a new image for the result and copy the background image to it
    new_img = Image.new('RGBA', background_img.size, (255, 255, 255, 255))
    new_img.paste(background_img, (0,0))

    # Paste the overlay image onto the new image with transparency
    new_img.paste(overlay_with_alpha, (x, y, x + w, y + h), overlay_with_alpha)

    return new_img.convert('RGB')  # Convert back to RGB if needed

# [] debug this function, change the output json file
def extend_arrow_color(img, color="red", thickness=8, debug=False, extension_length=600):

    if debug:
        # save the img
        img.save(f"original_image.jpg")
        print("Original image saved as original_image.jpg")

    def pil_to_cv2(pil_image):
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Convert OpenCV (numpy) format to PIL Image
    def cv2_to_pil(cv2_image):
        """Convert OpenCV (RGB) to PIL (RGB) - No BGR→RGB conversion needed"""
        return Image.fromarray(cv2_image)

    # read the image
    img = pil_to_cv2(img)  # Convert PIL Image to OpenCV format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get the color range
    color_ranges = {
    "red":    (np.array([210, 0, 0]),   np.array([255, 50, 50])),
    "blue":   (np.array([0, 100, 200]), np.array([50, 180, 255])),
    "green":  (np.array([0, 200, 0]),   np.array([80, 255, 80])),
    "yellow": (np.array([200, 200, 0]), np.array([255, 255, 100]))
    }

    color_ranges_lower, color_ranges_upper = color_ranges[color.lower()]
    color_mask = cv2.inRange(img_rgb, color_ranges_lower, color_ranges_upper)

    # Apply preprocessing
    def preprocess(mask):
        img_blur = cv2.GaussianBlur(mask, (5, 5), 1)
        img_canny = cv2.Canny(img_blur, 50, 50)
        kernel = np.ones((3, 3), np.uint8)
        img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
        img_erode = cv2.erode(img_dilate, kernel, iterations=1)
        return img_erode
    
    if debug:
        # save the color mask for debugging
        color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f"{color}_arrow_mask.jpg", color_mask_bgr)

    preprocessed = preprocess(color_mask)
    # Find contours
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create a canvas to draw
    canvas = img_rgb.copy()

    if contours:
        # Use the largest contour
        largest = max(contours, key=cv2.contourArea)
        points = largest[:, 0, :]  # Shape: (N, 2), x-y
        # Convert to y-x for consistency
        indices = np.array([[pt[1], pt[0]] for pt in points])
        if len(indices) >= 2:
            # Get furthest points
            dist = distance.cdist(indices, indices, 'euclidean')
            far_points_index = np.unravel_index(np.argmax(dist), dist.shape)
            pt1 = indices[far_points_index[0]]  # [y, x]
            pt2 = indices[far_points_index[1]]  # [y, x]

            # among pt1 and pt2, the one close to centroid is the tip
            centroid = np.mean(indices, axis=0)
            if np.linalg.norm(pt1 - centroid) >= np.linalg.norm(pt2 - centroid):
                pt1, pt2 = pt2, pt1
            # pt1 is the tip, pt2 is the tail

            # Calculate angle
            if pt2[1] != pt1[1]:
                slope = (pt2[0] - pt1[0]) / (pt2[1] - pt1[1])
                angle = math.degrees(math.atan(slope))
            else:
                angle = 90.0 if pt2[0] > pt1[0] else -90.0

            # Length
            arrow_length = np.linalg.norm(pt1 - pt2)

            # Thickness
            x = np.linspace(pt1[1], pt2[1], 20)
            y = np.linspace(pt1[0], pt2[0], 20)
            line = np.array([[int(round(yy)), int(round(xx))] for yy, xx in zip(y, x)])
            thickness_dists = np.amin(distance.cdist(line, indices), axis=0)
            n, bins, _ = plt.hist(thickness_dists, bins=150)
            thickness = 2 * bins[np.argmax(n)]

            # Print for debug
            if debug:
                print(f"Arrow (from contour):")
                print(f"  Length   : {arrow_length:.2f}")
                print(f"  Angle    : {angle:.2f} degrees")
                print(f"  Thickness: {thickness:.2f}\n")

                cv2.circle(canvas, (pt1[1], pt1[0]), 6, (255, 0, 0), -1)  # Tip
                cv2.circle(canvas, (pt2[1], pt2[0]), 6, (0, 0, 255), -1)  # Tail
                cv2.line(canvas, (pt1[1], pt1[0]), (pt2[1], pt2[0]), (255, 255, 0), 2)

            # Extend the arrow by extend with the dashed line
            # ===== Draw a dashed extension of the arrow direction =====

            # Direction vector from tail to tip
            vec = pt1 - pt2
            vec = vec / np.linalg.norm(vec)  # Normalize

            # How far to extend (in pixels)

            # Start and end points of the extension
            start_point = pt1
            end_point = (pt1[0] + int(vec[0] * extension_length), pt1[1] + int(vec[1] * extension_length))

            # Dashed line drawing
            def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
                dist = np.linalg.norm(np.array(pt2) - np.array(pt1))
                num_dashes = int(dist // (2 * dash_length))
                for i in range(num_dashes):
                    start_frac = i * 2 * dash_length / dist
                    end_frac = (i * 2 + 1) * dash_length / dist
                    x1 = int(pt1[1] + (pt2[1] - pt1[1]) * start_frac)
                    y1 = int(pt1[0] + (pt2[0] - pt1[0]) * start_frac)
                    x2 = int(pt1[1] + (pt2[1] - pt1[1]) * end_frac)
                    y2 = int(pt1[0] + (pt2[0] - pt1[0]) * end_frac)
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

            # Draw dashed yellow line
            draw_dashed_line(canvas, start_point, end_point, (255, 255, 0), thickness=4)

            # save as PIL file
            canvas = cv2_to_pil(canvas)

            # save PIL file for debugging
            if debug:
                canvas.save(f"arrow_analysis_from_contour.jpg")
                print("Arrow analysis saved as arrow_analysis_from_contour.jpg")

            # so the output is a PIL file, but it seems it cannot display?
            
            return canvas
        
def read_scene_graph():
    with open('scene_graph.json', 'r') as f:
        scene_graph = json.load(f)
    return scene_graph

def write_scene_graph(scene_graph):
    with open('scene_graph.json', 'w') as f:
        json.dump(scene_graph, f)