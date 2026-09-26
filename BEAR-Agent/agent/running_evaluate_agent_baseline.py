from pathlib import Path
import os
from typing import List, Tuple
from tqdm import tqdm
from PIL import Image
from openai import OpenAI
import json
import re
import numpy as np

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def extract_xy(reply: str):
    """
    Extract floating-point (x, y) coordinates from a reply string.
    
    Args:
        reply: String containing coordinate information
        
    Returns:
        Tuple[float, float] or None: (x, y) coordinates if found, else None
    """
    # Pattern for (x, y) where x and y can be integers or decimals
    match = re.search(r"\(([\d.]+),\s*([\d.]+)\)", reply)
    if match:
        try:
            x = float(match.group(1))
            y = float(match.group(2))
            return x, y
        except ValueError:
            pass
    return None

def check_point_in_mask(mask_path, xy, normalize=True):
    try:
        if normalize==False:
            """Return True if the (x,y) falls on the mask (white area), else False"""
            mask = Image.open(mask_path).convert("L")
            mask_np = np.array(mask)
                
            x, y = xy
            # Check boundary
            if x < 0 or x >= mask_np.shape[1] or y < 0 or y >= mask_np.shape[0]:
                return False

            return mask_np[y, x] > 0  # mask is binary: white is 255
        else:
            """Return True if the (x,y) falls on the mask (white area), else False"""
            mask = Image.open(mask_path).convert("L")
            mask_np = np.array(mask)

            x, y = xy
            # Normalize x and y to pixel coordinates
            x = int(x * mask_np.shape[1])
            y = int(y * mask_np.shape[0])

            # Check boundary
            if x < 0 or x >= mask_np.shape[1] or y < 0 or y >= mask_np.shape[0]:
                return False

            return mask_np[y, x] > 0
    except Exception as e:
        return 0

def extract_four_number(reply: str):
    match = re.search(r"\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", reply)
    if match:
        x1, y1, x2, y2 = map(float, match.groups())
        print(f"Extracted coordinates: ({x1}, {y1}, {x2}, {y2})")
        return x1, y1, x2, y2
    return 0, 0, 0, 0

def calculate_iou_bbox_vs_mask(mask_path, bbox, normalize=True):
    """
    Calculate IOU between a predicted bbox and a mask.

    Args:
        mask_path: Path to the GT mask.
        bbox: (x1, y1, x2, y2), normalized [0-1] if normalize=True.
        normalize: Whether the bbox is normalized or in absolute coords.

    Returns:
        IOU between the bbox and the GT mask.
    """
    print("mask_path is", mask_path)
    print("bbox is", bbox)
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)
    print("mask_np shape is", mask_np.shape)

    height, width = mask_np.shape

    if normalize:
        print(f"Normalizing bbox: {bbox}")
        print(bbox[0] * width)
        x1 = int(bbox[0] * width)
        print(bbox[1] * height)
        y1 = int(bbox[1] * height)
        print(bbox[2] * width)
        x2 = int(bbox[2] * width)
        print(bbox[3] * height)
        y2 = int(bbox[3] * height)
        print(f"Normalized bbox coordinates: ({x1}, {y1}, {x2}, {y2})")
    else:
        x1, y1, x2, y2 = bbox

    # Clip coordinates to image boundary
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    print(f"Calculating IOU for bbox ({x1}, {y1}, {x2}, {y2}) against mask of shape {mask_np.shape}")
    if x2 <= x1 or y2 <= y1:
        print(f"Warning: Invalid bbox ({x1}, {y1}, {x2}, {y2}) → IOU=0.")
        return 0.0

    # Create predicted bbox mask
    pred_mask = np.zeros_like(mask_np)
    pred_mask[y1:y2+1, x1:x2+1] = 1  # +1 to include the edge

    gt_mask = (mask_np > 0).astype(np.uint8)

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        print(f"Warning: Union is zero for mask {mask_path} at bbox ({x1}, {y1}, {x2}, {y2}).")
        return 0.0

    return intersection / union

def fix_seed():
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

def gpt_extract_option(reply, options):
    if not reply or not isinstance(reply, str):
        return None

    messages = [
        {"role": "system", "content": "You are an assistant that extracts the chosen option from the model's reply."},
        {"role": "user", "content": "Input is the answer from a VLM, you should return only the selected option as a single uppercase letter: A, B, C, or D. No explanation. The input is: " + reply + "The original options of the questions are: " + options}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        )
        answer = response.choices[0].message.content.strip().upper()
        match = re.match(r"^[ABCD]$", answer)
        if match:
            return answer
    except Exception as e:
        return "error"
    return "error"

def evaluate_answer(reply, request, task_type):
    if task_type == "pointing":
        point = extract_xy(reply)
        mask_path = request.get("mask", "")
        correctness = check_point_in_mask(mask_path, point, normalize=True)
        return correctness

    elif task_type == "bbox":
        try:
            bbox = extract_four_number(reply)
            print("extracted bbox is", bbox)
        except Exception as e:
            bbox = (0, 0, 0, 0)
        print("bbox is", bbox)
        mask_path = request.get("mask", "")
        iou = calculate_iou_bbox_vs_mask(mask_path, bbox, normalize=True)
        return iou
    
    else: # the others are all options
        # extract the option from the reply
        options = "A." + request["options"]["A"] + " B." + request["options"]["B"] + " C." + request["options"]["C"] + " D." + request["options"]["D"]
        option = gpt_extract_option(reply, options)
        if option == request["gt"]:
            return 1
        else:
            return 0

def run_evaluation(task_category: str, task_category_json: str = None):
    results = []

    # Load task category JSON
    with open(task_category_json, "r") as f:
        task_category_json = json.load(f)

    # evaluate direct and cot replies
    eval_result = []

    total_count = 0
    direct_correctness = 0
    cot_correctness = 0

    if task_category == "bbox":
        cot_iou = []
        direct_iou = []

    for item in tqdm(task_category_json, desc=f"Evaluating {task_category}"):
        total_count += 1
        copy_item = item.copy()
        for key in ["direct_reply"]:
            reply = item.get(key, "")
            correctness = evaluate_answer(reply, item, task_category)

            if task_category == "bbox":
                if key == "direct_reply":
                    copy_item["direct_iou"] = correctness
                    direct_iou.append(correctness)
                elif key == "cot_reply":
                    copy_item["cot_iou"] = correctness
                    cot_iou.append(correctness)
            else:
                if key == "direct_reply":
                    direct_correctness += int(correctness)
                    copy_item["direct_hit"] = int(correctness)
                elif key == "cot_reply":
                    cot_correctness += int(correctness)
                    copy_item["cot_hit"] = int(correctness)
        eval_result.append(copy_item)

    # save the direct result and cot result

    # change the evaluation output
    output_json_file_path = "eval_result/" + "hand_trajectory_result.json" 
    with open(output_json_file_path, "w") as f:
        json.dump(eval_result, f, indent=4)

    # save the success rate
    if task_category == "bbox":
        direct_iou_mean = np.mean(direct_iou)
        cot_iou_mean = np.mean(cot_iou)
        print(f"Direct IOU: {direct_iou_mean:.4f}, COT IOU: {cot_iou_mean:.4f}")

        # write it as a .txt file
        with open(f"{task_category}_iou.txt", "w") as f:
            f.write(f"Direct IOU: {direct_iou_mean:.4f}\n")
            f.write(f"COT IOU: {cot_iou_mean:.4f}\n")
    else:
        direct_success_rate = direct_correctness / total_count
        cot_success_rate = cot_correctness / total_count
        print(f"Direct Success Rate: {direct_success_rate:.4f}, COT Success Rate: {cot_success_rate:.4f}")

        # The final success rate will be written as a .txt file
        with open(f"{task_category}_success_rate.txt", "w") as f:
            f.write(f"Direct Success Rate: {direct_success_rate:.4f}\n")
            f.write(f"COT Success Rate: {cot_success_rate:.4f}\n")
            print(f"Saved evaluation results to {output_json_file_path}")

# change the input category and input task json file
run_evaluation("pointing", os.path.join("eval_result", "unseen_agent_gpt_5_output.json"))

# GPT-4o's result:
# 40%: for human hands
# 55%: for gripper
# 40%: for object

# GPT-5's result:
# 85% for gripper
# 70% for human hand
# 55% for object
