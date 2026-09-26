import json
import re
import base64
import numpy as np
from builder.build_prompt import generate_question_prompt
import cv2
from pathlib import Path
import os
from typing import List, Tuple
from tqdm import tqdm
from PIL import Image
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# defaults; override them with --tasks-root, --outputs-root, --task-type and --result-dir
TASKS_ROOT = "../tasks/z_agent/trajectory"
OUTPUTS_ROOT = "../outputs/z_agent"
CATEGORY = "trajectory"
RESULT_DIR = "eval_result"

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

def extract_four_number(reply: str):
    match = re.search(r"\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", reply)
    if match:
        x1, y1, x2, y2 = map(float, match.groups())
        # print(f"Extracted coordinates: ({x1}, {y1}, {x2}, {y2})")
        return x1, y1, x2, y2
    return 0,0,0,0

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
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)

    height, width = mask_np.shape

    if normalize:
        # (f"Normalizing bbox: {bbox}")
        # print(bbox[0] * width)
        x1 = int(bbox[0] * width)
        # print(bbox[1] * height)
        y1 = int(bbox[1] * height)
        # print(bbox[2] * width)
        x2 = int(bbox[2] * width)
        # print(bbox[3] * height)
        y2 = int(bbox[3] * height)
        # print(f"Normalized bbox coordinates: ({x1}, {y1}, {x2}, {y2})")
    else:
        x1, y1, x2, y2 = bbox

    # Clip coordinates to image boundary
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    # print(f"Calculating IOU for bbox ({x1}, {y1}, {x2}, {y2}) against mask of shape {mask_np.shape}")
    if x2 <= x1 or y2 <= y1:
        # print(f"Warning: Invalid bbox ({x1}, {y1}, {x2}, {y2}) → IOU=0.")
        return 0.0

    # Create predicted bbox mask
    pred_mask = np.zeros_like(mask_np)
    pred_mask[y1:y2+1, x1:x2+1] = 1  # +1 to include the edge

    gt_mask = (mask_np > 0).astype(np.uint8)

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        # print(f"Warning: Union is zero for mask {mask_path} at bbox ({x1}, {y1}, {x2}, {y2}).")
        return 0.0

    return intersection / union

def fix_seed():
    import torch
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
        bbox = extract_four_number(reply)
        mask_path = request.get("mask", "")
        iou = calculate_iou_bbox_vs_mask(mask_path, bbox, normalize=True)
        return iou
    
    else: # the others are all options
        # extract the option from the reply
        options = " ".join(f"{key}.{value}" for key, value in request["options"].items())
        option = gpt_extract_option(reply, options)
        if option == request["gt"]:
            return 1
        else:
            return 0

# add other tasks, like evaluate the multi-choice questions

def discover_and_run_tasks():
    """Discover all task entries in subdirectories and execute them"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    for task_category in os.listdir(TASKS_ROOT):
        # Process all entries in this category
        # for each category, maintain a json file to save the result and overall succ rate
        task_category_json = []
        category_total_question = 0
        category_correct = 0

        if CATEGORY == "bbox":
            category_iou = []

        # for each task category, we will run the evaluation
        task_category_path = os.path.join(TASKS_ROOT, task_category)
        output_category_path = os.path.join(OUTPUTS_ROOT, task_category)
        
        if not os.path.isdir(task_category_path):
            continue
            
        print(f"\nProcessing category: {task_category}")
        
        # Create output directory if needed
        os.makedirs(output_category_path, exist_ok=True)

        # for each entry in the task category
        for entry in os.listdir(task_category_path):
            category_total_question += 1
            # find the corresponding output path

            entry_path = os.path.join(task_category_path, entry)
            print(f"  Evaluating Entry: {entry_path}")

            request = json.load(open(os.path.join(entry_path, "request.json"), "r"))
            request = request.copy()

            output_entry_path = os.path.join(output_category_path, entry, "output.json")
            
            if not os.path.isdir(entry_path):
                continue
                
            # print(f"  Running Entry: {entry}")
            # print(f"  The Output Json path: {output_entry_path}")
            
            try:
                # get the last response
                output_json_content = json.load(open(output_entry_path, "r"))
                data = output_json_content[-1]
                if data.get("role") == "assistant" and data.get("content", [{}])[0].get("type") == "text":
                    # do the evaluation
                    reply = data["content"][0]["text"]
                    # evaluate the reply

                    # if there are any mask in the file
                    original_mask_path = request.get("mask", "")
                    if original_mask_path:
                        request["mask"] = os.path.join(output_category_path, entry, original_mask_path)
                    correct = evaluate_answer(reply, request, CATEGORY)
                     
                    answer = request.copy()
                    if CATEGORY == "bbox":
                        answer["iou"] = correct
                        category_iou.append(correct)
                    else:
                        answer["hit"] = int(correct)
                        if correct == 1:
                            category_correct += 1

                    answer["reply"] = reply
                    task_category_json.append(answer)

            except Exception as e:
                print(f"  ✗ Failed {entry}: {str(e)}")

        # Save the aggregated results for this category
        task_category_name = f"{task_category}_agent_output.json"
        with open(os.path.join(RESULT_DIR, task_category_name), "w") as f:
            json.dump(task_category_json, f, indent=4)

        # save the category-level success rate
        with open(os.path.join(RESULT_DIR, f"{task_category}_summary.txt"), "w") as f:
            if CATEGORY == "bbox":
                avg_iou = sum(category_iou) / len(category_iou) if category_iou else 0
                f.write(f"Average IOU for category {task_category}: {avg_iou:.4f}\n")
                print(f"Average IOU for category {task_category}: {avg_iou:.4f}\n")
            else:
                success_rate = category_correct / category_total_question if category_total_question > 0 else 0
                f.write(f"Success Rate for category {task_category}: {success_rate:.4f} ({category_correct}/{category_total_question})\n")
                print(f"Success Rate for category {task_category}: {success_rate:.4f} ({category_correct}/{category_total_question})\n")

    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score BEAR Agent outputs against request.json ground truth.")
    parser.add_argument("--tasks-root", default=TASKS_ROOT)
    parser.add_argument("--outputs-root", default=OUTPUTS_ROOT)
    parser.add_argument("--task-type", default=CATEGORY)
    parser.add_argument("--result-dir", default=RESULT_DIR)
    args = parser.parse_args()
    TASKS_ROOT, OUTPUTS_ROOT, CATEGORY, RESULT_DIR = args.tasks_root, args.outputs_root, args.task_type, args.result_dir
    discover_and_run_tasks()


