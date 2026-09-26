from vlmeval.config import supported_VLM
import json
import re
import base64
import numpy as np
import torch
import time
import colorama
from builder.build_prompt import generate_question_prompt
import cv2
from pathlib import Path
import os
from typing import List, Tuple

SAMPLE_FRAMES = 16

# save parallel
def generate_input_images(mp4_path, output_dir, num_frames=SAMPLE_FRAMES):
    """
    Extracts `num_frames` evenly spaced frames from a video (including first and last),
    and saves them to output_dir as images.

    Args:
        mp4_path (str): Path to the .mp4 video file.
        output_dir (str): Directory to save output images.
        num_frames (int): Number of frames to extract (default: 10).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {mp4_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    
    saved_count = 0
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = cap.read()
        if not success:
            print(f"Warning: failed to read frame {i}")
            continue

        img_name = output_dir / f"{saved_count:04d}.jpg"
        cv2.imwrite(str(img_name), frame)
        saved_count += 1

    cap.release()
    print(f"Saved {saved_count} frames to {output_dir}")

def fix_seed():
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

def evaluate_baseline_agent(entry_path, category_name):
    
    # for each entry, result the evaluation result for that entry
    fix_seed()
        
    # json load the input data
    input_json = os.path.join(entry_path, "request.json")
    # read the json file
    input_json = json.load(open(input_json))
    # copy the json file
    output_json = input_json.copy()

    question = input_json["question"]
    options = input_json.get("options", "")

    # usage:
    # the pointing and trajectory is saved in images[] (this is a list contains image paths)
    # the interleaved category and video category is saved in videos[], also a list
    image_list = input_json.get("images", None)
    video_list = input_json.get("video", None)

    print("entry_path:", entry_path)
    # for image input data
    if category_name in ["pointing", "trajectory", "bbox"]:
        for format in ["direct"]:

            question_prompt = generate_question_prompt(format, category_name, question, options, model_category="general")
            input_content = [{"image": os.path.join(entry_path, image_list[0])}, {"text":question_prompt}]

            print("Input content for model:", input_content)
            from builder.gpt import generate_content
            response = generate_content(model_name, input_content)
            # change the mask path
            if category_name in ["pointing", "bbox"]:
                output_json["mask"] = os.path.join(entry_path, input_json["mask"])
            output_json[f"{format}_reply"] = response

    if category_name == "path planning" or category_name == "relative direction":
        image  = input_json.get("images", None)
        video = input_json.get("video", None)
        output_images_dir = "./extracted_images"
        image_path = os.path.join(entry_path, image)
        video_path = os.path.join(entry_path, video)
        generate_input_images(video_path, output_images_dir, num_frames=SAMPLE_FRAMES)

        # load the images
        input_images = [{"image": os.path.join(output_images_dir, f"{i:04d}.jpg")} for i in range(SAMPLE_FRAMES)]

        for format in ["direct"]:
            (first_prompt, second_prompt, third_prompt) = generate_question_prompt(format, category_name, question, options, model_category="general")
            input_content = [{"text": first_prompt}] + input_images + [{"text": second_prompt}] + [{"image": image_path}] + [{"text": third_prompt}]

            print("Input content for model:", input_content)

            from builder.gpt import generate_content
            response = generate_content(model_name, input_content)
            output_json[f"{format}_reply"] = response

    # for the video data:
    if category_name == "task progress reasoning" or category_name == "next action prediction" or category_name == "object localization":
        for format in ["direct"]:
            video_name = input_json.get("video", None)
            video_path = os.path.join(entry_path, video_name)
            output_images_dir = "./extracted_images"
            generate_input_images(video_path, output_images_dir, num_frames=SAMPLE_FRAMES)

            input_images = [{"image": os.path.join(output_images_dir, f"{i:04d}.jpg")} for i in range(SAMPLE_FRAMES)]

            (first_prompt, second_prompt) = generate_question_prompt(format, category_name, question, options, model_category="general")
            input_content = [{"text": first_prompt}] + input_images + [{"text": second_prompt}]

            print("Input content for model:", input_content)
            from builder.gpt import generate_content
            response = generate_content(model_name, input_content)
            output_json[f"{format}_reply"] = response

    return output_json

# Base paths
# for each of the subcategory do the evaluation
# Change this if you would like to evaluate other tasks
TASKS_ROOT = "../tasks/z_agent/planning"
OUTPUTS_ROOT = "../outputs/z_agent/planning"
# You should also modify the task_category

def discover_and_run_tasks(model_name):
    """Discover all task entries in subdirectories and execute them"""
    for task_category in os.listdir(TASKS_ROOT):

        task_category_path = os.path.join(TASKS_ROOT, task_category)
        output_category_path = os.path.join(OUTPUTS_ROOT, task_category)
        
        if not os.path.isdir(task_category_path):
            continue
            
        print(f"\nProcessing category: {task_category}")
        
        os.makedirs(output_category_path, exist_ok=True)
        
        task_category_json = []
        for entry in os.listdir(task_category_path):
            entry_path = os.path.join(task_category_path, entry)
            
            if not os.path.isdir(entry_path):
                continue
                
            print(f"  Running entry: {entry}")
            
            try:
                # change the category of the task !!!
                entry_output_json = evaluate_baseline_agent(entry_path, "next action prediction")
                task_category_json.append(entry_output_json)
                print(f"  ✓ Completed: {entry}")
            except Exception as e:
                print(f"  ✗ Failed {entry}: {str(e)}")

        # Save the aggregated results for this category
        # Save all the results in eval_result folder
        task_category_name = f"{task_category}_gpt_5_output.json"
        with open(os.path.join("eval_result", task_category_name), "w") as f:
            json.dump(task_category_json, f, indent=4)


if __name__ == "__main__":

    model_name = "gpt-5"

    # Option 1: Directly execute all tasks
    discover_and_run_tasks(model_name)
