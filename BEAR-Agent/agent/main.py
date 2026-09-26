import json
import os
import shutil
from builder.build_prompt_new import generate_question_prompt

from agent import SketchpadUserAgent
from multimodal_conversable_agent import MultimodalConversableAgent
from prompt import (
    MathPrompt, GeoPrompt, PerceptionPrompt, TrajPrompt, ObjTrajPrompt,
    ObjectLocalizationPrompt, RelativeDirectionPrompt, PlanningPrompt,
    python_codes_for_images_reading, MULTIMODAL_ASSISTANT_MESSAGE,
)
from parse import Parser
from execution import CodeExecutor
from utils import custom_encoder
from config import MAX_REPLY, llm_config
import cv2
import numpy as np

SUPPORTED_TASK_TYPES = (
    "pointing", "bbox", "trajectory", "object trajectory", "object localization",
    "relative direction", "path planning", "next action prediction",
    "task progress reasoning", "math", "geo",
)

def sample_n_frames(video_path, output_dir, n):
    """
    Uniformly sample `n` frames from a video, save them as JPGs, and return their filenames.

    Args:
        video_path (str): Path to the video (.mp4)
        output_dir (str): Directory to save sampled images
        n (int): Number of frames to sample

    Returns:
        List[str]: List of saved image file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print("⚠️ No frames found in video.")
        return []

    frame_indices = np.linspace(0, total_frames - 1, n, dtype=int)
    frame_id = 0
    saved_paths = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in frame_indices:
            idx = list(frame_indices).index(frame_id) + 1
            save_path = f"image_{idx}.jpg"
            cv2.imwrite(os.path.join(output_dir, save_path), frame)
            saved_paths.append(save_path)

        frame_id += 1
        if len(saved_paths) >= n:
            break

    cap.release()
    return saved_paths

def checks_terminate_message(msg):
    if isinstance(msg, str):
        return msg.find("TERMINATE") > -1
    elif isinstance(msg, dict) and 'content' in msg:
        return msg['content'].find("TERMINATE") > -1
    else:
        print(type(msg), msg)
        raise NotImplementedError

def _prepare_task(task_input, task_type, task_name):
    """Build task inputs without starting agents or execution services."""
    metadata_file = "example.json" if task_type in {"math", "geo"} else "request.json"
    with open(os.path.join(task_input, metadata_file)) as handle:
        metadata = json.load(handle)

    if task_type == "math":
        return metadata, [], MathPrompt(task_name)
    if task_type == "geo":
        return metadata, [], GeoPrompt()

    image_task = task_type in {"pointing", "bbox", "trajectory", "object trajectory"}
    question_prompt = generate_question_prompt(
        format="direct",
        category=metadata.get("category", "") if image_task else task_type,
        original_question=metadata.get("question", ""),
        original_options=metadata.get("options", []),
        model_category="general",
    )
    prompt_classes = {
        "pointing": PerceptionPrompt,
        "bbox": PerceptionPrompt,
        "trajectory": TrajPrompt,
        "object trajectory": ObjTrajPrompt,
        "object localization": ObjectLocalizationPrompt,
        "relative direction": RelativeDirectionPrompt,
        "path planning": PlanningPrompt,
        "next action prediction": PlanningPrompt,
        "task progress reasoning": PlanningPrompt,
    }
    prompt_generator = prompt_classes[task_type]()
    if image_task:
        images = [os.path.join(task_input, name) for name in metadata["images"]]
        return question_prompt, images, prompt_generator

    image_names = sample_n_frames(os.path.join(task_input, metadata["video"]), task_input, 16)
    images = [os.path.join(task_input, name) for name in image_names]
    if task_type in {"relative direction", "path planning"}:
        _, _, question = question_prompt
        owner = "my" if task_type == "relative direction" else "your"
        query = (
            f"Given the history video and {owner} current observation image, answer the following questions. "
            f"According to the history video and {owner} current observation, " + question
            + "\nThe video is represented as a temporally ordered sequence of images, "
            f"and {owner} current observation image is provided at the end."
        )
        images.append(os.path.join(task_input, metadata["images"]))
    else:
        question, options = question_prompt
        query = question + "\n" + options + "\nThe video is represented as a temporally ordered sequence of images, provided below."
    return query, images, prompt_generator


def run_agent(task_input, output_dir, task_type="vision", task_name=None):
    """Run the Visual Sketchpad agent on one task instance.

    Args:
        task_input (str): a path to the task input directory
        output_dir (str): a path to the directory where the output will be saved
        task_type (str): Task type from SUPPORTED_TASK_TYPES. Pass it explicitly for BEAR tasks.
        task_name (str, optional): Only needed for math tasks. Defaults to None.
    """

    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"Unsupported task type {task_type!r}; choose from {SUPPORTED_TASK_TYPES}")

    # create a directory for the task
    task_input = task_input.rstrip('/')
    task_directory = os.path.join(output_dir, os.path.basename(task_input))

    # copy the task input to the output directory
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(task_input, task_directory, dirs_exist_ok=True)


    query, images, prompt_generator = _prepare_task(task_input, task_type, task_name)
    use_vision_tools = task_type not in {"math", "geo"}
    if use_vision_tools:
        try:
            from tools import som_client, gd_client, da_client
        except ImportError as exc:
            raise ImportError("Vision tools are not loaded. Please install vision_experts.") from exc

    parser = Parser()
    executor = CodeExecutor(working_dir=task_directory, use_vision_tools=use_vision_tools)
    user = planner = None
    try:
        if use_vision_tools:
            print(f"Query: {query}")
            print(f"Images: {images}")
            image_loading_result = executor.execute(python_codes_for_images_reading(images))
            if image_loading_result[0] != 0:
                raise RuntimeError(f"Error loading images: {image_loading_result[1]}")

        user = SketchpadUserAgent(
            name="multimodal_user_agent",
            human_input_mode='NEVER',
            max_consecutive_auto_reply=MAX_REPLY,
            is_termination_msg=checks_terminate_message,
            prompt_generator = prompt_generator,
            parser = parser,
            executor = executor
        )

        # running the planning experiment
        all_messages = {}

        planner = MultimodalConversableAgent(
            name="planner",
            human_input_mode='NEVER',
            max_consecutive_auto_reply=MAX_REPLY,
            is_termination_msg = lambda x: False,
            system_message=MULTIMODAL_ASSISTANT_MESSAGE,
            llm_config=llm_config
        )

        # run the agent
        try:
            user.initiate_chat(
                planner,
                n_image=len(images),
                task_id = "testing_case",
                message = query,
                log_prompt_only = False,
            )
            all_messages = planner.chat_messages[user]

        except Exception as e:
            print(e)
            all_messages = {'error': e.message if hasattr(e, 'message') else f"{e}"}


        # save the results
        with open(os.path.join(task_directory, "output.json"), "w") as f:
            print("task_directory is", task_directory)
            json.dump(all_messages, f, indent=4, default=custom_encoder)

        usage_summary = {'total': planner.client.total_usage_summary, 'actual': planner.client.actual_usage_summary}
        with open(os.path.join(task_directory, "usage_summary.json"), "w") as f:
            json.dump(usage_summary, f, indent=4)

    finally:
        executor.cleanup()
        if user is not None:
            user.reset()
        if planner is not None:
            planner.reset()
