"""Convert an official BEAR annotation file into BEAR Agent task folders.

Each selected sample becomes ``<out_dir>/entry_<idx>/`` with its media files and a
``request.json`` that ``main.run_agent`` reads. Media paths in the official JSON are
resolved relative to the JSON file's directory.

Example:
    python agent/convert_bear_data.py \
        --official-json bear_official_data/trajectory/gripper_trajectory_official.json \
        --out-dir tasks/z_agent/trajectory/gripper_trajectory \
        --task-type trajectory --num 50 --seed 0
"""

import argparse
import ast
import json
import os
import random
import shutil

IMAGE_TASKS = {"pointing", "bbox", "trajectory"}
VIDEO_TASKS = {"object localization", "relative direction", "path planning",
               "next action prediction", "task progress reasoning"}
# video tasks whose last input is the current observation image
VIDEO_WITH_IMAGE_TASKS = {"relative direction", "path planning"}


def copy_media(src_root, rel_path, folder, stem):
    """Copy one media file into the entry folder as <stem><ext>; return its new name."""
    _, ext = os.path.splitext(rel_path)
    name = f"{stem}{ext}"
    shutil.copy(os.path.join(src_root, rel_path), os.path.join(folder, name))
    return name


def convert_entry(entry, src_root, out_dir, task_type):
    folder = os.path.join(out_dir, f"entry_{entry['idx']}")
    os.makedirs(folder, exist_ok=True)

    options = entry["options"]
    if isinstance(options, str):
        options = ast.literal_eval(options) if options.strip().startswith("{") else options

    request = {
        "question": entry["question"],
        # image tasks select their question template by this field
        "category": task_type,
        "options": options,
        "gt": entry["gt"],
    }
    if task_type in IMAGE_TASKS:
        request["images"] = [copy_media(src_root, entry["image"], folder, "image")]
        if entry.get("mask"):
            request["mask"] = copy_media(src_root, entry["mask"], folder, "mask")
    else:
        if task_type in VIDEO_WITH_IMAGE_TASKS:
            # a single file name, appended after the sampled video frames
            request["images"] = copy_media(src_root, entry["image"], folder, "image")
        request["video"] = copy_media(src_root, entry["video"], folder, "video")

    with open(os.path.join(folder, "request.json"), "w") as handle:
        json.dump(request, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--official-json", required=True, help="official BEAR annotation file")
    parser.add_argument("--out-dir", required=True, help="task folder, e.g. tasks/z_agent/<task>/<subset>")
    parser.add_argument("--task-type", required=True, choices=sorted(IMAGE_TASKS | VIDEO_TASKS))
    parser.add_argument("--num", type=int, default=None, help="randomly sample this many entries (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="seed for --num sampling")
    parser.add_argument("--idx", nargs="+", default=None, help="convert exactly these idx values")
    args = parser.parse_args()

    with open(args.official_json) as handle:
        data = json.load(handle)
    src_root = os.path.dirname(os.path.abspath(args.official_json))

    if args.idx:
        wanted = set(args.idx)
        selected = [entry for entry in data if str(entry["idx"]) in wanted]
        missing = wanted - {str(entry["idx"]) for entry in selected}
        if missing:
            parser.error(f"idx not found in {args.official_json}: {sorted(missing)}")
    elif args.num is not None:
        selected = random.Random(args.seed).sample(data, min(args.num, len(data)))
    else:
        selected = data

    os.makedirs(args.out_dir, exist_ok=True)
    for entry in selected:
        convert_entry(entry, src_root, args.out_dir, args.task_type)
    print(f"Converted {len(selected)} entries to '{args.out_dir}'")


if __name__ == "__main__":
    main()
