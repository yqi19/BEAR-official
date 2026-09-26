# BEAR-Agent

<a href="https://bear-official66.github.io/"><strong>Project Page</strong></a>
  |
  <a href="https://arxiv.org/abs/2510.08759"><strong>arXiv</strong></a>
  |
  <a href="https://huggingface.co/datasets/yqi19/BEAR-benchmark"><strong>Data</strong></a>

BEAR-Agent is the official agent from **BEAR: Dissecting Embodied Abilities in Multimodal Language Models through Skill-level Evaluation and Diagnosis**. It uses **GPT-5** as the backbone and solves BEAR tasks by writing Python code that calls vision tools:

- GroundingDINO for detection
- Semantic-SAM for segmentation
- Depth-Anything for depth
- an arrow-extension tool for trajectory reasoning

The code runs in a Jupyter kernel, and the agent keeps reasoning over the text and image outputs until it reaches an answer.

## 📑 Contents

- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Data Preparation](#-data-preparation)
- [Running the Agent](#-running-the-agent)
- [Evaluation](#-evaluation)
- [Acknowledgement](#-acknowledgement)
- [Citation](#-citation)

## 📂 Repository Structure

```text
BEAR-Agent/
├── agent/
│   ├── config.py                        # model and vision-server settings (environment variables)
│   ├── convert_bear_data.py             # BEAR annotation file -> per-question task folders
│   ├── running_agent_batch.py           # run the agent on a folder of tasks
│   ├── running_evaluate_agent_batch.py  # score agent outputs
│   ├── main.py                          # single-task entry: run_agent(...)
│   ├── prompt.py                        # system prompts per task type (Perception / Traj / Planning ...)
│   ├── builder/build_prompt_new.py      # question and answer-format templates
│   ├── tools.py                         # vision tool clients and image tools
│   ├── agent.py  execution.py  parse.py # agent loop, Jupyter execution, code parsing
│   └── running_agent_baseline_batch_*.py  # non-agent VLM baselines (needs VLMEvalKit)
├── vision_experts/                      # GroundingDINO, Semantic-SAM (SoM) and Depth-Anything servers
├── record_viewer.ipynb                  # browse agent traces
├── sketchpad.yaml                       # exact conda env for the agent
└── vision_expert.yaml                   # exact conda env for the vision experts
```

## 🛠 Installation

### 1. Agent environment

```bash
conda create -n bear_agent python=3.9 -y
conda activate bear_agent

pip install pyautogen==0.2.26 'pyautogen[jupyter-executor]==0.2.26'
# GPT-5 needs a newer OpenAI SDK than pyautogen 0.2.26 pins; ignore pip's version-conflict warning
pip install openai==1.98.0
pip install Pillow numpy opencv-python matplotlib scipy networkx gradio_client tqdm \
            cairosvg chess reportlab ipython
```

We tested with `python 3.9`, `pyautogen 0.2.26` and `openai 1.98.0`. To reproduce the exact environment instead, run `conda env create -f sketchpad.yaml`.

### 2. Vision experts

Each vision expert runs as a gradio server, either on the same machine or on a separate GPU server.

```bash
conda env create -f vision_expert.yaml
conda activate vision_expert
```

Then follow [`vision_experts/installation.md`](vision_experts/installation.md) to build the experts, download their checkpoints and launch the three servers. By default the agent expects them on the following ports:

| Server | Default port |
| --- | --- |
| SoM | 8080 |
| GroundingDINO | 8081 |
| Depth-Anything | 8082 |

## ⚙️ Configuration

All settings are read from environment variables in [`agent/config.py`](agent/config.py):

```bash
export OPENAI_API_KEY='your-api-key'                    # required
export BEAR_MODEL='gpt-5'                               # default: gpt-5
export BEAR_MAX_REPLY='10'                              # max agent turns, default: 10
export SOM_ADDRESS='http://localhost:8080/'             # default values shown
export GROUNDING_DINO_ADDRESS='http://localhost:8081/'
export DEPTH_ANYTHING_ADDRESS='http://localhost:8082/'
```

GPT-5 only accepts its default sampling temperature, so no temperature is sent. If `BEAR_MODEL` is a non-reasoning model such as `gpt-4o`, the agent sends `temperature=0`.

## 📦 Data Preparation

### Download

Download the benchmark from [HuggingFace](https://huggingface.co/datasets/yqi19/BEAR-benchmark):

```bash
huggingface-cli download yqi19/BEAR-benchmark --repo-type dataset --local-dir BEAR-benchmark
```

Each task has one annotation file. Media paths inside the file are relative to it:

```text
BEAR-benchmark/
├── pointing/           {general_object,semantic_part,spatial_relationship}_pointing_official.json
├── bbox/               {general_object,semantic_part,spatial_relationship}_bbox_official.json
├── trajectory/         {gripper,human_hand,object}_trajectory_official.json
├── spatial_reasoning/  {object_localization,relative_direction,path_planning}_official.json
└── task_planning/      {next_action_prediction,task_progress_reasoning}_official.json
```

### Convert to agent task folders

The agent reads one folder per question. [`agent/convert_bear_data.py`](agent/convert_bear_data.py) turns an annotation file into `<out-dir>/entry_<idx>/`. Each entry folder holds a `request.json` and copies of the media:

| `--task-type` | Files per entry | `request.json` fields |
| --- | --- | --- |
| `pointing`, `bbox`, `trajectory` | `image.*`, and `mask.*` for pointing/bbox | `question`, `category`, `options`, `gt`, `images` (list), `mask` |
| `object localization`, `next action prediction`, `task progress reasoning` | `video.mp4` | `question`, `category`, `options`, `gt`, `video` |
| `relative direction`, `path planning` | `video.mp4`, `image.*` (current observation) | `question`, `category`, `options`, `gt`, `video`, `images` (one file name) |

To convert a single subset:

```bash
python agent/convert_bear_data.py \
  --official-json BEAR-benchmark/pointing/general_object_pointing_official.json \
  --out-dir tasks/bear/pointing/general_object_pointing \
  --task-type pointing
```

By default the script converts every question. Use `--num N --seed S` to sample N questions reproducibly, or `--idx 3 17 42` to convert specific questions.

To convert the whole benchmark:

```bash
BEAR=BEAR-benchmark
convert() { python agent/convert_bear_data.py --official-json "$BEAR/$1" --out-dir "tasks/bear/$2" --task-type "$3"; }

for s in general_object semantic_part spatial_relationship; do
  convert pointing/${s}_pointing_official.json pointing/${s}_pointing pointing
  convert bbox/${s}_bbox_official.json         bbox/${s}_bbox         bbox
done
for s in gripper human_hand object; do
  convert trajectory/${s}_trajectory_official.json trajectory/${s}_trajectory trajectory
done
convert spatial_reasoning/object_localization_official.json  object_localization/object_localization   "object localization"
convert spatial_reasoning/relative_direction_official.json   relative_direction/relative_direction     "relative direction"
convert spatial_reasoning/path_planning_official.json        path_planning/path_planning               "path planning"
convert task_planning/next_action_prediction_official.json   next_action_prediction/next_action_prediction   "next action prediction"
convert task_planning/task_progress_reasoning_official.json  task_progress_reasoning/task_progress_reasoning "task progress reasoning"
```

## 🚀 Running the Agent

Start the vision expert servers first, then run commands from the `BEAR-Agent/` root.

[`agent/running_agent_batch.py`](agent/running_agent_batch.py) runs every `entry_*` under each subset folder of `--tasks-root`. `--task-type` selects the system prompt:

| `--task-type` | Prompt class ([`agent/prompt.py`](agent/prompt.py)) |
| --- | --- |
| `pointing`, `bbox` | `PerceptionPrompt` |
| `trajectory` | `TrajPrompt` |
| `object localization` | `ObjectLocalizationPrompt` |
| `relative direction` | `RelativeDirectionPrompt` |
| `path planning`, `next action prediction`, `task progress reasoning` | `PlanningPrompt` |

To preview the calls without starting the model, Jupyter or the vision servers:

```bash
python agent/running_agent_batch.py \
  --tasks-root tasks/bear/pointing --outputs-root outputs/bear/pointing \
  --task-type pointing --dry-run
```

To run the agent:

```bash
python agent/running_agent_batch.py \
  --tasks-root tasks/bear/pointing --outputs-root outputs/bear/pointing \
  --task-type pointing
```

For video tasks, the agent samples 16 frames from `video.mp4`. For relative direction and path planning, the current observation image is appended as the last frame.

Each question writes `outputs/.../<subset>/entry_<idx>/` containing:
- copies of the inputs
- `output.json`, the full conversation
- `usage_summary.json`

Open [`record_viewer.ipynb`](record_viewer.ipynb) to browse a trace.

To run a single question:

```bash
cd agent
python -c "from main import run_agent; run_agent('../tasks/bear/trajectory/gripper_trajectory/entry_6', '../outputs/bear/trajectory/gripper_trajectory', task_type='trajectory')"
```

## 📊 Evaluation

```bash
cd agent
python running_evaluate_agent_batch.py \
  --tasks-root ../tasks/bear/pointing --outputs-root ../outputs/bear/pointing \
  --task-type pointing --result-dir eval_result
```

| Task | Metric |
| --- | --- |
| Pointing | The predicted point must fall inside the ground-truth mask |
| Bbox | Mean IoU between the predicted box and the ground-truth mask |
| Multiple-choice tasks | The chosen letter is extracted from the final reply with `gpt-3.5-turbo` (needs `OPENAI_API_KEY`) and compared with `gt` |

Per-question results and a summary for each subset are written to `--result-dir`.

The `running_agent_baseline_batch_*.py` scripts run the non-agent VLM baselines. They additionally require [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

## 🙏 Acknowledgement

The agent framework builds on [Visual Sketchpad](https://github.com/Yushi-Hu/VisualSketchpad) (Hu et al., NeurIPS 2024). The vision experts use [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM) and [Depth-Anything](https://github.com/LiheYoung/Depth-Anything).

## 📝 Citation

```bibtex
@article{qi2025bear,
  title={BEAR: Dissecting Embodied Abilities in Multimodal Language Models through Skill-level Evaluation and Diagnosis},
  author={Qi, Yu and Zhao, Haibo and Guo, Ziyu and Ma, Siyuan and Chen, Ziyan and Han, Yaokun and Zhang, Renrui and Lin, Zitiantao and Zhu, Yizhe and Xin, Shiji and Huang, Yijian and Hu, Boce and Cheng, Kai and Zhang, Jiayi and Wang, Peiheng and Liu, Jiazheng and Wang, Wenqing and Qin, Yiran and Huang, Haojie and Wong, Lawson L.S.},
  journal={arXiv preprint arXiv:2510.08759},
  year={2025}
}
```
