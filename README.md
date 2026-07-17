# ⛱️ICML 2026 [BEAR: Benchmarking and Enhancing Multimodal Language Models with Atomic Embodied Capabilities](https://bear-official66.github.io/)

<a href="https://bear-official66.github.io/"><strong>Project Page</strong></a>
  |
  <a href="https://arxiv.org/abs/2510.08759"><strong>arXiv</strong></a>
  |
  <a href="https://huggingface.co/datasets/yqi19/BEAR-benchmark"><strong>Data</strong></a>
  |
  <a href="#"><strong>Twitter</strong></a> 
  | <a href="https://huggingface.co/papers/2510.08759"><strong>Huggingface</strong></a>

 ## 🚀 Todos

 - [X] Release BEAR Benchmark with evaluation scripts
 - [ ] Release the official code of BEAR-Agent

## 💻 Get access to benchmark data

Our official benchmark is released at [<a href="https://huggingface.co/datasets/yqi19/BEAR-benchmark/tree/main"><strong>Link</strong></a>]. The readme.md is organized for your agent to set up your evaluation scripts.

### Dataset Structure

The layout of the data structure:

```python
.
├── README.md
├── requirements.txt
├── run_api_model.py      # inference with API models (gpt / gemini / claude)
├── run_image_model.py    # inference with local VLMs (VLMEvalKit)
├── run_custom_model.py   # inference with any local model, NO VLMEvalKit
├── bear_models.py        # pluggable adapters (Cosmos, Qwen2.5-VL, Echo)
├── eval.py               # grading -> final score
├── util/                 # prompt templates, API wrappers, image grid
│   ├── prompt_generation.py
│   ├── gpt.py  gemini.py  claude.py
│   └── concate_image.py
└── <task folders>/       # data (*.json + media) + run.sh
```

### API and model preparations

You need to have the access to OpenAI API models to calculate the final score. At each category, the output file of the test model's answer will be saved in a `json` file, and the judge model will calculate the final success rate. `run_custom_model.py` is designed for custom model evaluation.


## BibTex
We would appreciate it if you find this work useful and consider citing it🎉!
```
@article{qi2025bear,
  title={BEAR: Benchmarking and Enhancing Multimodal Language Models for Atomic Embodied Capabilities},
  author={Qi, Yu and Zhao, Haibo and Guo, Ziyu and Ma, Siyuan and Chen, Ziyan and Han, Yaokun and Zhang, Renrui and Lin, Zitiantao and Xin, Shiji and Huang, Yijian and others},
  journal={arXiv preprint arXiv:2510.08759},
  year={2025}
}
```

## Contact
Please contact Yu (qi.yu2@northeastern.edu) or Haibo (zhao.haib@northeastern.edu) if you would like discussions!
