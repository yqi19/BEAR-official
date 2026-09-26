"""Shared agent settings; override deployment values with environment variables."""

import os

# set up the agent
MAX_REPLY = int(os.environ.get("BEAR_MAX_REPLY", "10"))
os.environ.setdefault("AUTOGEN_USE_DOCKER", "False")

# set up the LLM for the agent. The API key is read from OPENAI_API_KEY only.
MODEL = os.environ.get("BEAR_MODEL", "gpt-5")
_model_config = {"model": MODEL, "api_key": os.environ.get("OPENAI_API_KEY")}
# GPT-5 and o-series reasoning models only accept the default temperature.
if not MODEL.startswith(("gpt-5", "o1", "o3", "o4")):
    _model_config["temperature"] = 0.0
llm_config = {"cache_seed": None, "config_list": [_model_config]}

# use this after building your own server. You can also set up the server in other machines and paste them here.
SOM_ADDRESS = os.environ.get("SOM_ADDRESS", "http://localhost:8080/")
GROUNDING_DINO_ADDRESS = os.environ.get("GROUNDING_DINO_ADDRESS", "http://localhost:8081/")
DEPTH_ANYTHING_ADDRESS = os.environ.get("DEPTH_ANYTHING_ADDRESS", "http://localhost:8082/")
