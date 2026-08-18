# <div align="center">Semantic Segmentation and VLM Reasoning in ROS2</div>

This repository extends [semantic_inference](https://github.com/MIT-SPARK/semantic_inference) to provide **closed and open set semantic segmentation** methods. Additionally, it provides methods to extract **CLIP embeddings** of objects and **relational embeddings** using Visual Language Models (VLMs).

## Setup

### General Requirements

These instructions assume `python3.12` and `ros-jazzy-desktop-full` is installed on **Ubuntu 24.04**.  

Install the general dependencies:

```bash
sudo apt update
sudo apt install \
  python3-rosdep \
  python3-colcon-common-extensions \
  python3-venv
```

Clone the repository and initialize submodules:

```bash
git clone https://github.com/ihmcrobotics/semantic_inference_ros.git
cd semantic_inference_ros
git submodule update --init
```
Create the Python virtual environment:
```bash
cd semantic_inference_python
python3.12 -m venv --system-site-packages ros_semantics_env
source ros_semantics_env/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```
Install ROS dependencies from root package (`reasoning-hydra-sg`):

```bash
cd ../../..
source /opt/ros/jazzy/setup.bash

rosdep install \
  --from-paths src/semantic_inference_ros \
  --ignore-src \
  --rosdistro jazzy \
  -r \
  -y
```
Build and source the workspace from root package (`reasoning-hydra-sg`): 
```bash
colcon build \
  --symlink-install \
  --packages-up-to semantic_inference_ros \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

## Usage

Before launching any node, activate the virtual environment and source the ROS workspace:

```bash
source semantic_inference_python/ros_semantics_env/bin/activate

source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

Launch the open-set semantic segmentation node:

```bash
ros2 launch semantic_inference_ros openset_segmentation.launch.py
```

Configuration:

```
semantic_inference_ros/config/openset_segmentation.yaml
```

Supported segmentation models include:

- YOLOE
- YOLO-World


Launch the CLIP feature extraction node:

```bash
ros2 launch semantic_inference_ros clip_features.launch.py
```


This node extracts relational visual features between segmented objects for downstream VLM reasoning.

Launch:

```bash
ros2 launch semantic_inference_ros vlm_features.launch.py
```

Configuration:

```
semantic_inference_ros/config/vlm.yaml
```

The maintained VLM backbones are:

- [InstructBLIP](https://huggingface.co/collections/Salesforce/instructblip-models)
- Qwen3VL, used by [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)

For **InstructBLIP**, exporting the vision encoder is optional. The standard
wrapper loads the checkpoint and selects its `.vision_model` automatically.
Use `export_blip_visual.py` only when preparing the vision encoder for an
ONNX/TensorRT deployment path:

```text
InstructBLIP vision encoder
        ↓
       ONNX
        ↓
     TensorRT
```

For **Cosmos-Reason-2-2B**, extract its Qwen3VL vision component locally:

The setup instructions above already install the pinned Transformers version
from `requirements.txt`; no additional installation is needed here.

```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

python3 -c "from transformers import Qwen3VLForConditionalGeneration; print('Qwen3VL support is available')"

python3 semantic_inference_python/scripts/extract_qwen3vl_visual.py \
  --model_name nvidia/Cosmos-Reason2-2B \
  --output_path "$HF_HOME/semantic_inference/cosmos-reason2-2b-visual"
```

Transformers 4.57.6 is pinned by this repository to provide the required
Qwen3VL model and processor APIs reproducibly.

The full downloaded checkpoint remains in Hugging Face's managed
`$HF_HOME/hub` cache. The command produces `config.json`, `vision_config.json`,
and `vision.pt` under `$HF_HOME/semantic_inference`, outside the Hub cache's
internal directory structure.
The `qwen3vl_visual` runtime wrapper loads the extracted model and returns
fixed-size relationship features. Select it in
`semantic_inference_ros/config/vlm.yaml`, rebuild the package, and launch
`vlm_features_node`.

Cosmos-Reason2 is used in two stages:

```text
RGB image + object relationships
        ↓
qwen3vl_visual (extracted vision encoder)
        ↓
196 × 2048 relationship-feature tensor
        ↓
cosmos_reason2 (language model from the full checkpoint)
        ↓
relationship labels or navigation decisions
```

In the complete navigation pipeline, detection, retrieval, and VLM reasoning
have distinct roles:

```text
RGB + depth → YOLOE → masks/classes → Hydra object nodes
                                           │
Instruction → Qwen3-1.7B → object names    │
                         → OpenCLIP → candidate object search
                                           │
RGB + object pairs → Qwen3VL relationship features
                                           │
candidate objects + prompt + relationship features
                         → Cosmos-Reason2 → selected objects
                         → Hydra find_paths → navigation output
```

YOLOE creates the object observations and OpenCLIP retrieves candidates from
the instruction. The VLM connection occurs after that retrieval: Qwen3VL
encodes object-pair appearance, and Cosmos-Reason2 decides whether those
relationships are relevant to the requested navigation task.

The extracted `vision.pt` is sufficient for `vlm_features_node`. The
generative nodes also load the language portion of the full
`nvidia/Cosmos-Reason2-2B` checkpoint from the Hugging Face cache.

To turn relationship-feature messages into text labels, run:

```bash
ros2 launch semantic_inference_ros vlm_text_generator.launch.py
```

This launch uses `semantic_inference_ros/config/vlm_prompting.yaml`.

Launch for VLM Reasoning:

```bash
ros2 launch semantic_inference_ros vlm_for_navigation.launch.py
```

Configuration:

```
semantic_inference_ros/config/vlm_for_navigation.yaml
```

The default configuration runs Cosmos-Reason2 locally (`use_server: false`)
and disables the optional GPT response parser, so it does not require OpenAI
or FastAPI credentials. API keys are only needed if those optional remote
paths are explicitly enabled.

Initial task instructions are parsed locally by `Qwen/Qwen3-1.7B`, configured
in `semantic_inference_ros/config/navigation_prompt_parser.yaml`. Qwen returns
the same structured `objects` and `interactions` JSON previously requested
from OpenAI. OpenCLIP then embeds those object names for Hydra object search.
The model is downloaded to the Hugging Face cache on first launch:

```bash
ros2 launch semantic_inference_ros navigation_prompt_parser.launch.py
```

For the complete local VLM path, start the nodes in this order:

```bash
ros2 launch semantic_inference_ros vlm_features.launch.py
ros2 launch semantic_inference_ros vlm_text_generator.launch.py
ros2 launch semantic_inference_ros vlm_for_navigation.launch.py
```

## Repository Structure

```
semantic_inference_ros/
├── config_utilities/
├── semantic_inference/
├── semantic_inference_msgs/
├── semantic_inference_python/
│   ├── ros_semantics_env/
│   └── src/
│       └── semantic_inference_python/
│           └── models/
├── semantic_inference_ros/
├── LICENSE
└── README.md
```

The repository includes the following project submodule:

- `config_utilities`

Initialize them after cloning with:

```bash
git submodule update --init
```


## Contact

For questions or support, contact:

- Arghya Chatterjee (achatterjee@ihmc.org)
