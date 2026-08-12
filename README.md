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

Supported VLM backbones include:

- InstructBLIP
- DeepSeek-VL2
- Qwen3VL (Cosmos-Reason-2-2B)

To use InstructBLIP, using `export_blip_visual.py` script is optional because the normal InstructBLIP wrapper already knows how to extract `.vision_model` automatically. That export script becomes useful when you want to avoid that normal PyTorch path and instead do:
```
InstructBLIP vision encoder
        ↓
       ONNX
        ↓
     TensorRT
```

To use DeepSeek-VL2, first extract the visual encoder.

For the large model used in our experiments:

https://huggingface.co/ntnu-arl/deepseek-vl2-vision-enc

Alternatively, extract deepseek locally:

```bash
python3 semantic_inference_python/scripts/extract_deepseek_visual.py \
    --model_name <model_name> \
    --output_path <output_path>
```

Or, extract qwen3-vl locally:

```bash
python3 semantic_inference_python/scripts/extract_qwen3vl_visual.py \
    --model_name <model_name> \
    --output_path <output_path>
```

Then update the model path in:

```
semantic_inference_ros/config/vlm.yaml
```

Launch for VLM Reasoning:

```bash
ros2 launch semantic_inference_ros vlm_for_navigation.launch.py
```

Configuration:

```
semantic_inference_ros/config/vlm_for_navigation.yaml
```

Before launching, export the required API keys:

```bash
export OPENAI_API_KEY=<your_openai_api_key>
export FASTAPI_API_KEY=<your_fastapi_api_key>
```

The DeepSeek-VL2 FastAPI server can be found at:

https://github.com/ntnu-arl/DeepSeek-VL2/tree/server



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
│               └── deepseek/
├── semantic_inference_ros/
├── LICENSE
└── README.md
```

The repository includes the following Git submodules:

- `config_utilities`
- `DeepSeek-VL2`

Initialize them after cloning with:

```bash
git submodule update --init
```


## Contact

For questions or support, contact:

- Arghya Chatterjee (achatterjee@ihmc.org)
