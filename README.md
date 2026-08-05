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
git clone git@github.com:ihmcrobotics/ihmc-object-detection-pipeline.git
cd ihmc-object-detection-pipeline/instance_segmentation/instance_segmentation_2d/semantic_inference_ros
git clone git@github.com:ntnu-arl/DeepSeek-VL2.git   semantic_inference_python/src/semantic_inference_python/models/deepseek
git clone git@github.com:MIT-SPARK/config_utilities.git
```

Clone hydra and corresponding modules:
```
echo "build: {cmake-args: [-DCMAKE_BUILD_TYPE=Release]}" > colcon_defaults.yaml

cd src
git clone git@github.com:MIT-SPARK/Hydra-ROS.git hydra_ros
vcs import . < hydra_ros/install/ros2.yaml
rosdep install --from-paths . --ignore-src -r -y

cd ..
colcon build --continue-on-error
```


Clone the IHMC object-detection pipeline:

```bash
git clone git@github.com:ihmcrobotics/ihmc-object-detection-pipeline.git
cd ihmc-object-detection-pipeline/instance_segmentation/instance_segmentation_2d
```

Clone DeepSeek-VL2 into the expected Python module location:

```bash
git clone git@github.com:ntnu-arl/DeepSeek-VL2.git \
  semantic_inference_ros/semantic_inference_python/src/semantic_inference_python/models/deepseek
```

Clone Hydra-ROS:

```bash
git clone git@github.com:MIT-SPARK/Hydra-ROS.git hydra_ros
```

Import the ROS 2 dependencies specified by Hydra-ROS:

```bash
vcs import . < hydra_ros/install/ros2.yaml
```

> Before importing, check whether the manifest includes packages already present in `semantic_inference_ros`, such as `config_utilities` or `semantic_inference`, to avoid duplicate ROS package names.

Install dependencies. Source ROS 2 Jazzy:

```bash
source /opt/ros/jazzy/setup.bash
```

Install ROS dependencies:

```bash
rosdep install \
  --from-paths . \
  --ignore-src \
  -r \
  -y \
  --rosdistro jazzy
  
sudo apt install ros-jazzy-gtsam
```

Configure the build. Create `colcon_defaults.yaml` in `instance_segmentation_2d`:

```yaml
build:
  symlink-install: true
  cmake-args:
    - -DCMAKE_BUILD_TYPE=Release
```

Install Hydra dependencies:
```
sudo apt update
sudo apt install \
    nlohmann-json3-dev \
    libgoogle-glog-dev \
    libcli11-dev
```

Build. From `instance_segmentation_2d`, run:

```bash
colcon build \
  --symlink-install \
  --continue-on-error \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

After the build completes:

```bash
source install/setup.bash
```


### Virtual Environment

It is highly recommended to set up a **Python virtual environment** to run ROS Python nodes:

```bash
cd semantic_inference_python
python3.12 -m venv --system-site-packages ros_semantics_env
source ros_semantics_env/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### Building

Install ROS2 dependencies:

```bash
cd ..
source /opt/ros/jazzy/setup.bash
rosdep install \
  --from-paths semantic_inference_ros \
  --ignore-src \
  -r \
  -y
```

For **closed-set segmentation**, follow the setup instructions (skip Python utilities) in [semantic_inference closed-set docs](https://github.com/MIT-SPARK/semantic_inference/blob/archive/ros_noetic/docs/closed_set.md).

Build the workspace:

```bash
colcon build --symlink-install --base-paths semantic_inference_ros
source install/setup.bash
```

---

## Usage

### Open-set Segmentation

Open-set segmentation consumes **RGB-D images** and camera information to perform semantic segmentation and extract **open-vocabulary features** for each object.  

- Launch file: [openset_segmentation.launch](./semantic_inference_ros/launch/openset_segmentation.launch.py)  
- Configuration: [openset_segmentation.yaml](./semantic_inference_ros/config/openset_segmentation.yaml)  

Supported open-set detectors: [YOLOe](https://docs.ultralytics.com/models/yoloe/) and [YOLOw](https://docs.ultralytics.com/models/yolo-world/). These can detect any list of objects without re-training.

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch semantic_inference_ros openset_segmentation.launch.py
```

### VLM for Object Relationship Embeddings

This method takes a segmented image along with its **original RGB-D frame** and computes **visual features** for each pair of detected objects. These features can be used to prompt a VLM for reasoning about relationships.  

- Launch file: [vlm_features_node.launch](./semantic_inference_ros/launch/vlm_features_node.launch)  
- Configuration: [vlm.yaml](./semantic_inference_ros/config/vlm.yaml)  

Supported VLMs: [InstructBLIP](https://huggingface.co/collections/Salesforce/instructblip-models) and [DeepSeek-VL2](https://huggingface.co/deepseek-ai/deepseek-vl2).

To use **DeepSeek-VL2**, first extract the visual encoder as a standalone model. For the large model (used in our experiments), we provide it [here](https://huggingface.co/ntnu-arl/deepseek-vl2-vision-enc). 

Alternmatively, the models can be extracted with the following command (~100GB RAM required for the large moded):
```bash
python semantic_inference_python/scripts/extract_deepseek_visual.py --model_name <model to use> --output_path <path to store model>
```

Then, set the model path in [vlm.yaml](./semantic_inference_ros/config/vlm.yaml).

Launch the node:

```bash
roslaunch semantic_inference_ros vlm_features.launch
```

### VLM/LLM Reasoning

This section enables reasoning on the [relationship-aware hierarchical scene graph](https://github.com/ntnu-arl/reasoning_hydra).  

- LLMs predict relevant objects and interactions for given tasks  
- VLM responses are parsed by LLMs  
- **OpenAI API key** required, run:  
```bash
export OPENAI_API_KEY=<Your OpenAI API Key>
``` 

VLM reasoning is performed on the cloud. Use [DeepSeek-VL2 server code](https://github.com/ntnu-arl/DeepSeek-VL2/tree/server) to run FastAPI server.

Steps to set up the server:

1. Clone the server repo:

```bash
git clone git@github.com:ntnu-arl/DeepSeek-VL2.git -b server
cd DeepSeek-VL2
```

2. Set up the Python virtual environment:

```bash
bash setup.sh
```

3. Configure server path, port, and API key in `run_server.sh`.

4. Run the server (model download may take time):

```bash
bash run_server.sh
```

Finally, set the **server URL** in [vlm_for_navigation.yaml](./semantic_inference_ros/config/vlm_for_navigation.yaml) and export your FASTAPI_KEY:
```bash
export FASTAPI_API_KEY=<Your server FastAPI Key>
``` 

## Contact

For questions or support, reach out via or contact the authors:

- [Arghya Chatterjee](mailto:achatterjee@ihmc.org) 
