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
Create and synchronize the native uv project environment. The environment is
created with access to ROS Jazzy's system-installed Python packages:

```bash
cd semantic_inference_python
uv venv \
  --python /usr/bin/python3.12 \
  --system-site-packages \
  .venv
uv sync --frozen
```

uv may report `No requires-python value found ... Defaulting to >=3.12`.
Python 3.12 is selected explicitly above; the metadata is intentionally omitted
for compatibility with ROS Jazzy's colcon/setuptools package introspection.

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

At this point, `pwd` must end in `/reasoning-hydra-sg`. Do not build from its
`ihmc-scene-graph` parent because that directory also contains `hydra-sg`, and
colcon will reject the duplicate package names provided by the two independent
workspaces.

Build and source the workspace from `reasoning-hydra-sg`:

```bash
uv run --project src/semantic_inference_ros/semantic_inference_python \
  colcon build \
    --symlink-install \
    --packages-up-to semantic_inference_ros \
    --cmake-args -DCMAKE_BUILD_TYPE=Release

source install/setup.bash
```

The build enables the native C++ TensorRT backend by default and discovers it
from the active CUDA toolkit. Closed-vocabulary segmentation requires the
TensorRT development headers plus the `nvinfer`, `nvonnxparser`, and
`nvinfer_plugin` libraries. No TensorRT-specific CMake arguments are required
for standard x86-64 CUDA installations or Jetson/aarch64 installations whose
libraries are exposed through the system or CUDA toolkit paths.

For a nonstandard installation, set `CUDA_HOME` before building:

```bash
export CUDA_HOME=/path/to/cuda
```

The Python TensorRT package inside `.venv` is used by Python export/runtime
tools; it does not replace the native C++ headers and libraries required by the
closed-vocabulary component.

## Usage

Before launching any node, source ROS and the workspace. For repeated ROS
commands, activate the uv-managed environment once:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source src/semantic_inference_ros/semantic_inference_python/.venv/bin/activate
export SCENE_GRAPH_ASSETS="${SCENE_GRAPH_ASSETS:-$PWD/scene_graph_assets}"
export HF_HOME="${HF_HOME:-$SCENE_GRAPH_ASSETS/models/vlm}"
export CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-$SCENE_GRAPH_ASSETS/models/vlm_embedding_encoder/clip}"
```

Use the dedicated launch file for the required vocabulary mode. Neither launch
requires model-selection arguments:

```bash
# Open vocabulary: selectable YOLOE or YOLO-World + FastSAM backend
ros2 launch semantic_inference_ros \
  open_vocabulary_segmentation.launch.py

# Closed vocabulary: ADE20K EfficientViT semantic segmentation
ros2 launch semantic_inference_ros \
  closed_vocabulary_segmentation.launch.py
```

Open-vocabulary mode uses the Alex label space configured in the selected
engine. Closed-vocabulary mode uses `ade20k_full` and reads:

```text
$SCENE_GRAPH_ASSETS/models/segmentation/pretrained/ade20k/
├── ade20k-efficientvit_seg_l2.onnx
└── ade20k-efficientvit_seg_l2.trt  # generated locally when absent
```

The closed-vocabulary C++ backend loads the TensorRT engine when present. If it
is missing, it builds the engine from the ONNX model and saves it alongside the
ONNX file. TensorRT engines remain platform-specific and should not be copied
between incompatible GPU/TensorRT environments.

Configuration:

```
semantic_inference_ros/config/openset_segmentation.yaml
```

Open-set segmentation supports two pipelines:

1. **YOLOE:** the YOLOE engine detects objects and produces instance masks
   directly.
2. **YOLO-World + FastSAM:** the YOLO-World engine detects and classifies object
   bounding boxes, then FastSAM generates an instance mask for each detected
   region.

Both pipelines use the configured CLIP/OpenCLIP encoder to generate compatible
image-text embeddings. Select exactly one `model.segmentation` block in
`semantic_inference_ros/config/openset_segmentation.yaml`, then run the same
launch command:

```bash
ros2 launch semantic_inference_ros \
  open_vocabulary_segmentation.launch.py
```

### TensorRT YOLOE deployment

YOLOE uses PyTorch CUDA by default. For lower-latency deployment, it can use a
fixed-class FP16 TensorRT engine. Choose one of the following workflows:

- **Use a prebuilt model:** download the published engine and launch the node.
  This is the normal path for users whose deployment platform matches the
  published artifact.
- **Export a new model:** use the isolated export environment when changing the
  YOLO checkpoint, label space, GPU platform, TensorRT version, or input shape.

```text
Maintainer: YOLOE checkpoint + labels
                    ↓ one-time export
              ONNX model + engine
                    ↓ publish
                 artifact store
                    ↓ download
User:          local model cache
                    ↓
          openset_segmentation_node
```

#### Using a prebuilt model

Prebuilt engines may be provisioned by the deployment environment while a
Docker image is built or before the segmentation node starts. The public
`semantic_inference_ros` package does not contain organization-specific
artifact URLs or credentials; it loads the local model path selected in its
configuration. Deployment-managed artifacts can be cached under:

```text
$SCENE_GRAPH_ASSETS/models/segmentation/pretrained/yoloe
```

The TensorRT model used at runtime can be found locally at:

```text
$SCENE_GRAPH_ASSETS/models/segmentation/pretrained/yoloe/alex-yoloe-11l-seg.engine
```

Deployment tooling should validate each download against its published
`.sha256` sidecar before making it available to ROS. Once cached, subsequent
launches can reuse the local engine without downloading or rebuilding it.

Select the backend and model in
`semantic_inference_ros/config/openset_segmentation.yaml`. The prebuilt YOLOE
TensorRT configuration is:

```yaml
model:
  segmentation:
    type: yoloe
    yolo_model_name: $SCENE_GRAPH_ASSETS/models/segmentation/pretrained/yoloe/alex-yoloe-11l-seg.engine
```

To use the pre-exported large YOLOE-26 engine instead, change the same entries:

```yaml
model:
  segmentation:
    type: yoloe
    yolo_model_name: $SCENE_GRAPH_ASSETS/models/segmentation/pretrained/yoloe/alex-yoloe-26l-seg.engine
```

To use YOLO-World with FastSAM mask generation:

```yaml
model:
  segmentation:
    type: yolosam
    yolo_model_name: $SCENE_GRAPH_ASSETS/models/segmentation/pretrained/yolow/alex-yolov8l-worldv2.engine
    model_name: $SCENE_GRAPH_ASSETS/models/segmentation/pretrained/fastsam/FastSAM-x.pt
```

After saving the configuration, relaunch the node without model-selection
arguments:

```bash
ros2 launch semantic_inference_ros \
  open_vocabulary_segmentation.launch.py
```

The no-argument open-vocabulary launch selects the Alex label-space and grouping
files for the prebuilt Alex engines. Any replacement fixed-class engine must be
exported with the same ordered class list, or its matching label-space and
grouping configuration must be selected.

Segmentation artifacts are organized by backend:

```text
$SCENE_GRAPH_ASSETS/models/segmentation/
├── yoloe/
├── yolow/
├── fastsam/
│   ├── FastSAM-x.pt
│   └── FastSAM-x.pt.sha256
└── ade20k/
    ├── ade20k-efficientvit_seg_l2.onnx
    └── ade20k-efficientvit_seg_l2.onnx.sha256
```

YOLO-World supplies open-vocabulary detections but not instance masks, so the
`yolosam` configuration additionally loads the local FastSAM checkpoint to
segment each detected region. ADE20K also has a local EfficientViT ONNX model
under `models/segmentation/pretrained/ade20k/`. A TensorRT engine is not currently present;
it may be generated from this ONNX model for a compatible deployment platform
and stored in the same directory with its own checksum.

#### Exporting and publishing a new model

TensorRT 11 export uses NumPy 2 and a newer PyTorch stack. These dependencies
are intentionally isolated from the main `.venv` because ROS Jazzy's
`cv_bridge` binary requires NumPy 1.x on Ubuntu 24.04.

From the `reasoning-hydra-sg` workspace root, create the pinned export
environment once:

```bash
uv venv --python /usr/bin/python3.12 .venv-tensorrt-export
uv pip install \
  --python .venv-tensorrt-export/bin/python \
  -r src/semantic_inference_ros/semantic_inference_python/requirements-tensorrt-export.txt
```

Export the checkpoint with the required label space:

```bash
export SCENE_GRAPH_ASSETS="${SCENE_GRAPH_ASSETS:-$PWD/scene_graph_assets}"

.venv-tensorrt-export/bin/python \
  src/semantic_inference_ros/semantic_inference_python/scripts/export_yolo.py \
  --model "$SCENE_GRAPH_ASSETS/models/segmentation/pretrained/yoloe/yoloe-11l-seg.pt" \
  --labels src/reasoning_hydra/config/label_spaces/alex_label_space.yaml \
  --output "$SCENE_GRAPH_ASSETS/models/segmentation/pretrained/yoloe/alex-yoloe-11l-seg.engine" \
  --device cuda:0 \
  --half
```

For a different deployment model, change both the source checkpoint and output
artifact path. Keep YOLOE artifacts under
`$SCENE_GRAPH_ASSETS/models/segmentation/pretrained/yoloe/` and YOLO-World artifacts under
`$SCENE_GRAPH_ASSETS/models/segmentation/pretrained/yolow/`. For example, use
`yoloe-26l-seg.pt` with
`$SCENE_GRAPH_ASSETS/models/segmentation/pretrained/yoloe/alex-yoloe-26l-seg.engine`, or
`yolov8l-worldv2.pt` with
`$SCENE_GRAPH_ASSETS/models/segmentation/pretrained/yolow/alex-yolov8l-worldv2.engine`.
The exporter
detects YOLO-World checkpoints and embeds the configured Alex labels before
producing the static engine.

The exporter creates the customized checkpoint, ONNX representation, and
TensorRT engine in the output directory. Run a representative inference test
on the resulting engine before publishing it.

Generate one checksum sidecar per published artifact:

```bash
cd "$SCENE_GRAPH_ASSETS/models/segmentation/pretrained/yoloe"
sha256sum alex-yoloe-11l-seg.onnx \
  > alex-yoloe-11l-seg.onnx.sha256
sha256sum alex-yoloe-11l-seg.engine \
  > alex-yoloe-11l-seg.engine.sha256
```

Upload the `.engine`, `.onnx`, and their individual `.sha256` files together. Publish
their artifact version, TensorRT version, GPU architecture, CUDA/runtime stack,
input shape, precision, and label-space revision alongside the download links.

ONNX artifacts are relatively portable across compatible runtimes. TensorRT
engines are coupled to the GPU architecture, TensorRT version, CUDA/runtime
stack, precision, input profile, and embedded label configuration. Regenerate
the engine when any of these change. The Python, PyTorch, and NumPy versions in
the export and ROS environments may differ, provided the resulting engine is
executed by a compatible TensorRT runtime.

The labels are embedded during export and cannot be changed dynamically at
runtime. Re-export the engine after modifying the label space. Perform artifact
downloads during machine or container setup rather than whenever the ROS node
starts. Keep private credentials outside the repository. A deployment that
must operate without network access should include the artifacts in its
installation bundle or mount them from a local model volume.


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

The extraction procedure uses the Transformers version pinned in `uv.lock`.

```bash
export SCENE_GRAPH_ASSETS="${SCENE_GRAPH_ASSETS:-$PWD/scene_graph_assets}"
export HF_HOME="${HF_HOME:-$SCENE_GRAPH_ASSETS/models/vlm}"

uv run --project src/semantic_inference_ros/semantic_inference_python python -c \
  "from transformers import Qwen3VLForConditionalGeneration; print('Qwen3VL support is available')"

uv run --project src/semantic_inference_ros/semantic_inference_python python \
  src/semantic_inference_ros/semantic_inference_python/scripts/extract_qwen3vl_visual.py \
  --model_name nvidia/Cosmos-Reason2-2B \
  --output_path "$SCENE_GRAPH_ASSETS/models/vlm_vision_encoder/cosmos-reason2-2b-visual"
```

Transformers 4.57.6 is pinned by this repository to provide the required
Qwen3VL model and processor APIs reproducibly.

The full downloaded checkpoint remains in Hugging Face's managed
`$HF_HOME/hub` cache. The command produces `config.json`, `vision_config.json`,
and `vision.pt` under `$SCENE_GRAPH_ASSETS/models/vlm_vision_encoder`, outside the Hub cache's
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
and disables the optional OpenAI response parser
(`use_llm_response_parser: false`). The OpenAI and FastAPI clients are inactive
in this configuration. Their credentials apply only to configurations that
explicitly enable the corresponding remote client.

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
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── .venv/                  # Generated by uv; not committed
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
