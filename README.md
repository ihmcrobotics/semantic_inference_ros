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

On an x86-64 NVIDIA deployment that runs `.engine` models through the Python
segmentation node, install the locked TensorRT runtime extra:

```bash
uv sync --frozen --extra tensorrt
```

The isolated `.venv-tensorrt-export` remains the model-conversion environment;
the `tensorrt` extra only makes compatible TensorRT engines loadable from the
ROS runtime `.venv`.

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

# Fixed-vocabulary instance segmentation: pretrained COCO YOLOv8
ros2 launch semantic_inference_ros \
  pretrained_yolov8_segmentation.launch.py

# Fixed-vocabulary instance segmentation: custom-trained YOLOv8
ros2 launch semantic_inference_ros \
  custom_yolov8_segmentation.launch.py
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

### Fixed-vocabulary YOLOv8 instance segmentation

YOLOv8-seg is closed vocabulary, but it is not the same interface as the
closed-vocabulary ADE20K backend. ADE20K publishes one semantic class per
pixel; YOLOv8 publishes object detections and instance masks. The YOLOv8 paths
therefore share the Python instance-segmentation node while using a dedicated
`yolo` wrapper and fixed class lists.

Two no-argument launches prevent their class indices from being mixed:

```bash
# 80 COCO classes, 640-by-640 input
ros2 launch semantic_inference_ros \
  pretrained_yolov8_segmentation.launch.py

# 15 custom classes, 736-by-1280 input
ros2 launch semantic_inference_ros \
  custom_yolov8_segmentation.launch.py
```

Each launch loads its version-controlled native class ordering from
`semantic_inference/config/label_groupings/` and validates the declared class
count. These small YAML files are installed with the ROS package; they are not
downloaded as model artifacts. The pretrained and custom class indices are
local to their respective models. The custom launch explicitly translates its native
15-class output into `ihmc_custom_yolov8_label_space.yaml`, which extends the
73-class Alex space. The translation:

- discards `robot_hand` before scene-graph integration;
- maps `person` to the existing Alex label while retaining `person_operator`
  as a distinct dynamic class for designated personnel wearing a vest;
- maps `bottle` and `trash_can` to their existing Alex equivalents; and
- retains the door components and other IHMC classes under distinct IDs.

For door semantics, `door` denotes the complete assembly, `door_panel` denotes
the movable panel, and `door_frame` denotes the fixed structure and preferred
navigation reference. The current custom model detects `door_panel` and door
hardware but does not detect `door_frame`. Consequently, it cannot yet create
a stable composite `door` object from vision alone. A future model must detect
the frame, after which spatial association can attach the panel and hardware to
the stable door assembly.

The no-argument open-vocabulary launch selects the Alex label-space and grouping
files for the prebuilt Alex engines. Any replacement fixed-class engine must be
exported with the same ordered class list, or its matching label-space and
grouping configuration must be selected.

Segmentation artifacts are organized by backend:

```text
$SCENE_GRAPH_ASSETS/models/segmentation/
├── pretrained/
│   ├── yoloe/
│   ├── yolov8/
│   ├── yolow/
│   ├── fastsam/
│   └── ade20k/
└── custom_trained/
    ├── yoloe/
    ├── yolov8/
    │   └── MODEL_RELEASE/
    │       ├── MODEL_RELEASE.engine
    │       ├── MODEL_RELEASE.engine.sha256
    │       ├── class_names.yaml
    │       └── class_names.yaml.sha256
    └── yolow/
```

`pretrained` contains upstream checkpoints and artifacts derived from them.
Embedding a deployment label list during export does not make a model
custom-trained. `custom_trained` is reserved for checkpoints trained or
fine-tuned on an organization-specific dataset and their derived artifacts.
Each custom release has its own immutable directory so a new training run does
not silently replace an older model or checksum. Select the active release in
`semantic_inference_ros/config/yolov8_custom_segmentation.yaml`.
Keep the corresponding class-order YAML under
`semantic_inference/config/label_groupings/` synchronized with every
fixed-vocabulary model so its output indices remain auditable.

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

##### Fixed-vocabulary YOLOv8 development workflow

The following steps are for model developers and debugging. A normal runtime
installation should use already prepared artifacts.

For a newly trained custom model, begin with one immutable release directory.
The directory name and ONNX basename must identify the training release:

```text
$SCENE_GRAPH_ASSETS/models/segmentation/custom_trained/yolov8/
└── best_multi_08_22_2026/
    ├── best_multi_08_22_2026.onnx
    └── class_names.yaml
```

Verify that `class_names.yaml` lists classes in exactly the same index order as
the ONNX output. Then build the engine on the target NVIDIA deployment class:

```bash
.venv-tensorrt-export/bin/python \
  src/semantic_inference_ros/semantic_inference_python/scripts/build_tensorrt_engine.py \
  --onnx "$SCENE_GRAPH_ASSETS/models/segmentation/custom_trained/yolov8/best_multi_08_22_2026/best_multi_08_22_2026.onnx" \
  --output "$SCENE_GRAPH_ASSETS/models/segmentation/custom_trained/yolov8/best_multi_08_22_2026/best_multi_08_22_2026.engine" \
  --fp16 \
  --workspace-gib 4

cd "$SCENE_GRAPH_ASSETS/models/segmentation/custom_trained/yolov8/best_multi_08_22_2026"
sha256sum best_multi_08_22_2026.engine \
  > best_multi_08_22_2026.engine.sha256
sha256sum class_names.yaml > class_names.yaml.sha256
```

The completed runtime release is:

```text
best_multi_08_22_2026/
├── best_multi_08_22_2026.onnx
├── best_multi_08_22_2026.engine
├── best_multi_08_22_2026.engine.sha256
├── class_names.yaml
└── class_names.yaml.sha256
```

Publish the engine, class file, and both checksum sidecars as one release.
Before selecting the release, copy its class order into a reviewed YAML under
`semantic_inference/config/label_groupings/`, update the native-to-Hydra label
mapping if any class or index changed, and point
`semantic_inference_ros/config/yolov8_custom_segmentation.yaml` at the new
engine. Rebuild `semantic_inference` and `semantic_inference_ros` after changing
these source-controlled files. The private IHMC integration README describes
publishing the engine and registering its Drive IDs for automatic provisioning.

For an upstream PyTorch checkpoint, retain the `.pt` source and export a
static-shape FP32 ONNX model. This example uses a 640-by-640 input:

```bash
export SCENE_GRAPH_ASSETS="${SCENE_GRAPH_ASSETS:-$PWD/scene_graph_assets}"

.venv-tensorrt-export/bin/python -c "from ultralytics import YOLO; YOLO('$SCENE_GRAPH_ASSETS/models/segmentation/pretrained/yolov8/yolov8l-seg.pt').export(format='onnx', imgsz=640, batch=1, dynamic=False, simplify=True, opset=19, device='cpu')"
```

Convert the resulting ONNX model to an FP16 TensorRT engine on the target
NVIDIA platform:

```bash
.venv-tensorrt-export/bin/python \
  src/semantic_inference_ros/semantic_inference_python/scripts/build_tensorrt_engine.py \
  --onnx "$SCENE_GRAPH_ASSETS/models/segmentation/pretrained/yolov8/yolov8l-seg.onnx" \
  --output "$SCENE_GRAPH_ASSETS/models/segmentation/pretrained/yolov8/yolov8l-seg.engine" \
  --fp16 \
  --workspace-gib 4
```

When training has already produced an ONNX model, skip the `.pt`-to-ONNX step
and build directly from that file:

```bash
.venv-tensorrt-export/bin/python \
  src/semantic_inference_ros/semantic_inference_python/scripts/build_tensorrt_engine.py \
  --onnx "$SCENE_GRAPH_ASSETS/models/segmentation/custom_trained/yolov8/MODEL/MODEL.onnx" \
  --output "$SCENE_GRAPH_ASSETS/models/segmentation/custom_trained/yolov8/MODEL/MODEL.engine" \
  --fp16 \
  --workspace-gib 4
```

For TensorRT 11, the builder converts the graph to FP16 in memory, builds a
strongly typed network, and embeds the ONNX metadata in the engine wrapper.
The source ONNX file remains unchanged. Preserving metadata is required by the
Ultralytics runtime to recover the segmentation task, class names, image size,
and output-decoding configuration.

The original Hydra closed-vocabulary C++ backend accepts ONNX models that
directly output one integer label per pixel. YOLOv8-seg does not have that
interface: it outputs detections, mask coefficients, and mask prototypes.
Consequently, YOLOv8-seg artifacts must run through the YOLO Python wrapper,
which performs decoding, confidence filtering, NMS, and panoptic-mask
construction before publishing results to Hydra.

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
