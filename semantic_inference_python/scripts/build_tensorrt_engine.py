#!/usr/bin/env python3

# Copyright (c) 2026, IHMC Robotics Lab.
# All rights reserved.

"""Build a TensorRT engine from a static-shape ONNX model."""

import argparse
import json
from pathlib import Path

import tensorrt as trt


LOGGER = trt.Logger(trt.Logger.INFO)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True, help="Input ONNX model.")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output TensorRT engine."
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable FP16 kernels when supported (enabled by default).",
    )
    parser.add_argument(
        "--workspace-gib",
        type=float,
        default=4.0,
        help="TensorRT workspace limit in GiB (default: 4).",
    )
    return parser.parse_args()


def main() -> None:
    """Parse the ONNX graph and serialize a TensorRT engine."""
    args = parse_args()
    onnx_path = args.onnx.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {onnx_path}")
    if args.workspace_gib <= 0:
        raise ValueError("--workspace-gib must be positive")

    builder = trt.Builder(LOGGER)
    fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
    strongly_typed_fp16 = args.fp16 and fp16_flag is None

    # Explicit batch is unconditional in newer TensorRT releases, where the
    # legacy EXPLICIT_BATCH enum was removed.
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    network_flags = 0 if explicit_batch is None else 1 << int(explicit_batch)
    if strongly_typed_fp16:
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, LOGGER)

    import onnx

    onnx_model = onnx.load(str(onnx_path))
    metadata = {item.key: item.value for item in onnx_model.metadata_props}

    if strongly_typed_fp16:
        from modelopt.onnx.autocast import convert_to_f16

        print("TensorRT 11+ detected; converting the ONNX graph to FP16 in memory.")
        onnx_model = convert_to_f16(onnx_model, keep_io_types=True)
        parsed = parser.parse(onnx_model.SerializeToString())
    else:
        parsed = parser.parse_from_file(str(onnx_path))

    if not parsed:
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse '{onnx_path}':\n{errors}")

    dynamic_inputs = []
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        if any(dimension < 0 for dimension in tensor.shape):
            dynamic_inputs.append(f"{tensor.name}: {tuple(tensor.shape)}")
    if dynamic_inputs:
        inputs = "\n".join(dynamic_inputs)
        raise ValueError(
            "This builder requires static ONNX input shapes. Dynamic inputs:\n"
            f"{inputs}"
        )

    config = builder.create_builder_config()
    workspace_bytes = int(args.workspace_gib * (1 << 30))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

    if args.fp16 and fp16_flag is not None:
        # TensorRT 11 removed both Builder.platform_has_fast_fp16 and the FP16
        # builder flag. This legacy path is only used by older releases.
        has_fast_fp16 = getattr(builder, "platform_has_fast_fp16", None)
        if has_fast_fp16 is None or has_fast_fp16:
            config.set_flag(fp16_flag)
        else:
            LOGGER.log(
                trt.Logger.WARNING,
                "FP16 was requested, but this platform has no fast FP16 support; "
                "building without the FP16 flag.",
            )

    print(f"TensorRT version: {trt.__version__}")
    print(f"ONNX model: {onnx_path}")
    print(f"Output engine: {output_path}")
    print(f"Workspace: {args.workspace_gib:g} GiB")
    print(f"FP16 requested: {args.fp16}")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT failed to build the serialized engine")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_stream:
        # Ultralytics prefixes TensorRT plans with their JSON metadata. Preserve
        # ONNX metadata so its runtime can identify segmentation outputs and
        # recover the trained class names. Plain TensorRT runtimes should use
        # engines without this wrapper.
        if metadata:
            encoded_metadata = json.dumps(metadata).encode("utf-8")
            output_stream.write(
                len(encoded_metadata).to_bytes(4, byteorder="little", signed=True)
            )
            output_stream.write(encoded_metadata)
        output_stream.write(serialized_engine)
    print(f"Saved TensorRT engine: {output_path}")


if __name__ == "__main__":
    main()
