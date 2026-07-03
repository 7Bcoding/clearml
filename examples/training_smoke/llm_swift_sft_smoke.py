import argparse
import json
import os
import subprocess
from pathlib import Path

from clearml import Task


def add_arg(command: list[str], flag: str, value) -> None:
    if value is None or value == "":
        return
    command.extend([flag, str(value)])


def main() -> None:
    parser = argparse.ArgumentParser(description="ClearML smoke wrapper for ms-swift SFT/LoRA.")
    parser.add_argument("--queue", default="volcano-queue")
    parser.add_argument("--docker-image", default="ms-swift-cuda12.1:latest")
    parser.add_argument("--clearml-project", default="training-smoke/llm")
    parser.add_argument("--clearml-task-name", default="swift-sft-lora-smoke")

    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="/data/output/clearml-swift-sft-smoke")
    parser.add_argument("--run-name", default="clearml-swift-sft-smoke")
    parser.add_argument("--tuner-type", default="lora")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--num-train-epochs", type=float, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", default="1e-5")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--target-modules", default="all-linear")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--split-dataset-ratio", default="0.01")
    parser.add_argument("--save-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--agent-template", default="")
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--upload-output-dir", action="store_true")

    parser.add_argument("--vgpu-number", type=int, default=1)
    parser.add_argument("--vgpu-memory", type=int, default=24)
    parser.add_argument("--vgpu-cores", type=int, default=100)
    args = parser.parse_args()

    reqs = Path(__file__).with_name("requirements-smoke.txt")
    Task.force_requirements_env_freeze(force=True, requirements_file=str(reqs))

    task = Task.init(project_name=args.clearml_project, task_name=args.clearml_task_name)
    task.set_tags(["smoke", "llm", "ms-swift", "volcano-vgpu"])
    task.set_base_docker(args.docker_image)
    task.connect(
        {
            "vgpu_number": args.vgpu_number,
            "vgpu_memory": args.vgpu_memory,
            "vgpu_cores": args.vgpu_cores,
        },
        name="VGPU",
    )
    task.execute_remotely(queue_name=args.queue)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("MAX_PIXELS", "1003520")
    os.environ.setdefault("VIDEO_MAX_PIXELS", "50176")
    os.environ.setdefault("FPS_MAX_FRAMES", "12")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = ["swift", "sft"]
    add_arg(command, "--model", args.model)
    add_arg(command, "--dataset", args.dataset)
    add_arg(command, "--output_dir", args.output_dir)
    add_arg(command, "--run_name", args.run_name)
    add_arg(command, "--tuner_type", args.tuner_type)
    add_arg(command, "--torch_dtype", args.torch_dtype)
    add_arg(command, "--max_steps", args.max_steps)
    add_arg(command, "--num_train_epochs", args.num_train_epochs)
    add_arg(command, "--per_device_train_batch_size", args.per_device_train_batch_size)
    add_arg(command, "--per_device_eval_batch_size", args.per_device_eval_batch_size)
    add_arg(command, "--gradient_accumulation_steps", args.gradient_accumulation_steps)
    add_arg(command, "--learning_rate", args.learning_rate)
    add_arg(command, "--lora_rank", args.lora_rank)
    add_arg(command, "--lora_alpha", args.lora_alpha)
    add_arg(command, "--target_modules", args.target_modules)
    add_arg(command, "--max_length", args.max_length)
    add_arg(command, "--split_dataset_ratio", args.split_dataset_ratio)
    add_arg(command, "--save_steps", args.save_steps)
    add_arg(command, "--eval_steps", args.eval_steps)
    add_arg(command, "--logging_steps", args.logging_steps)
    add_arg(command, "--save_total_limit", args.save_total_limit)
    add_arg(command, "--gradient_checkpointing", "true")
    if args.agent_template:
        add_arg(command, "--agent_template", args.agent_template)
    if args.extra_args:
        command.extend(args.extra_args.split())

    print("[swift-smoke] running:")
    print(" ".join(command))
    subprocess.run(command, check=True)

    manifest = {
        "model": args.model,
        "dataset": args.dataset,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "vgpu_number": args.vgpu_number,
        "vgpu_memory": args.vgpu_memory,
        "vgpu_cores": args.vgpu_cores,
    }
    manifest_path = output_dir / "clearml_smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    task.upload_artifact("training-output-manifest", artifact_object=str(manifest_path))

    if args.upload_output_dir:
        task.upload_artifact("swift-output-dir", artifact_object=str(output_dir))


if __name__ == "__main__":
    main()
