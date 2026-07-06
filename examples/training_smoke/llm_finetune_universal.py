import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from clearml import Task


def str_to_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def add_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    command.extend([flag, str(value)])


def load_extra_json(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    path = Path(raw)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(raw)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except OSError:
        # Fallback: use the source path directly in dataset_info.json. This is
        # accepted in many setups, but symlinking is preferred for portability.
        pass


def build_dataset_info(args: argparse.Namespace, work_dir: Path) -> tuple[str, str]:
    """Return (dataset_name, dataset_dir) for LLaMA-Factory."""
    if not args.dataset_path:
        if not args.dataset or not args.dataset_dir:
            raise ValueError("llama-factory backend requires --dataset + --dataset-dir, or --dataset-path")
        return args.dataset, args.dataset_dir

    dataset_path = Path(args.dataset_path)
    dataset_name = args.dataset_name or "clearml_dataset"
    dataset_dir = work_dir / "llamafactory_dataset"
    linked_path = dataset_dir / dataset_path.name
    safe_symlink(dataset_path, linked_path)

    file_name = linked_path.name if linked_path.exists() or linked_path.is_symlink() else str(dataset_path)
    entry: dict[str, Any] = {
        "file_name": file_name,
        "formatting": args.dataset_format,
    }

    if args.dataset_format == "sharegpt":
        entry["columns"] = {
            "messages": args.messages_column,
            "system": args.system_column,
            "tools": args.tools_column,
        }
        if args.dataset_openai_messages:
            entry["columns"] = {"messages": args.messages_column}
            entry["tags"] = {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            }
    else:
        entry["columns"] = {
            "prompt": args.prompt_column,
            "query": args.query_column,
            "response": args.response_column,
            "system": args.system_column,
            "history": args.history_column,
        }

    if args.ranking:
        entry["ranking"] = True
        if args.dataset_format == "sharegpt":
            entry["columns"].update({"chosen": args.chosen_column, "rejected": args.rejected_column})
        else:
            entry["columns"].update(
                {
                    "chosen": args.chosen_column,
                    "rejected": args.rejected_column,
                }
            )

    dataset_info = {dataset_name: entry}
    write_json(dataset_dir / "dataset_info.json", dataset_info)
    return dataset_name, str(dataset_dir)


def llama_factory_config(args: argparse.Namespace, work_dir: Path) -> tuple[Path, list[str]]:
    dataset_name, dataset_dir = build_dataset_info(args, work_dir)
    config: dict[str, Any] = {
        "model_name_or_path": args.model_path,
        "stage": args.train_method,
        "do_train": True,
        "finetuning_type": args.finetuning_type,
        "dataset": dataset_name,
        "dataset_dir": dataset_dir,
        "template": args.template,
        "cutoff_len": args.max_length,
        "overwrite_cache": True,
        "preprocessing_num_workers": args.preprocessing_num_workers,
        "output_dir": args.output_dir,
        "overwrite_output_dir": args.overwrite_output_dir,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "plot_loss": True,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": float(args.learning_rate),
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": float(args.warmup_ratio),
        "report_to": args.report_to,
    }

    if args.max_steps > 0:
        config["max_steps"] = args.max_steps
    else:
        config["num_train_epochs"] = args.num_train_epochs

    if args.max_samples > 0:
        config["max_samples"] = args.max_samples

    if args.finetuning_type in {"lora", "qlora"}:
        config["lora_rank"] = args.lora_rank
        config["lora_alpha"] = args.lora_alpha
        config["lora_dropout"] = args.lora_dropout
        config["lora_target"] = args.lora_target

    if args.finetuning_type == "qlora":
        config["quantization_bit"] = args.quantization_bit

    if args.dtype == "bfloat16":
        config["bf16"] = True
    elif args.dtype == "float16":
        config["fp16"] = True

    config.update(load_extra_json(args.extra_backend_config_json))
    config_path = work_dir / "llamafactory_train.yaml"
    write_json(config_path, config)

    command = ["llamafactory-cli", "train", str(config_path)]
    if args.extra_args:
        command.extend(shlex.split(args.extra_args))
    return config_path, command


def ms_swift_command(args: argparse.Namespace) -> list[str]:
    command = ["swift", "sft"]
    add_arg(command, "--model", args.model_path)
    add_arg(command, "--dataset", args.dataset_path or args.dataset)
    add_arg(command, "--output_dir", args.output_dir)
    add_arg(command, "--run_name", args.run_name)
    add_arg(command, "--tuner_type", args.finetuning_type)
    add_arg(command, "--torch_dtype", "bfloat16" if args.dtype == "bfloat16" else args.dtype)
    if args.max_steps > 0:
        add_arg(command, "--max_steps", args.max_steps)
    add_arg(command, "--num_train_epochs", args.num_train_epochs)
    add_arg(command, "--per_device_train_batch_size", args.per_device_train_batch_size)
    add_arg(command, "--per_device_eval_batch_size", args.per_device_eval_batch_size)
    add_arg(command, "--gradient_accumulation_steps", args.gradient_accumulation_steps)
    add_arg(command, "--learning_rate", args.learning_rate)
    add_arg(command, "--lora_rank", args.lora_rank)
    add_arg(command, "--lora_alpha", args.lora_alpha)
    add_arg(command, "--target_modules", args.lora_target)
    add_arg(command, "--max_length", args.max_length)
    add_arg(command, "--split_dataset_ratio", args.val_size)
    add_arg(command, "--save_steps", args.save_steps)
    add_arg(command, "--eval_steps", args.eval_steps)
    add_arg(command, "--logging_steps", args.logging_steps)
    add_arg(command, "--save_total_limit", args.save_total_limit)
    add_arg(command, "--gradient_checkpointing", str(args.gradient_checkpointing).lower())
    if args.extra_args:
        command.extend(shlex.split(args.extra_args))
    return command


def custom_command(args: argparse.Namespace) -> list[str]:
    if not args.custom_command:
        raise ValueError("--custom-command is required when --backend custom")
    rendered = args.custom_command.format(
        model_path=args.model_path,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    return ["bash", "-lc", rendered]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal ClearML LLM fine-tuning template.")

    parser.add_argument("--queue", default="llm-finetune-vgpu")
    parser.add_argument("--docker-image", default="harbor.example.com/ai/llamafactory:latest")
    parser.add_argument("--clearml-project", default="training-template/llm")
    parser.add_argument("--clearml-task-name", default="llm-finetune-universal")
    parser.add_argument("--backend", choices=["llama-factory", "ms-swift", "custom"], default="llama-factory")
    parser.add_argument(
        "--store-standalone-script",
        type=str_to_bool,
        default=True,
        help="Store this script in the ClearML Task so the remote Agent does not need to clone a Git repository.",
    )

    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--dataset-name", default="clearml_dataset")
    parser.add_argument("--output-dir", default="/data/output/llm-finetune-universal")
    parser.add_argument("--run-name", default="llm-finetune-universal")

    parser.add_argument("--train-method", default="sft")
    parser.add_argument("--finetuning-type", choices=["lora", "qlora", "full", "freeze"], default="lora")
    parser.add_argument("--template", default="qwen")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", default="1e-5")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cutoff-len", type=int, default=None)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", default="0.05")
    parser.add_argument("--val-size", default="0.01")
    parser.add_argument("--save-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--preprocessing-num-workers", type=int, default=4)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--overwrite-output-dir", type=str_to_bool, default=True)
    parser.add_argument("--gradient-checkpointing", type=str_to_bool, default=True)

    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target", default="all")
    parser.add_argument("--quantization-bit", type=int, default=4)

    parser.add_argument("--dataset-format", choices=["alpaca", "sharegpt"], default="alpaca")
    parser.add_argument("--dataset-openai-messages", action="store_true")
    parser.add_argument("--ranking", action="store_true")
    parser.add_argument("--prompt-column", default="instruction")
    parser.add_argument("--query-column", default="input")
    parser.add_argument("--response-column", default="output")
    parser.add_argument("--history-column", default="history")
    parser.add_argument("--messages-column", default="conversations")
    parser.add_argument("--system-column", default="system")
    parser.add_argument("--tools-column", default="tools")
    parser.add_argument("--chosen-column", default="chosen")
    parser.add_argument("--rejected-column", default="rejected")

    parser.add_argument("--extra-backend-config-json", default="")
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--custom-command", default="")
    parser.add_argument("--upload-output-dir", action="store_true")

    parser.add_argument("--vgpu-number", type=int, default=1)
    parser.add_argument("--vgpu-memory", type=int, default=24)
    parser.add_argument("--vgpu-cores", type=int, default=100)
    return parser


def main() -> None:
    reqs = Path(__file__).with_name("requirements-smoke.txt")
    if reqs.exists():
        Task.force_requirements_env_freeze(force=True, requirements_file=str(reqs))
    else:
        Task.force_requirements_env_freeze(force=True)

    parser = build_parser()
    pre_args, _ = parser.parse_known_args()
    if pre_args.store_standalone_script:
        if not hasattr(Task, "force_store_standalone_script"):
            raise RuntimeError(
                "This ClearML SDK does not support Task.force_store_standalone_script. "
                "Please upgrade clearml, or use an internal Git repository reachable by training Pods."
            )
        Task.force_store_standalone_script(True)
    task = Task.init(project_name=pre_args.clearml_project, task_name=pre_args.clearml_task_name)
    args = parser.parse_args()
    if args.cutoff_len is not None:
        args.max_length = args.cutoff_len

    task.set_tags(["llm", "finetune", args.backend, "volcano-vgpu"])
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
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", "/data/cache/huggingface")
    os.environ.setdefault("MODELSCOPE_CACHE", "/data/cache/modelscope")

    output_dir = Path(args.output_dir)
    work_dir = output_dir / "_clearml_template"
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "llama-factory":
        config_path, command = llama_factory_config(args, work_dir)
        task.upload_artifact("llamafactory-train-config", artifact_object=str(config_path))
        dataset_info = work_dir / "llamafactory_dataset" / "dataset_info.json"
        if dataset_info.exists():
            task.upload_artifact("llamafactory-dataset-info", artifact_object=str(dataset_info))
    elif args.backend == "ms-swift":
        command = ms_swift_command(args)
    else:
        command = custom_command(args)

    manifest = {
        "backend": args.backend,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "dataset_dir": args.dataset_dir,
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "command": command,
        "vgpu": {
            "vgpu_number": args.vgpu_number,
            "vgpu_memory": args.vgpu_memory,
            "vgpu_cores": args.vgpu_cores,
        },
    }
    manifest_path = work_dir / "clearml_llm_finetune_manifest.json"
    write_json(manifest_path, manifest)
    task.upload_artifact("llm-finetune-manifest", artifact_object=str(manifest_path))

    print("[llm-finetune] backend:", args.backend)
    print("[llm-finetune] running:")
    print(" ".join(shlex.quote(item) for item in command))
    subprocess.run(command, check=True)

    if args.upload_output_dir:
        task.upload_artifact("llm-finetune-output-dir", artifact_object=str(output_dir))


if __name__ == "__main__":
    main()
