import argparse
import subprocess
from pathlib import Path

from clearml import OutputModel, Task


def write_fibocv_yaml(path: Path, args: argparse.Namespace) -> None:
    classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    class_expr = "[" + ", ".join(repr(item) for item in classes) + "]"
    lines = [
        f"project_name: {args.fibocv_project_name}",
        f"model_type: {args.model_type}",
        f"classes: {class_expr}",
        f"train_img_path: {args.train_img_path}",
        f"train_ann_path: {args.train_ann_path}",
        f"val_img_path: {args.val_img_path}",
        f"val_ann_path: {args.val_ann_path}",
        f"total_epochs: {args.total_epochs}",
        "gpu_id: auto",
        f"load_from: {args.load_from}",
    ]
    if args.input_size:
        lines.extend(["extra:", f"  input_size: [{args.input_size}]"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def upload_if_exists(task: Task, name: str, path: Path) -> None:
    if path.exists():
        task.upload_artifact(name, artifact_object=str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="ClearML smoke wrapper for FiboCV training.")
    parser.add_argument("--queue", default="volcano-queue")
    parser.add_argument("--docker-image", default="fibo_cv_train:latest")
    parser.add_argument("--clearml-project", default="training-smoke/fibocv")
    parser.add_argument("--clearml-task-name", default="fibocv-voc-smoke")
    parser.add_argument("--workspace", default="/app/fibo_cv_train/workspace/clearml_smoke")
    parser.add_argument("--entry-cmd", default="python /app/fibo_cv_train/tools/train_dispatcher.py {config_path}")
    parser.add_argument("--use-voc-example", action="store_true", default=True)
    parser.add_argument("--no-use-voc-example", action="store_false", dest="use_voc_example")
    parser.add_argument("--config-path", default="")

    parser.add_argument("--fibocv-project-name", default="clearml_fibocv_smoke")
    parser.add_argument("--model-type", default="fibodet_light")
    parser.add_argument("--classes", default="person")
    parser.add_argument("--train-img-path", default="/data/cv_dataset/train/JPEGImages")
    parser.add_argument("--train-ann-path", default="/data/cv_dataset/train/annotations.json")
    parser.add_argument("--val-img-path", default="/data/cv_dataset/val/JPEGImages")
    parser.add_argument("--val-ann-path", default="/data/cv_dataset/val/annotations.json")
    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--load-from", default="None")
    parser.add_argument("--input-size", default="")

    parser.add_argument("--vgpu-number", type=int, default=1)
    parser.add_argument("--vgpu-memory", type=int, default=4)
    parser.add_argument("--vgpu-cores", type=int, default=30)
    args = parser.parse_args()

    reqs = Path(__file__).with_name("requirements-smoke.txt")
    Task.force_requirements_env_freeze(force=True, requirements_file=str(reqs))

    task = Task.init(project_name=args.clearml_project, task_name=args.clearml_task_name)
    task.set_tags(["smoke", "fibocv", "volcano-vgpu"])
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

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    if args.config_path:
        config_path = Path(args.config_path)
    elif args.use_voc_example:
        config_path = Path("/app/fibo_cv_train/configs/user_configs/voc_example.yaml")
    else:
        config_path = workspace / "user_config.yaml"
        write_fibocv_yaml(config_path, args)

    command = args.entry_cmd.format(config_path=str(config_path))
    print(f"[fibocv-smoke] running: {command}")
    subprocess.run(["bash", "-lc", command], check=True)

    upload_if_exists(task, "fibocv-work-dirs", Path("/app/fibo_cv_train/workspace/work_dirs"))
    upload_if_exists(task, "fibocv-onnx", Path("/app/fibo_cv_train/workspace/onnx"))
    upload_if_exists(task, "fibocv-auto-generated-configs", Path("/app/fibo_cv_train/workspace/auto_generated_configs"))

    onnx_files = sorted(Path("/app/fibo_cv_train/workspace").glob("onnx/**/*.onnx"))
    if onnx_files:
        output_model = OutputModel(task=task, framework="onnx", name="fibocv-onnx-smoke")
        output_model.update_weights(str(onnx_files[-1]))
        task.get_logger().report_text(f"registered onnx model: {onnx_files[-1]}")
    else:
        task.get_logger().report_text("no onnx file found under workspace/onnx")


if __name__ == "__main__":
    main()
