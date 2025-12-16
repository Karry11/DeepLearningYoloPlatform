import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr
import pandas as pd

from app.models import (
    DatasetConfig,
    DatasetValidationResult,
    ModelConfig,
    RuntimePaths,
    TaskConfig,
    TaskRecord,
    TaskStatus,
    TrainParams,
)
from app.services import dataset_validator, resource_checker
from app.services.deepseek_client import DeepSeekClient
from app.services.log_manager import contains_error, tail_log
from app.services.metrics_parser import parse_results_csv
from app.services.run_executor import RunExecutor
from app.services.task_manager import TaskManager


BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
PROJECTS_DIR = STORAGE_DIR / "projects"
TASKS_DB = STORAGE_DIR / "tasks.json"

task_manager = TaskManager(TASKS_DB)
executor = RunExecutor()
deepseek = DeepSeekClient()


def _runtime_paths(task_id: str) -> RuntimePaths:
    project_dir = PROJECTS_DIR / task_id
    results_dir = project_dir / task_id
    return RuntimePaths(
        project_dir=str(project_dir),
        log_path=str(project_dir / "train.log"),
        script_path=str(project_dir / f"train_v1.py"),
        weights_output_dir=str(results_dir / "weights"),
        results_path=str(results_dir / "results.csv"),
    )


def _build_task_config(
    dataset_root: str,
    data_yaml: Optional[str],
    task_type: str,
    model_family: str,
    model_task: str,
    model_size: str,
    pretrained_weights: Optional[str],
    device: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    lr0: float,
    workers: int,
    lrf: Optional[float] = None,
    momentum: Optional[float] = None,
    save_period: Optional[int] = None,
    seed: Optional[int] = None,
) -> TaskConfig:
    task_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S-") + str(random.randint(1000, 9999))
    runtime = _runtime_paths(task_id)
    dataset_cfg = DatasetConfig(
        root=dataset_root,
        data_yaml=data_yaml,
        task_type=task_type,  # type: ignore[arg-type]
    )
    model_cfg = ModelConfig(
        family=model_family,  # type: ignore[arg-type]
        task=model_task,  # type: ignore[arg-type]
        size=model_size,  # type: ignore[arg-type]
        pretrained_weights=pretrained_weights,
        device=device,
    )
    train_params = TrainParams(
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz,
        lr0=lr0,
        workers=workers,
        lrf=lrf,
        momentum=momentum,
        save_period=save_period,
        seed=seed,
        project=runtime.project_dir,
        name=task_id,
    )
    return TaskConfig(
        task_id=task_id,
        dataset=dataset_cfg,
        model=model_cfg,
        train_params=train_params,
        runtime=runtime,
    )


def _persist_script(task: TaskConfig, script: str) -> None:
    path = Path(task.runtime.script_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script, encoding="utf-8")


def _maybe_locate_results(task: TaskRecord) -> Optional[Path]:
    results_path = Path(task.config.runtime.results_path)
    if results_path.exists():
        return results_path
    project_dir = Path(task.config.runtime.project_dir)
    for candidate in project_dir.rglob("results.csv"):
        return candidate
    return None


def _sync_status(task: TaskRecord) -> TaskRecord:
    rc = executor.poll(task.task_id)
    if rc is None:
        return task
    # Process finished; update status if needed.
    if task.status in {TaskStatus.RUNNING, TaskStatus.RETRYING, TaskStatus.PENDING}:
        new_status = TaskStatus.SUCCEEDED if rc == 0 else TaskStatus.FAILED
        err_msg = None
        log_tail = tail_log(Path(task.config.runtime.log_path))
        if rc != 0 or contains_error(log_tail.lines):
            err_msg = " ".join(log_tail.lines[-5:]) if log_tail.lines else "Unknown error"
        updated = task_manager.update_status(task.task_id, new_status, error_message=err_msg)
        return updated or task
    return task


def list_tasks_ui(status_filter: Optional[str]) -> List[Dict]:
    status_list = [TaskStatus(status_filter)] if status_filter else None
    tasks = task_manager.list_tasks(status_list)
    rows: List[Dict] = []
    for task in tasks:
        task = _sync_status(task)
        cfg = task.config
        rows.append(
            {
                "task_id": task.task_id,
                "status": task.status.value,
                "model": cfg.model.model_name,
                "dataset": cfg.dataset.root,
                "created_at": task.created_at.isoformat(),
                "updated_at": task.updated_at.isoformat(),
                "retries": task.retries,
            }
        )
    return rows


def run_env_check() -> str:
    summary = resource_checker.summarize_environment()
    lines = [
        f"CPU: {summary['cpu']}",
        f"GPU: {summary['gpu']}",
        f"CUDA: {summary['cuda']}",
        f"Python: {summary['python']}",
    ]
    if summary.get("warnings"):
        lines.append(f"Warnings: {summary['warnings']}")
    return "\n".join(lines)


def run_dataset_validation(
    dataset_root: str,
    data_yaml: str,
    task_type: str,
    train_dir: str,
    val_dir: str,
    test_dir: str,
    labels_dir: str,
) -> str:
    cfg = DatasetConfig(
        root=dataset_root,
        data_yaml=data_yaml or None,
        task_type=task_type,  # type: ignore[arg-type]
        train_dir=train_dir or None,
        val_dir=val_dir or None,
        test_dir=test_dir or None,
        labels_dir=labels_dir or None,
    )
    result: DatasetValidationResult = dataset_validator.validate_dataset(cfg)
    lines = [f"Validation {'PASSED' if result.ok else 'FAILED'}"]
    for msg in result.messages:
        prefix = msg.level.upper()
        lines.append(f"[{prefix}] {msg.text}")
    if result.stats:
        lines.append(
            f"Train {result.stats.train_images}, Val {result.stats.val_images}, Test {result.stats.test_images}"
        )
        lines.append(f"Classes: {result.stats.classes}, Names: {result.stats.names}")
    return "\n".join(lines)


def start_task(
    dataset_root: str,
    data_yaml: str,
    task_type: str,
    model_family: str,
    model_task: str,
    model_size: str,
    pretrained_weights: str,
    device: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    lr0: float,
    workers: int,
    lrf: float,
    momentum: float,
    save_period: int,
    seed: int,
) -> str:
    cfg = _build_task_config(
        dataset_root=dataset_root,
        data_yaml=data_yaml or None,
        task_type=task_type,
        model_family=model_family,
        model_task=model_task,
        model_size=model_size,
        pretrained_weights=pretrained_weights or None,
        device=device or "0",
        epochs=int(epochs),
        batch_size=int(batch_size),
        imgsz=int(imgsz),
        lr0=float(lr0),
        workers=int(workers),
        lrf=float(lrf) if lrf else None,
        momentum=float(momentum) if momentum else None,
        save_period=int(save_period) if save_period else None,
        seed=int(seed) if seed else None,
    )
    validation = dataset_validator.validate_dataset(cfg.dataset)
    if not validation.ok:
        details = "; ".join([m.text for m in validation.messages])
        return f"Dataset validation failed: {details}"

    record = task_manager.create_task(cfg)
    env_summary = resource_checker.summarize_environment().get("as_json", "")
    script = deepseek.generate_script(cfg, env_summary)
    _persist_script(cfg, script)

    try:
        executor.start(record)
        task_manager.update_status(record.task_id, TaskStatus.RUNNING)
    except Exception as exc:
        task_manager.update_status(record.task_id, TaskStatus.FAILED, error_message=str(exc))
        return f"Failed to start task: {exc}"
    return f"Task {record.task_id} started."


def _metrics_to_df(metrics: Dict[str, List[Dict[str, float]]]):
    rows: List[Dict] = []
    for name, series in metrics.items():
        for point in series:
            rows.append({"metric": name, "epoch": point["x"], "value": point["y"]})
    return pd.DataFrame(rows, columns=["metric", "epoch", "value"])


def get_task_detail(task_id: str):
    task = task_manager.get(task_id)
    if not task:
        empty_df = pd.DataFrame(columns=["metric", "epoch", "value"])
        return "Task not found", "", {}, empty_df
    task = _sync_status(task)
    status = f"Status: {task.status.value}; Retries: {task.retries}"
    log_tail = tail_log(Path(task.config.runtime.log_path)).lines
    log_text = "\n".join(log_tail)
    metrics_path = _maybe_locate_results(task)
    metrics = parse_results_csv(metrics_path) if metrics_path else {}
    metrics_df = _metrics_to_df(metrics)
    return status, log_text, metrics, metrics_df


def cancel_task(task_id: str) -> str:
    ok = executor.cancel(task_id)
    if ok:
        task_manager.update_status(task_id, TaskStatus.CANCELLED)
        return f"Task {task_id} cancelled."
    return "No running process found."


def build_ui():
    with gr.Blocks(theme="base", css="") as demo:
        gr.Markdown("# YOLO 训练平台（Ultralytics）")
        with gr.Tab("任务列表"):
            status_filter = gr.Dropdown(
                [s.value for s in TaskStatus], label="状态筛选", value=None, allow_custom_value=True
            )
            refresh = gr.Button("刷新任务列表")
            tasks_grid = gr.Dataframe(
                headers=["task_id", "status", "model", "dataset", "created_at", "updated_at", "retries"],
                datatype=["str", "str", "str", "str", "str", "str", "number"],
                interactive=False,
                wrap=True,
                elem_id="tasks-grid",
            )
            refresh.click(list_tasks_ui, inputs=[status_filter], outputs=[tasks_grid])

        with gr.Tab("新建任务"):
            gr.Markdown("## 环境检测")
            env_btn = gr.Button("检测环境")
            env_output = gr.Textbox(label="检测结果", lines=6)
            env_btn.click(run_env_check, outputs=env_output)

            gr.Markdown("## 数据集")
            dataset_root = gr.Textbox(
                label="数据集根目录",
                value="/mnt/e/IDE/AI_Project/yolo_platform/datasets",
            )
            data_yaml = gr.Textbox(
                label="data.yaml 路径（推荐）",
                value="/mnt/e/IDE/AI_Project/yolo_platform/datasets/data.yaml",
            )
            task_type = gr.Dropdown(["detect", "segment"], label="任务类型", value="detect")
            train_dir = gr.Textbox(label="train 图像目录（可选）")
            val_dir = gr.Textbox(label="val 图像目录（可选）")
            test_dir = gr.Textbox(label="test 图像目录（可选）")
            labels_dir = gr.Textbox(label="labels 目录（可选）")
            validate_btn = gr.Button("校验数据集")
            validate_out = gr.Textbox(label="校验结果", lines=6)
            validate_btn.click(
                run_dataset_validation,
                inputs=[dataset_root, data_yaml, task_type, train_dir, val_dir, test_dir, labels_dir],
                outputs=validate_out,
            )

            gr.Markdown("## 模型与训练参数")
            with gr.Row():
                model_family = gr.Dropdown(["yolov8", "yolov10"], label="模型族", value="yolov8")
                model_task = gr.Dropdown(["detect", "segment"], label="模型任务", value="detect")
                model_size = gr.Dropdown(["n", "s", "m", "l", "x"], label="模型规模", value="n")
            pretrained_weights = gr.Textbox(
                label="预训练权重路径（可选）",
                value="/mnt/e/IDE/AI_Project/yolo_platform/model/yolov8n.pt",
            )
            device = gr.Textbox(label="设备（如 0 或 0,1 或 cpu）", value="0")
            with gr.Row():
                epochs = gr.Number(label="epochs", value=100, precision=0)
                batch_size = gr.Number(label="batch_size", value=16, precision=0)
                imgsz = gr.Number(label="imgsz", value=640, precision=0)
            with gr.Row():
                lr0 = gr.Number(label="lr0", value=0.01)
                lrf = gr.Number(label="lrf（可选）", value=None)
                momentum = gr.Number(label="momentum（可选）", value=None)
            with gr.Row():
                workers = gr.Number(label="workers", value=8, precision=0)
                save_period = gr.Number(label="save_period（可选）", value=None, precision=0)
                seed = gr.Number(label="seed（可选）", value=None, precision=0)
            start_btn = gr.Button("生成并启动任务", variant="primary")
            start_out = gr.Textbox(label="任务创建结果", lines=4)
            start_btn.click(
                start_task,
                inputs=[
                    dataset_root,
                    data_yaml,
                    task_type,
                    model_family,
                    model_task,
                    model_size,
                    pretrained_weights,
                    device,
                    epochs,
                    batch_size,
                    imgsz,
                    lr0,
                    workers,
                    lrf,
                    momentum,
                    save_period,
                    seed,
                ],
                outputs=start_out,
            )

        with gr.Tab("任务详情"):
            task_id_box = gr.Textbox(label="任务 ID")
            refresh_detail = gr.Button("刷新详情")
            status_box = gr.Textbox(label="状态")
            log_box = gr.Textbox(label="日志尾部", lines=12)
            metrics_json = gr.JSON(label="指标 JSON")
            metrics_plot = gr.LinePlot(
                x="epoch",
                y="value",
                color="metric",
                title="训练指标",
                overlay_point=True,
                height=300,
            )
            refresh_detail.click(
                get_task_detail,
                inputs=[task_id_box],
                outputs=[status_box, log_box, metrics_json, metrics_plot],
            )

            cancel_btn = gr.Button("终止任务", variant="stop")
            cancel_out = gr.Textbox(label="终止结果")
            cancel_btn.click(cancel_task, inputs=[task_id_box], outputs=cancel_out)

    return demo


app = build_ui()

if __name__ == "__main__":
    app.launch()
