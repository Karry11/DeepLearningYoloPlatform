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
    LogTail,
)
from app.services import dataset_validator, resource_checker
from app.services.deepseek_client import DeepSeekClient
from app.services.log_manager import contains_error, tail_log
from app.services.metrics_parser import parse_results_csv
from app.services.run_executor import RunExecutor
from app.services.task_manager import TaskManager


APP_CSS = """
:root {
  --brand-primary: #0f766e;
  --brand-secondary: #f97316;
  --surface: #ffffff;
  --muted: #f1f5f9;
  --border: #d6e3f0;
  --text: #0f172a;
}

#yolo-app {
  background:
    radial-gradient(circle at 20% 20%, #f0f9ff 0%, #ffffff 28%),
    radial-gradient(circle at 85% 0%, #fff7ed 0%, #ffffff 32%);
  color: var(--text);
}

#yolo-app .hero {
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px 20px;
  background: linear-gradient(135deg, #e0f4ff 0%, #fff4e5 100%);
  box-shadow: 0 12px 28px rgba(15, 118, 110, 0.12);
}

#yolo-app .card {
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--surface);
  box-shadow: 0 10px 24px rgba(15, 118, 110, 0.08);
  padding: 16px;
  margin-bottom: 14px;
}

#yolo-app .card h3 {
  margin-top: 0;
  margin-bottom: 8px;
}

#yolo-app .cta-card {
  background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
  color: #ffffff;
  box-shadow: 0 16px 32px rgba(20, 184, 166, 0.35);
}

#yolo-app .cta-card textarea,
#yolo-app .cta-card label {
  color: #0f172a;
}

#yolo-app .accent-btn button {
  background: linear-gradient(135deg, #0f766e 0%, #0ea5e9 100%);
  color: #ffffff;
  border: none;
}

#yolo-app .accent-secondary button {
  background: linear-gradient(135deg, #f97316 0%, #f59e0b 100%);
  color: #0f172a;
  border: none;
}

#yolo-app .compact-input input,
#yolo-app .compact-input textarea,
#yolo-app .compact-input select {
  border-radius: 10px;
  border-color: var(--border);
  background: #f8fafc;
}

#tasks-grid {
  border-radius: 12px;
  border: 1px solid var(--border);
  box-shadow: 0 6px 18px rgba(15, 118, 110, 0.05);
}
"""

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
    model_kind: str,
    model_family: str,
    model_task: str,
    model_size: str,
    pretrained_weights: Optional[str],
    custom_description: Optional[str],
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
    script_mode: str = "template",
) -> TaskConfig:
    task_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S-") + str(random.randint(1000, 9999))
    runtime = _runtime_paths(task_id)
    dataset_cfg = DatasetConfig(
        root=dataset_root,
        data_yaml=data_yaml,
        task_type=task_type,  # type: ignore[arg-type]
    )
    model_cfg = ModelConfig(
        kind=model_kind,  # type: ignore[arg-type]
        family=model_family,  # type: ignore[arg-type]
        task=model_task,  # type: ignore[arg-type]
        size=model_size,  # type: ignore[arg-type]
        pretrained_weights=pretrained_weights,
        device=device,
        custom_description=custom_description,
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
        script_mode=script_mode,  # type: ignore[arg-type]
    )


def _persist_script(task: TaskConfig, script: str) -> None:
    path = Path(task.runtime.script_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script, encoding="utf-8")


FORBIDDEN_SCRIPT_TOKENS = [
    "subprocess",
    "os.system",
    "shutil.rmtree",
    "requests",
    "urllib",
    "socket",
]


def _validate_script_safety(script: str) -> None:
    lowered = script.lower()
    for token in FORBIDDEN_SCRIPT_TOKENS:
        if token in lowered:
            raise ValueError(f"Generated script contains forbidden token: {token}")


def _maybe_locate_results(task: TaskRecord) -> Optional[Path]:
    results_path = Path(task.config.runtime.results_path)
    if results_path.exists():
        return results_path
    project_dir = Path(task.config.runtime.project_dir)
    for candidate in project_dir.rglob("results.csv"):
        return candidate
    return None


def _retry_task(task: TaskRecord, log_tail: "LogTail") -> Optional[TaskRecord]:
    """Attempt a DeepSeek-assisted retry with a new script version."""
    if task.retries >= task.max_retry:
        task_manager.update_status(task.task_id, TaskStatus.FAILED, error_message="Max retries reached")
        return task

    last_snippet = "\n".join(log_tail.lines[-100:]) if log_tail.lines else ""
    prev_script = None
    try:
        prev_script = Path(task.config.runtime.script_path).read_text(encoding="utf-8")
    except Exception:
        prev_script = None

    new_version = task.script_version + 1
    cfg_copy = task.config.model_copy(deep=True)
    cfg_copy.runtime.script_path = str(Path(cfg_copy.runtime.project_dir) / f"train_v{new_version}.py")

    env_summary = resource_checker.summarize_environment().get("as_json", "")
    try:
        script = deepseek.generate_script(cfg_copy, env_summary, last_snippet, prev_script)
        _validate_script_safety(script)
        _persist_script(cfg_copy, script)
    except Exception as exc:
        return task_manager.update_status(task.task_id, TaskStatus.FAILED, error_message=str(exc))

    updated = task_manager.update_task(
        task.task_id,
        retries=task.retries + 1,
        script_version=new_version,
        config=cfg_copy,
        status=TaskStatus.RETRYING,
        error_message=None,
    )
    if not updated:
        return task
    try:
        executor.start(updated)
        task_manager.update_status(task.task_id, TaskStatus.RUNNING)
    except Exception as exc:
        return task_manager.update_status(task.task_id, TaskStatus.FAILED, error_message=str(exc))
    return updated


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
        # Retry logic: if failed and retries remain, regenerate script via DeepSeek and restart.
        if new_status == TaskStatus.FAILED and task.retries < task.max_retry:
            retry_task = _retry_task(task, log_tail)
            return retry_task or task

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
    # Return a DataFrame to avoid Gradio rendering dicts as "[object Object]"
    return pd.DataFrame(
        rows,
        columns=["task_id", "status", "model", "dataset", "created_at", "updated_at", "retries"],
    )


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
    model_kind: str,
    model_family: str,
    model_task: str,
    model_size: str,
    pretrained_weights: str,
    custom_description: str,
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
    script_mode: str,
) -> str:
    cfg = _build_task_config(
        dataset_root=dataset_root,
        data_yaml=data_yaml or None,
        task_type=task_type,
        model_kind=model_kind,
        model_family=model_family,
        model_task=model_task,
        model_size=model_size,
        pretrained_weights=pretrained_weights or None,
        custom_description=custom_description or None,
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
        script_mode=script_mode or "template",
    )
    validation = dataset_validator.validate_dataset(cfg.dataset)
    if not validation.ok:
        details = "; ".join([m.text for m in validation.messages])
        return f"Dataset validation failed: {details}"

    # Custom模型需要 DeepSeek 模式
    if cfg.model.kind == "custom" and cfg.script_mode != "deepseek":
        return "自定义模型需要选择 DeepSeek 生成脚本模式。"

    record = task_manager.create_task(cfg)
    env_summary = resource_checker.summarize_environment().get("as_json", "")
    try:
        script = deepseek.generate_script(cfg, env_summary)
        _validate_script_safety(script)
        _persist_script(cfg, script)
    except Exception as exc:
        task_manager.update_status(cfg.task_id, TaskStatus.FAILED, error_message=str(exc))
        return f"Failed to generate script: {exc}"

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
    with gr.Blocks(theme="base", css=APP_CSS, elem_id="yolo-app") as demo:
        gr.Markdown(
            """
            <div class="hero">
              <h1 style="margin:0;">深度学习训练平台</h1>
              <p style="margin:6px 0 0 0;">可视化创建任务、校验数据、启动训练与查看指标。支持 DeepSeek 生成脚本或本地模板。</p>
            </div>
            """,
            elem_id="hero",
        )
        with gr.Tab("任务列表"):
            gr.Markdown("### 任务总览")
            with gr.Row():
                status_filter = gr.Dropdown(
                    [s.value for s in TaskStatus],
                    label="状态筛选",
                    value=None,
                    allow_custom_value=True,
                    elem_classes=["compact-input"],
                )
                refresh = gr.Button("刷新任务列表", elem_classes=["accent-btn"])
            tasks_grid = gr.Dataframe(
                headers=["task_id", "status", "model", "dataset", "created_at", "updated_at", "retries"],
                datatype=["str", "str", "str", "str", "str", "str", "number"],
                interactive=False,
                wrap=True,
                elem_id="tasks-grid",
            )
            refresh.click(list_tasks_ui, inputs=[status_filter], outputs=[tasks_grid])

        with gr.Tab("新建任务"):
            gr.Markdown("### 配置与启动")
            with gr.Row():
                with gr.Column(scale=6):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("#### 环境检测")
                        env_btn = gr.Button("检测环境", elem_classes=["accent-btn"])
                        env_output = gr.Textbox(label="检测结果", lines=5)
                        env_btn.click(run_env_check, outputs=env_output)

                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("#### 数据集配置")
                        dataset_root = gr.Textbox(
                            label="数据集根目录",
                            value="/mnt/e/IDE/AI_Project/yolo_platform/datasets",
                            elem_classes=["compact-input"],
                        )
                        data_yaml = gr.Textbox(
                            label="data.yaml 路径（推荐）",
                            value="/mnt/e/IDE/AI_Project/yolo_platform/datasets/data.yaml",
                            elem_classes=["compact-input"],
                        )
                        task_type = gr.Dropdown(
                            ["detect", "segment", "classify"],
                            label="任务类型",
                            value="detect",
                            elem_classes=["compact-input"],
                        )
                        with gr.Row():
                            train_dir = gr.Textbox(label="train 图像目录（可选）", elem_classes=["compact-input"])
                            val_dir = gr.Textbox(label="val 图像目录（可选）", elem_classes=["compact-input"])
                        with gr.Row():
                            test_dir = gr.Textbox(label="test 图像目录（可选）", elem_classes=["compact-input"])
                            labels_dir = gr.Textbox(label="labels 目录（可选）", elem_classes=["compact-input"])
                        validate_btn = gr.Button("校验数据集", elem_classes=["accent-secondary"])
                        validate_out = gr.Textbox(label="校验结果", lines=5)
                        validate_btn.click(
                            run_dataset_validation,
                            inputs=[dataset_root, data_yaml, task_type, train_dir, val_dir, test_dir, labels_dir],
                            outputs=validate_out,
                        )
                with gr.Column(scale=6):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("#### 模型配置")
                        model_kind = gr.Radio(
                            ["yolo", "custom"],
                            label="模型类型",
                            value="yolo",
                            info="YOLO 标准模型或自定义描述模型（自定义需 DeepSeek 生成脚本）",
                        )
                        with gr.Row():
                            model_family = gr.Dropdown(
                                ["yolov8", "yolov10"],
                                label="模型族",
                                value="yolov8",
                                elem_classes=["compact-input"],
                            )
                            model_task = gr.Dropdown(
                                ["detect", "segment", "classify"],
                                label="模型任务",
                                value="detect",
                                elem_classes=["compact-input"],
                            )
                            model_size = gr.Dropdown(
                                ["n", "s", "m", "l", "x"],
                                label="模型规模",
                                value="n",
                                elem_classes=["compact-input"],
                            )
                        pretrained_weights = gr.Textbox(
                            label="预训练权重路径（可选）",
                            value="/mnt/e/IDE/AI_Project/yolo_platform/model/yolov8n.pt",
                            elem_classes=["compact-input"],
                        )
                        custom_description = gr.Textbox(
                            label="自定义模型描述",
                            placeholder="例如：ResNet50 backbone + FPN + 3 heads；或简单描述 5 层卷积网络用于分类/检测",
                            lines=3,
                        )
                        device = gr.Textbox(label="设备（如 0 或 0,1 或 cpu）", value="0", elem_classes=["compact-input"])

                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("#### 训练参数")
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

                    with gr.Group(elem_classes=["card", "cta-card"]):
                        gr.Markdown("#### 脚本生成与启动")
                        script_mode = gr.Radio(
                            ["template", "deepseek"],
                            label="脚本生成方式",
                            value="template",
                            info="选择使用本地模板或 DeepSeek 生成训练脚本（DeepSeek 至多尝试 3 次后回退模板）",
                        )
                        start_btn = gr.Button("生成并启动任务", variant="primary")
                        start_out = gr.Textbox(label="任务创建结果", lines=4)
                        start_btn.click(
                            start_task,
                            inputs=[
                                dataset_root,
                                data_yaml,
                                task_type,
                                model_kind,
                                model_family,
                                model_task,
                                model_size,
                                pretrained_weights,
                                custom_description,
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
                                script_mode,
                            ],
                            outputs=start_out,
                        )

        with gr.Tab("任务详情"):
            gr.Markdown("### 任务详情")
            with gr.Group(elem_classes=["card"]):
                with gr.Row():
                    task_id_box = gr.Textbox(label="任务 ID", elem_classes=["compact-input"])
                    refresh_detail = gr.Button("刷新详情", elem_classes=["accent-btn"])
                cancel_btn = gr.Button("终止任务", variant="stop", elem_classes=["accent-secondary"])
                cancel_out = gr.Textbox(label="终止结果")
                cancel_btn.click(cancel_task, inputs=[task_id_box], outputs=cancel_out)

            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["card"]):
                        status_box = gr.Textbox(label="状态")
                        log_box = gr.Textbox(label="日志尾部", lines=12)
                with gr.Column(scale=7):
                    with gr.Group(elem_classes=["card"]):
                        metrics_json = gr.JSON(label="指标 JSON")
                        metrics_plot = gr.LinePlot(
                            x="epoch",
                            y="value",
                            color="metric",
                            title="训练指标",
                            overlay_point=True,
                            height=320,
                        )
            refresh_detail.click(
                get_task_detail,
                inputs=[task_id_box],
                outputs=[status_box, log_box, metrics_json, metrics_plot],
            )

    return demo


app = build_ui()

if __name__ == "__main__":
    app.launch()
