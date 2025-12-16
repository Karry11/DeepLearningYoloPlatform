import json
from pathlib import Path
from textwrap import dedent
from typing import Optional

from app.models import TaskConfig


class DeepSeekClient:
    """
    Placeholder DeepSeek client.

    当前环境网络受限，使用本地模板生成训练脚本。
    如需接入真实 DeepSeek API，可在 `generate_script` 中替换实现。
    """

    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None):
        self.api_key = api_key
        self.endpoint = endpoint

    def generate_script(
        self,
        task: TaskConfig,
        env_summary: str,
        last_log_snippet: Optional[str] = None,
        previous_script: Optional[str] = None,
    ) -> str:
        cfg = task
        model_ref = cfg.model.pretrained_weights or f"{cfg.model.family}{cfg.model.size}-{cfg.model.task}"
        data_path = cfg.dataset.data_yaml or ""
        params = cfg.train_params
        env_summary_literal = json.dumps(env_summary)
        model_exists = bool(cfg.model.pretrained_weights) and Path(model_ref).exists()

        # Assemble train kwargs
        kwargs = {
            "data": data_path,
            "epochs": params.epochs,
            "batch": params.batch_size,
            "imgsz": params.imgsz,
            "device": cfg.model.device,
            "project": params.project or cfg.runtime.project_dir,
            "name": params.name or cfg.task_id,
            "exist_ok": True,
            "save": True,
        }
        if params.lr0:
            kwargs["lr0"] = params.lr0
        if params.lrf:
            kwargs["lrf"] = params.lrf
        if params.momentum:
            kwargs["momentum"] = params.momentum
        if params.workers is not None:
            kwargs["workers"] = params.workers
        if params.save_period:
            kwargs["save_period"] = params.save_period
        if params.seed is not None:
            kwargs["seed"] = params.seed
        if model_exists:
            # Prevent Ultralytics from auto-downloading newer weights when custom weights are provided.
            kwargs["pretrained"] = False

        # Build kwargs string
        kwargs_lines = [f"    '{k}': {repr(v)}," for k, v in kwargs.items()]
        kwargs_block = "\n".join(kwargs_lines)

        template = f'''
import json
import logging
from pathlib import Path
from ultralytics import YOLO


def setup_logger(log_file: str):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main():
    log_path = Path(r"{cfg.runtime.log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(log_path))
    logger.info("Starting YOLO training")
    try:
        env_info = json.loads({env_summary_literal})
    except Exception:
        env_info = {env_summary_literal}
    logger.info("Environment summary: %s", env_info)

    model_ref = r"{model_ref}"
    logger.info(f"Loading model: {{model_ref}}")
    if not Path(model_ref).exists():
        logger.error("Model weights not found at %s", model_ref)
        raise FileNotFoundError(f"Model weights not found: {model_ref}")
    model = YOLO(model_ref)

    train_kwargs = {{
{kwargs_block}
    }}
    logger.info(f"Train kwargs: {{json.dumps(train_kwargs, indent=2)}}")
    results = model.train(**train_kwargs)
    logger.info("Training finished")
    logger.info(f"Results saved to: {{results.save_dir}}")

    # Save best weights path for downstream usage
    best = Path(results.save_dir) / "weights" / "best.pt"
    last = Path(results.save_dir) / "weights" / "last.pt"
    info = {{
        "best": str(best) if best.exists() else "",
        "last": str(last) if last.exists() else "",
    }}
    (log_path.parent / "weights_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
'''
        return dedent(template)
