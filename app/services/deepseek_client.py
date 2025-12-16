import json
import os
import re
from pathlib import Path
from textwrap import dedent
from typing import Optional, Tuple

from app.models import TaskConfig

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None


class DeepSeekClient:
    """
    DeepSeek client with local template fallback.

    - If script_mode == "template" or缺少 API Key，则使用本地模板。
    - DeepSeek 生成最多尝试 3 次，失败后回退模板。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: str = "deepseek-coder",
        max_attempts: int = 3,
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.endpoint = endpoint or os.getenv("DEEPSEEK_ENDPOINT") or "https://api.deepseek.com/v1/chat/completions"
        self.model = model
        self.max_attempts = max_attempts
        self.timeout = timeout

    def _build_prompt(
        self,
        task: TaskConfig,
        env_summary: str,
        last_log_snippet: Optional[str],
        previous_script: Optional[str],
    ) -> str:
        """
        Assemble a safe prompt for DeepSeek code generation.
        """
        task_json = task.model_dump(mode="json")
        log_part = last_log_snippet or "No error log available."
        prev_part = previous_script or ""
        instructions = dedent(
            """
            You are an assistant that writes a single Python training script for Ultralytics YOLO (v8/v10).
            Constraints:
            - Only produce Python code fenced in ```python ... ```.
            - Use `from ultralytics import YOLO`.
            - Use the provided `data.yaml`, device, epochs, batch size, imgsz, and other train params.
            - Write logs to the given log_path via the provided logger pattern.
            - Do NOT run shell/system commands. Do NOT write files outside the project_dir.
            - If a local pretrained_weights path exists, load it and call train(..., pretrained=False) to prevent auto-download. If it does not exist, use the official weight name (e.g., yolov8n.pt / yolov8n-seg.pt) and let YOLO resolve it.
            """
        )
        return dedent(
            f"""
            {instructions}

            Task config (JSON):
            {json.dumps(task_json, ensure_ascii=False, indent=2)}

            Environment summary:
            {env_summary}

            Last log snippet (for fixes):
            {log_part}

            Previous script (if any):
            {prev_part}
            """
        )

    def _extract_code(self, content: str) -> Optional[str]:
        fence = re.findall(r"```python(.*?)```", content, re.DOTALL | re.IGNORECASE)
        if fence:
            return fence[-1].strip()
        fence_any = re.findall(r"```(.*?)```", content, re.DOTALL)
        if fence_any:
            return fence_any[-1].strip()
        return content.strip() if content.strip() else None

    def _call_api(self, prompt: str) -> str:
        if not requests:
            raise RuntimeError("requests not installed; cannot call DeepSeek API")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("DeepSeek API returned no choices")
        return choices[0]["message"]["content"]

    def _render_template(self, task: TaskConfig, env_summary: str) -> str:
        cfg = task
        # Default weight names: yolov8n.pt / yolov8n-seg.pt etc.
        default_weight = f"{cfg.model.family}{cfg.model.size}" + ("-seg.pt" if cfg.model.task == "segment" else ".pt")
        model_ref = cfg.model.pretrained_weights or default_weight
        data_path = cfg.dataset.data_yaml or ""
        params = cfg.train_params
        try:
            env_obj = json.loads(env_summary) if isinstance(env_summary, str) else env_summary
        except Exception:
            env_obj = env_summary
        env_summary_literal = json.dumps(env_obj, ensure_ascii=False, indent=2)
        model_exists = bool(cfg.model.pretrained_weights) and Path(str(cfg.model.pretrained_weights)).exists()

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
            kwargs["pretrained"] = False

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
        logger.warning("Model weights not found locally, will let YOLO resolve by name: %s", model_ref)
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

    def _generate_via_api(
        self,
        task: TaskConfig,
        env_summary: str,
        last_log_snippet: Optional[str],
        previous_script: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Try DeepSeek up to max_attempts, return (script, error_msg).
        """
        prompt = self._build_prompt(task, env_summary, last_log_snippet, previous_script)
        last_error: Optional[str] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                content = self._call_api(prompt)
                code = self._extract_code(content)
                if code:
                    return dedent(code), None
                last_error = "No code block found in DeepSeek response"
            except Exception as exc:  # pragma: no cover - network/runtime path
                last_error = str(exc)
                continue
        return None, last_error

    def generate_script(
        self,
        task: TaskConfig,
        env_summary: str,
        last_log_snippet: Optional[str] = None,
        previous_script: Optional[str] = None,
    ) -> str:
        """
        Generate training script; respect task.script_mode.
        """
        use_api = task.script_mode == "deepseek" and self.api_key
        if not use_api:
            return self._render_template(task, env_summary)

        script, err = self._generate_via_api(task, env_summary, last_log_snippet, previous_script)
        if script:
            return script
        # Fallback to template with inline comment indicating fallback reason.
        template = self._render_template(task, env_summary)
        if err:
            notice = f"# Fallback to template because DeepSeek failed: {err}\n"
            return notice + template
        return template
