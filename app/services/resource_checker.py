import json
import platform
import subprocess
from typing import Dict, List, Optional


def _safe_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


def _run_cmd(cmd: List[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            cmd, capture_output=True, check=False, text=True, timeout=5
        )
        return result.stdout.strip() if result.stdout else result.stderr.strip()
    except Exception:
        return None


def get_cpu_info() -> Dict:
    psutil = _safe_import("psutil")
    physical = psutil.cpu_count(logical=False) if psutil else None
    logical = psutil.cpu_count(logical=True) if psutil else None
    return {
        "physical_cores": physical or "unknown",
        "logical_cores": logical or "unknown",
        "processor": platform.processor(),
    }


def get_gpu_info() -> Dict:
    torch = _safe_import("torch")
    info: Dict[str, object] = {"gpus": []}
    nvidia = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader",
        ]
    )
    if nvidia:
        lines = [line.strip() for line in nvidia.splitlines() if line.strip()]
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                info["gpus"].append({"name": parts[0], "memory_total": parts[1]})
    elif torch and torch.cuda.is_available():
        count = torch.cuda.device_count()
        for idx in range(count):
            props = torch.cuda.get_device_properties(idx)
            info["gpus"].append(
                {"name": props.name, "memory_total": f"{props.total_memory/1e9:.1f} GB"}
            )
    info["count"] = len(info["gpus"])
    return info


def get_cuda_info() -> Dict:
    torch = _safe_import("torch")
    info = {"cuda_version": None, "driver": None}
    smi = _run_cmd(["nvidia-smi"])
    if smi:
        for line in smi.splitlines():
            if "Driver Version" in line and "CUDA Version" in line:
                parts = line.split()
                for idx, token in enumerate(parts):
                    if token == "Version":
                        if idx + 1 < len(parts):
                            info["driver"] = parts[idx + 1]
                    if token == "CUDA":
                        if idx + 2 < len(parts):
                            info["cuda_version"] = parts[idx + 2]
                break
    if torch and torch.version.cuda:
        info["cuda_version"] = torch.version.cuda
    return info


def get_python_info() -> Dict:
    versions = {}
    for name in ["torch", "ultralytics", "numpy", "opencv", "cv2", "pandas", "pyyaml"]:
        mod = _safe_import(name)
        ver = None
        if mod:
            ver = getattr(mod, "__version__", None)
        versions[name] = ver
    return {
        "python_version": platform.python_version(),
        "packages": versions,
    }


def summarize_environment() -> Dict:
    cpu = get_cpu_info()
    gpu = get_gpu_info()
    cuda = get_cuda_info()
    pyinfo = get_python_info()
    summary = {
        "cpu": cpu,
        "gpu": gpu,
        "cuda": cuda,
        "python": pyinfo,
    }

    warnings: List[str] = []
    torch = _safe_import("torch")
    ultralytics = _safe_import("ultralytics")
    if not torch:
        warnings.append("torch not installed")
    else:
        if not torch.cuda.is_available():
            warnings.append("CUDA not available; training will run on CPU")
    if not ultralytics:
        warnings.append("ultralytics not installed")
    summary["warnings"] = warnings
    summary["as_json"] = json.dumps(summary, ensure_ascii=False, indent=2)
    return summary
