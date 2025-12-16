
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
    log_path = Path(r"/mnt/e/IDE/AI_Project/require/app/storage/projects/20251216-002933-7992/train.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(log_path))
    logger.info("Starting YOLO training")
    try:
        env_info = json.loads("{\n  \"cpu\": {\n    \"physical_cores\": 8,\n    \"logical_cores\": 16,\n    \"processor\": \"x86_64\"\n  },\n  \"gpu\": {\n    \"gpus\": [\n      {\n        \"name\": \"NVIDIA GeForce RTX 5060 Ti\",\n        \"memory_total\": \"16311 MiB\"\n      }\n    ],\n    \"count\": 1\n  },\n  \"cuda\": {\n    \"cuda_version\": \"12.8\",\n    \"driver\": null\n  },\n  \"python\": {\n    \"python_version\": \"3.10.19\",\n    \"packages\": {\n      \"torch\": \"2.7.1+cu128\",\n      \"ultralytics\": \"8.3.146\",\n      \"numpy\": \"2.2.6\",\n      \"opencv\": null,\n      \"cv2\": \"4.11.0\",\n      \"pandas\": \"2.2.3\",\n      \"pyyaml\": null\n    }\n  },\n  \"warnings\": []\n}")
    except Exception:
        env_info = "{\n  \"cpu\": {\n    \"physical_cores\": 8,\n    \"logical_cores\": 16,\n    \"processor\": \"x86_64\"\n  },\n  \"gpu\": {\n    \"gpus\": [\n      {\n        \"name\": \"NVIDIA GeForce RTX 5060 Ti\",\n        \"memory_total\": \"16311 MiB\"\n      }\n    ],\n    \"count\": 1\n  },\n  \"cuda\": {\n    \"cuda_version\": \"12.8\",\n    \"driver\": null\n  },\n  \"python\": {\n    \"python_version\": \"3.10.19\",\n    \"packages\": {\n      \"torch\": \"2.7.1+cu128\",\n      \"ultralytics\": \"8.3.146\",\n      \"numpy\": \"2.2.6\",\n      \"opencv\": null,\n      \"cv2\": \"4.11.0\",\n      \"pandas\": \"2.2.3\",\n      \"pyyaml\": null\n    }\n  },\n  \"warnings\": []\n}"
    logger.info("Environment summary: %s", env_info)

    model_ref = r"/mnt/e/IDE/AI_Project/yolo_platform/model/yolov8n.pt"
    logger.info(f"Loading model: {model_ref}")
    model = YOLO(model_ref)

    train_kwargs = {
    'data': '/mnt/e/IDE/AI_Project/yolo_platform/datasets/data.yaml',
    'epochs': 1,
    'batch': 16,
    'imgsz': 640,
    'device': '0',
    'project': '/mnt/e/IDE/AI_Project/require/app/storage/projects/20251216-002933-7992',
    'name': '20251216-002933-7992',
    'exist_ok': True,
    'save': True,
    'lr0': 0.01,
    'workers': 8,
    }
    logger.info(f"Train kwargs: {json.dumps(train_kwargs, indent=2)}")
    results = model.train(**train_kwargs)
    logger.info("Training finished")
    logger.info(f"Results saved to: {results.save_dir}")

    # Save best weights path for downstream usage
    best = Path(results.save_dir) / "weights" / "best.pt"
    last = Path(results.save_dir) / "weights" / "last.pt"
    info = {
        "best": str(best) if best.exists() else "",
        "last": str(last) if last.exists() else "",
    }
    (log_path.parent / "weights_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
