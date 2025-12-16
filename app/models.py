from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    RETRYING = "RETRYING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class DatasetConfig(BaseModel):
    root: str = Field(..., description="Dataset root directory")
    data_yaml: Optional[str] = Field(
        None, description="Path to YOLO data.yaml; required for YOLO style datasets"
    )
    task_type: Literal["detect", "segment"] = Field(..., description="YOLO task type")
    train_dir: Optional[str] = Field(None, description="Optional train images dir")
    val_dir: Optional[str] = Field(None, description="Optional val images dir")
    test_dir: Optional[str] = Field(None, description="Optional test images dir")
    labels_dir: Optional[str] = Field(None, description="Optional labels dir")


class ModelConfig(BaseModel):
    family: Literal["yolov8", "yolov10"]
    task: Literal["detect", "segment"]
    size: Literal["n", "s", "m", "l", "x"] = "n"
    pretrained_weights: Optional[str] = None
    device: str = "0"

    @property
    def model_name(self) -> str:
        return f"{self.family}{self.size}-{self.task}"


class TrainParams(BaseModel):
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    lr0: float = 0.01
    lrf: Optional[float] = None
    momentum: Optional[float] = None
    workers: int = 8
    project: Optional[str] = None
    name: Optional[str] = None
    save_period: Optional[int] = None
    seed: Optional[int] = None


class RuntimePaths(BaseModel):
    project_dir: str
    log_path: str
    script_path: str
    weights_output_dir: str
    results_path: str


class TaskConfig(BaseModel):
    task_id: str
    dataset: DatasetConfig
    model: ModelConfig
    train_params: TrainParams
    runtime: RuntimePaths

    @validator("task_id", pre=True, always=True)
    def _fill_task_id(cls, v: Optional[str]) -> str:
        return v or datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


class ValidationMessage(BaseModel):
    level: Literal["info", "warning", "error"] = "info"
    text: str


class DatasetStats(BaseModel):
    train_images: int = 0
    val_images: int = 0
    test_images: int = 0
    classes: int = 0
    names: List[str] = Field(default_factory=list)
    image_shapes: Optional[Dict[str, tuple]] = None


class DatasetValidationResult(BaseModel):
    ok: bool
    messages: List[ValidationMessage]
    stats: Optional[DatasetStats] = None


class TaskRecord(BaseModel):
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    config: TaskConfig
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    retries: int = 0
    max_retry: int = 3
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    error_message: Optional[str] = None
    script_version: int = 1

    def touch(self) -> None:
        self.updated_at = datetime.utcnow()


@dataclass
class LogTail:
    lines: List[str]
    timestamp: datetime
