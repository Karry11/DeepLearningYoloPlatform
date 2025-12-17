# 深度学习训练平台（YOLO / 自定义模型）

基于 Gradio 5.23.1 的本地可视化平台，支持 YOLOv8/YOLOv10 检测/分割/分类，以及自定义模型描述（通过 DeepSeek 生成脚本）的开放式训练。提供环境检测、数据集校验、任务管理、重试与指标可视化。

## 主要功能
- 环境检测：CPU/GPU/CUDA/关键包摘要；可附加根目录 `python_env.json` 里的核心包版本注入 DeepSeek 提示。
- 数据集校验：YOLO 检测/分割使用 data.yaml；分类任务支持目录结构（train/val/test 按子目录为类）。
- 模型与脚本：
  - YOLO 模式：本地模板或 DeepSeek 生成，默认防止自动下载，缺库写入日志后退出。
  - 自定义模型模式：输入文本描述（如 ResNet / ViT / 多层 CNN），DeepSeek 根据描述与硬件约束（显存/CPU）生成 PyTorch 脚本；脚本长度限制 <= 300 行。
  - 安全检查：生成脚本落盘前进行黑名单过滤（subprocess/os.system/requests 等）。
- 任务管理：创建、重试（失败会读日志+旧脚本触发 DeepSeek 再生成，最多 3 次）、终止、状态刷新、日志尾部查看。
- 指标可视化：解析 Ultralytics `results.csv`，在 Gradio 折线展示。
- 重试与自愈：失败时自动生成新的 `train_v{n}.py` 并重启，超过 `max_retry` 标记失败。

## 目录结构
```
app/
  main.py              # Gradio 入口、UI、重试逻辑、脚本安全校验
  models.py            # Pydantic 数据模型
  services/
    resource_checker.py    # 环境摘要
    dataset_validator.py   # 数据集校验（YOLO / 分类目录）
    task_manager.py        # 任务存储与状态
    run_executor.py        # 子进程启动/停止
    deepseek_client.py     # DeepSeek 调用/模板生成，读取 python_env.json
    log_manager.py         # 日志尾部与错误识别
    metrics_parser.py      # Ultralytics 指标解析
  storage/             # 任务元数据与输出
python_env.json        # 核心包版本（可选，供 DeepSeek 提示）
tests/
  test_dataset_validator.py
```

## 运行方式
```bash
python -m app.main
```
- 启动后访问 Gradio 提示地址。
- “新建任务”页填写数据集、模型、训练参数，选择脚本生成方式（模板/DeepSeek），点击“生成并启动任务”。
- “任务列表”页查看状态；“任务详情”页查看日志尾部与指标；可终止任务。

## 架构图（逻辑示意）
```
[Gradio UI]
  ├─ 任务列表 / 详情（状态、日志、指标）
  └─ 新建任务（数据集、模型、脚本生成方式、训练参数）
       |
       v
[main.py]
  ├─ 调用 dataset_validator 校验数据
  ├─ DeepSeek/模板生成脚本（含 python_env.json 补充、脚本安全检查）
  ├─ 任务落盘 task_manager / storage
  ├─ run_executor 启动子进程 + 日志重定向
  └─ 失败检测 + _retry_task（最多 3 次，DeepSeek 重新生成）

[DeepSeekClient]
  ├─ 读取环境摘要 + python_env.json
  ├─ 自定义模型/YOLO 双提示，显存/CPU 自适应约束，缺库/OOM 提示
  └─ 生成 train_v{n}.py（<=300 行约束）

[DatasetValidator]
  ├─ YOLO: data.yaml 路径/键/标签粗检
  └─ 分类: 目录结构(train/val/test)，按子目录推断类别

[Outputs]
  ├─ projects/<task_id>/train_v{n}.py
  ├─ train.log（含缺库/错误/OOM）
  ├─ results.csv / weights/
  └─ weights_info.json
```

## 深度集成提示
- 接入 DeepSeek：设置 `DEEPSEEK_API_KEY`（可选 `DEEPSEEK_ENDPOINT`）；自定义模型必须选 DeepSeek。
- 若有 `python_env.json` 会自动加入 prompt；否则使用基础环境摘要。
- 默认训练日志写入 `projects/<task_id>/train.log`，Ultralytics 输出在 `projects/<task_id>/<task_id>/`。
- 重试策略：非 0 退出或日志含错误时，若 `retries < max_retry`，会读取日志尾部 + 旧脚本交给 DeepSeek 再生成；超限则标记 FAILED。
