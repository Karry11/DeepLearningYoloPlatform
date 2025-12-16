# YOLO 深度学习训练平台（Ultralytics YOLOv8/YOLOv10）

基于 Gradio 5.23.1 的本地可视化平台，支持 YOLOv8/YOLOv10 检测/分割任务的创建、环境检测、数据集校验、训练启动、日志与指标查看，并预留 DeepSeek 代码生成接口（当前使用本地模板）。

## 主要功能
- 环境检测：CPU/GPU/CUDA/关键包版本检查。
- 数据集校验：data.yaml/路径合法性、样本计数、标签格式粗检。
- 任务管理：创建、状态刷新、终止、日志尾部查看。
- 训练脚本生成：DeepSeek client 抽象（离线模板版），保存到 `projects/<task_id>/train_v1.py`。
- 训练执行：子进程启动，日志重定向，状态自动更新。
- 指标可视化：解析 Ultralytics `results.csv`，在 Gradio 中折线展示。

## 目录结构
```
app/
  main.py              # Gradio 入口
  models.py            # Pydantic 数据模型
  services/            # 业务模块
    resource_checker.py
    dataset_validator.py
    task_manager.py
    run_executor.py
    deepseek_client.py
    log_manager.py
    metrics_parser.py
  storage/             # 任务元数据与项目输出目录
tests/
  test_dataset_validator.py
```

## 运行方式
```bash
python -m app.main
```
- 启动后访问 Gradio 提示的本地地址。
- “新建任务”页填写数据集、模型和训练参数，点击“生成并启动任务”。
- “任务列表”页查看状态，“任务详情”页查看日志尾部与指标。

## 深度集成提示
- 若需接入真实 DeepSeek API，可在 `app/services/deepseek_client.py` 的 `generate_script` 中替换模板逻辑。
- 默认训练日志写入 `projects/<task_id>/train.log`，Ultralytics 输出在 `projects/<task_id>/<task_id>/`。
- 结束/失败时会扫描日志关键字；可在 `app/services/log_manager.py` 中调整关键词。
