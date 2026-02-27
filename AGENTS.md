# AGENTS.md - 蘑菇毒性预测项目开发者指南

## 1. 项目概览
本项目是一个基于图像识别与机器学习集成的蘑菇毒性预测系统。它由三个主要部分组成：
- **分类器服务 (Classifier API)**: 基于 XGBoost 和 LightGBM 的集成模型，负责最终毒性判定。
- **特征提取服务 (VLM API)**: 调用视觉大模型 (Local LM Studio) 从图片中解析形态特征。
- **用户界面 (Streamlit App)**: 提供图片上传、特征复核及结果展示的交互界面。

---

## 2. 运行与测试指令

### 2.1 环境配置
项目依赖 Python 3.10 环境及 Conda 管理。
```bash
conda env create -f environment.yml
conda activate mushroom
```

### 2.2 服务启动
项目需要并发启动以下三个进程：
1. **分类器**: `python src/classifier_api.py` (默认端口: 8000)
2. **VLM 接口**: `python src/imgprocess_api.py` (默认端口: 8001)
3. **前端界面**: `streamlit run src/front.py` (默认端口: 8501)

**Windows 快速启动**:
根目录下提供了 `run_services.bat`，双击即可一键启动所有组件。

### 2.3 功能验证 (单一测试)
可使用 `curl` 验证分类器逻辑：
```bash
curl -X 'POST' 'http://127.0.0.1:8000/predict' \
-H 'Content-Type: application/json' \
-d '[{"id": 1, "cap-diameter": 5.0, "cap-shape": "x", "cap-color": "n", "does-bruise-or-bleed": "f", "stem-height": 10.0, "stem-width": 15.0, "has-ring": "t", "habitat": "d", "season": "a"}]'
```

---

## 3. 代码风格与规范

- **命名规范**: 函数与变量使用 `snake_case`；类名使用 `PascalCase`；常量使用 `UPPER_SNAKE_CASE`。
- **类型注解**: 所有函数必须包含类型提示 (Type Hinting)，例如 `def func(data: Dict[str, Any]) -> List[str]:`。
- **导入规范**: 按“标准库 -> 第三方库 -> 本地模块”顺序分组，组间空一行。
- **异常处理**: API 内部需使用 FastAPI 的 `HTTPException` 返回结构化错误。
- **模型路径**: 脚本运行需在项目根目录下进行，以确保 `../models` 等相对路径正确引用。

---

## 4. 核心业务逻辑约束 (重要)

### 4.1 特征工程协议
模型输入依赖严格的特征顺序（共 20 项），修改 `feature_order` 可能会导致模型预测完全失效。

### 4.2 空值映射约定
在推理阶段，所有输入的 `None` 或 `null` 必须显式转换为字符串 `"nan"`。这是因为训练阶段使用的 LabelEncoder 将 NaN 视为了特定的字符串类别。

---

## 5. 待优化路线图 (Roadmap)

### 5.1 工程化改进
- [ ] **Docker化**: 编写 `docker-compose.yml` 实现多服务一键编排。
- [x] **自动化测试**: 引入 `pytest` 针对特征转换逻辑编写单元测试。
- [x] **配置解耦**: 将硬编码的 URL 和文件路径迁移至 `.env` 文件。

### 5.2 算法与逻辑优化
- [ ] **预测解释性**: 集成 SHAP 或 LIME，在前端展示影响“有毒”判定的关键形态特征。
- [ ] **Prompt 调优**: 优化 VLM 的提示词，提高形态提取的零样本识别精度。
- [ ] **异常检测**: 在 API 层增加对输入数值（如菌盖直径）的物理范围校验。

### 5.3 用户体验增强
- [x] **计算缓存**: 在 Streamlit 中对昂贵的 VLM 调用增加 `@st.cache_data`。
- [ ] **纠错闭环**: 增加“识别纠错”按钮，收集用户反馈数据用于后续模型微调。

---

## 6. 变更日志 (Change Log)

### [2026-02-27] - 初始工程化优化
- **配置解耦**: 引入 `.env` 文件和 `python-dotenv`，解耦了 API 端口、模型路径及服务 URL。
- **性能优化**: 在 Streamlit 前端增加了对 VLM 分析结果的缓存机制，减少重复调用开销。
- **测试框架**: 建立了 `tests/` 目录，引入 `pytest` 并编写了第一个针对特征预处理逻辑的单元测试。
- **依赖更新**: `environment.yml` 已同步更新，增加了 `python-dotenv` 和 `pytest`。

---
*注：本文件为 AI 代理与开发者协作生成的动态文档，更新代码逻辑后请同步维护。*
