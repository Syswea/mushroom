# 🍄 蘑菇毒性预测系统 (Mushroom Toxicity Prediction System)

本项目是一个基于图像识别与机器学习集成技术的蘑菇毒性预测系统。通过结合视觉大模型（VLM）的形态特征提取能力与高性能梯度提升树（XGBoost/LightGBM）的分类能力，实现对蘑菇“可食用”或“有毒”的智能判定。

---

## 🚀 系统特性

- **智能特征提取**: 利用视觉大模型 (VLM) 自动从上传的蘑菇图片中解析 20 项形态学特征（如菌盖形状、菌褶颜色、菌柄表面等）。
- **多模型集成推理**: 后端采用 XGBoost 与 LightGBM 的集成模型，基于 Stacking 策略提供高精度的毒性分类预测。
- **人工核验闭环**: 前端界面支持对 AI 提取结果的实时复核与手动修正，确保预测输入的准确性。
- **工程化优化**: 支持 `.env` 配置解耦、Streamlit 计算缓存以及完善的单元测试。

---

## 🏗️ 系统架构

系统由三个主要组件构成：

1.  **前端界面 (Streamlit)**: 提供用户交互，支持图片上传、特征展示、手动修正及预测结果可视化。
2.  **特征提取服务 (VLM API)**: 基于 FastAPI 封装，调用本地视觉大模型（通过 LM Studio 部署）进行形态学分析。
3.  **分类器服务 (Classifier API)**: 基于 FastAPI 封装，加载预训练的集成学习模型，执行最终的毒性概率计算。

---

## 🛠️ 技术栈

- **编程语言**: Python 3.10
- **Web 框架**: FastAPI (后端), Streamlit (前端)
- **机器学习**: XGBoost, LightGBM, Scikit-learn, Pandas, Numpy
- **视觉模型接口**: OpenAI SDK (对接本地 LM Studio)
- **环境管理**: Conda, python-dotenv
- **测试框架**: Pytest

---

## 📦 快速开始

### 1. 环境准备
确保已安装 [Conda](https://docs.conda.io/en/latest/)。

```bash
# 创建并激活环境
conda env create -f environment.yml
conda activate mushroom
```

### 2. 配置环境变量
项目根目录下包含 `.env` 文件，可根据实际情况修改服务端口和模型路径：
- `CLASSIFIER_PORT`: 分类器 API 端口 (默认 8000)
- `VLM_PORT`: VLM 接口端口 (默认 8001)
- `LM_STUDIO_BASE_URL`: 本地视觉模型服务地址 (默认 localhost:1234)

### 3. 启动服务

#### **Windows 用户 (推荐)**
双击根目录下的 `run_services.bat`，即可一键启动所有组件。

#### **手动启动 (多终端运行)**
1.  **启动分类器**: `python src/classifier_api.py`
2.  **启动 VLM 接口**: `python src/imgprocess_api.py`
3.  **启动前端**: `streamlit run src/front.py`

---

## 📖 使用指南

1.  **上传图片**: 在前端页面选择一张清晰的蘑菇全貌图。
2.  **AI 分析**: 点击“AI 提取特征”，等待视觉大模型解析形态特征。
3.  **特征复核**: 在右侧表单中检查识别结果。如有明显偏差（如颜色、形状），可手动调整。
4.  **毒性预测**: 点击“确认并预测毒性”，查看最终的鉴定结果及其中毒概率。

---

## 📂 目录结构

```text
C:\Workspace\mushroom
├── src/                # 源代码目录
│   ├── classifier_api.py   # 分类器推理服务
│   ├── imgprocess_api.py   # 视觉特征提取服务
│   └── front.py            # Streamlit 前端界面
├── models/             # 预训练模型文件 (.pkl)
├── tests/              # 单元测试代码
├── run_services.bat    # Windows 一键启动脚本
├── .env                # 配置文件
├── environment.yml     # Conda 环境依赖
├── AGENTS.md           # 开发者与 AI 代理技术指南
└── README.md           # 项目说明文档
```

---

## 🛠️ 开发与贡献

详细的代码规范、业务逻辑约束及优化路线图，请参阅 [AGENTS.md](./AGENTS.md)。

---
**免责声明**: 本项目预测结果仅供科学研究参考，**严禁**将其作为野外蘑菇食用的判定依据。
