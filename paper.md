# 基于视觉大模型与集成学习的蘑菇毒性预测系统

## 1. 系统概述

### 1.1 机器学习的发展

机器学习作为人工智能的核心分支，其发展历程经历了从符号主义到连接主义、从浅层模型到深度学习的深刻变革。在早期的统计学习阶段，以逻辑回归、支持向量机（SVM）和决策树为代表的浅层模型在结构化数据上取得了广泛应用。2000年代，集成学习方法（Ensemble Learning）的兴起标志着机器学习在结构化数据领域的重要突破，其中Bagging和Boosting两大范式成为研究热点。

Boosting方法的核心思想是通过迭代训练一系列弱学习器，每个后续模型聚焦于前序模型的预测残差，从而逐步提升整体预测精度。Freund和Schapire于1997年提出的AdaBoost算法开创了这一方向，随后Friedman在1999年将梯度下降引入Boosting框架，形成了梯度提升机（Gradient Boosting Machine, GBM）。在此基础上，陈天奇于2014年发布的XGBoost（eXtreme Gradient Boosting）通过引入二阶泰勒展开近似损失函数、正则化项控制模型复杂度以及并行化训练等创新，显著提升了梯度提升树的计算效率和泛化能力。2017年，微软研究院推出的LightGBM（Light Gradient Boosting Machine）则进一步通过直方图算法（Histogram-based Algorithm）、基于梯度的单边采样（Gradient-based One-Side Sampling, GOSS）和互斥特征捆绑（Exclusive Feature Bundling, EFB）三大核心优化，在大规模数据场景下实现了训练速度的量级提升。

XGBoost与LightGBM在树的生长策略上存在本质差异。XGBoost采用Level-wise（按层生长）策略，逐层分裂所有叶节点，倾向于构建较为均衡的树结构，虽有利于防止过拟合但计算开销较大；LightGBM采用Leaf-wise（按叶生长）策略，每次选择增益最大的叶节点进行分裂，在相同深度下能获得更低的损失值，但在小数据集上更易过拟合，需通过`max_depth`和`min_child_samples`等参数加以约束。二者均支持L1/L2正则化、缺失值自动处理和GPU加速训练，已成为Kaggle等数据科学竞赛中结构化数据任务的主流选择。

在集成策略方面，Stacking（堆叠泛化）作为一种高级集成学习技术，通过构建两层模型架构——基础层（Base Layer）和元层（Meta Layer）——来整合多个异构基模型的预测输出。基础层各模型通过交叉验证生成样本外预测（Out-of-Fold Predictions），元模型以这些预测概率作为输入特征进行二次学习，从而实现比单一模型或简单加权平均更优的泛化性能。本项目正是基于XGBoost和LightGBM作为基础层分类器，以逻辑回归作为元模型，构建Stacking集成架构来完成蘑菇毒性的二分类预测任务。

### 1.2 多模态模型的发展

多模态模型（Multimodal Model）是指能够同时处理和理解多种模态信息（如文本、图像、音频等）的机器学习模型。其发展根植于深度学习在计算机视觉和自然语言处理两大领域的突破性进展。

在计算机视觉领域，从AlexNet（2012）到ResNet（2015），卷积神经网络在图像分类、目标检测等任务上不断刷新性能纪录。Vision Transformer（ViT, 2020）的提出则将自然语言处理中的Transformer架构引入视觉领域，通过将图像分割为固定大小的图块（Patch）并作为序列输入，实现了与CNN架构相当甚至更优的视觉表征能力。在自然语言处理领域，从Word2Vec到BERT（2018），再到GPT系列大语言模型，文本理解与生成能力实现了质的飞跃。

多模态模型的关键突破在于跨模态对齐（Cross-modal Alignment）技术。CLIP（Contrastive Language-Image Pre-training, 2021）通过对比学习在4亿图文对上进行训练，实现了图像与文本在共享嵌入空间中的语义对齐，为后续视觉语言模型（Vision-Language Model, VLM）的发展奠定了基础。LLaVA（Large Language-and-Vision Assistant, 2023）等项目进一步将视觉编码器与大语言模型结合，使模型能够以自然语言描述图像内容并回答视觉相关问题。

视觉语言模型（VLM）是多模态模型的重要子类，专门处理图像与文本的联合理解任务。VLM通常由视觉编码器（如ViT或CLIP的视觉部分）和语言解码器（如LLaMA、Qwen等）组成，通过投影层或交叉注意力机制实现视觉特征到语言空间的映射。在零样本（Zero-shot）场景下，VLM能够通过精心设计的提示词（Prompt）引导模型对图像进行结构化分析，无需针对特定任务进行微调即可完成特征提取。

本项目利用VLM的零样本图像理解能力，通过LM Studio在本地部署量化视觉语言模型，以结构化提示词引导模型从蘑菇图片中自动提取20项形态学特征，实现了从非结构化图像到结构化特征数据的自动化转换，为后续分类模型提供输入。

### 1.3 视觉模型、分类模型和提示词工程

本系统的核心创新在于将视觉大模型的图像理解能力与传统机器学习分类模型的高效推理能力进行有机结合，形成"视觉特征提取→人工核验修正→集成分类预测"的完整闭环。

**视觉模型的角色**：VLM在本系统中承担图像到特征的映射任务。用户上传蘑菇图片后，VLM通过视觉编码器提取图像的深层视觉表征，语言解码器根据提示词约束生成结构化的JSON特征描述。这一过程无需针对蘑菇数据集进行专门训练，完全依赖模型的零样本泛化能力和提示词的引导质量。

**分类模型的角色**：XGBoost和LightGBM作为梯度提升树模型，在结构化表格数据上具有卓越的分类性能。相比深度神经网络，梯度提升树在中小规模表格数据上训练更快、可解释性更强，且对特征工程的要求相对较低。本项目使用基于Kaggle大规模蘑菇数据集（约311万条样本）训练的Stacking集成模型，在验证集上取得了MCC（Matthews Correlation Coefficient）约0.984的高分。

**提示词工程（Prompt Engineering）**：提示词工程是引导大语言模型产生期望输出的关键技术，在VLM应用中尤为重要。本系统设计了一套严格的结构化提示词，包含以下核心要素：（1）任务描述——明确要求模型分析蘑菇图片并输出JSON对象；（2）特征枚举映射表——以Markdown表格形式列出20项特征的键名、允许值和含义，确保模型输出严格遵循预定义的编码体系；（3）输出格式约束——要求模型仅输出JSON，不包含任何解释文本；（4）示例输出——提供一个标准JSON样例，规范字段格式和空值表示方式（使用`null`而非Python的`None`）。该提示词设计通过降低温度参数（`temperature=0.1`）抑制随机性，确保输出格式的高度稳定性。

## 2. 开发环境与工具

### 2.1 Python环境

本项目基于Python 3.10开发，采用Conda进行环境管理。Python 3.10在类型注解（Type Hinting）方面引入了重要改进，包括`Union`语法的简化（使用`X | Y`替代`Union[X, Y]`）和`ParamSpec`等高级类型工具，这些特性为项目的类型安全提供了良好支撑。

项目通过`environment.yml`文件定义完整的Conda环境依赖，主要依赖包括：

| 类别 | 核心依赖 | 版本 | 用途 |
|------|---------|------|------|
| Web框架 | FastAPI | 0.133.0 | 后端API服务构建 |
| Web框架 | Uvicorn | 0.41.0 | ASGI服务器 |
| Web框架 | Streamlit | 1.54.0 | 前端交互界面 |
| 机器学习 | XGBoost | 3.2.0 | 梯度提升树分类器 |
| 机器学习 | LightGBM | 4.6.0 | 梯度提升树分类器 |
| 机器学习 | Scikit-learn | 1.7.2 | 数据预处理与评估指标 |
| 数据处理 | Pandas | 2.3.3 | 数据框操作 |
| 数据处理 | NumPy | 2.2.6 | 数值计算 |
| VLM接口 | OpenAI | 2.24.0 | 对接LM Studio |
| 数据验证 | Pydantic | 2.12.5 | API请求模型验证 |
| 配置管理 | python-dotenv | 1.2.1 | 环境变量加载 |
| 测试框架 | Pytest | 9.0.2 | 单元测试 |
| 图像处理 | Pillow | 12.1.1 | 图像格式转换 |

环境创建命令为`conda env create -f environment.yml`，激活命令为`conda activate mushroom`。

### 2.2 LM Studio和OpenAI接口

LM Studio是一款本地大模型部署与管理工具，支持加载GGUF格式的量化模型，并提供与OpenAI API兼容的REST接口。其核心优势在于：（1）完全本地化运行，无需将数据上传至云端，保障数据隐私；（2）提供`/v1/chat/completions`等标准端点，可直接使用OpenAI官方SDK进行调用；（3）支持多模态模型加载，可运行具备视觉理解能力的量化VLM。

本项目中，VLM特征提取服务（`imgprocess_api.py`）通过OpenAI Python SDK与LM Studio通信。具体实现方式为：创建`OpenAI`客户端实例时，将`base_url`参数设置为LM Studio的本地服务地址（默认`http://localhost:1234/v1`），`api_key`设置为`"lm-studio"`（LM Studio不验证API密钥，但需提供非空值以满足SDK格式要求）。调用`client.chat.completions.create()`方法时，`model`参数设为`"local-model"`（LM Studio会自动路由至当前加载的模型），消息内容包含文本提示词和Base64编码的图片。

LM Studio的OpenAI兼容接口支持多模态消息格式，图片通过`image_url`类型的消息内容传入，URL格式为`data:image/jpeg;base64,{base64_string}`。这种设计使得项目代码无需依赖特定的模型推理库，仅通过标准的OpenAI SDK即可实现与任意兼容模型的对接，具有高度的灵活性和可替换性。

### 2.3 FastAPI和Uvicorn

FastAPI是本项目中两个后端API服务的核心框架。FastAPI基于Starlette构建，采用ASGI（Asynchronous Server Gateway Interface）规范，原生支持异步请求处理。其核心特性包括：（1）基于Python类型注解的自动请求参数解析与数据验证（借助Pydantic）；（2）自动生成OpenAPI文档和交互式Swagger UI；（3）异步端点定义（`async def`）实现高并发处理。

Uvicorn作为ASGI服务器，负责接收HTTP请求并将其传递给FastAPI应用处理。Uvicorn基于`uvloop`（高性能异步事件循环）和`httptools`（高速HTTP解析器）实现，在Python Web服务器性能基准测试中表现优异。在本项目中，分类器API和VLM API分别运行在8000和8001端口，通过`uvicorn.run()`启动，监听`0.0.0.0`以接受所有网络接口的请求。

项目采用`.env`文件配合`python-dotenv`实现配置解耦。环境变量包括`CLASSIFIER_PORT`（分类器端口）、`VLM_PORT`（VLM端口）、`LM_STUDIO_BASE_URL`（LM Studio地址）、`MODEL_DIR`（模型文件目录）、`VLM_API_URL`和`CLASSIFIER_API_URL`（前端调用的API地址），所有配置项均设有默认值，确保系统在无`.env`文件时仍可使用默认配置运行。

### 2.4 Kaggle数据集

本项目使用的数据集来源于Kaggle Playground Series Season 4 Episode 8（S4E8）竞赛——"Binary Prediction of Poisonous Mushrooms"。该数据集由深度学习模型基于UCI蘑菇数据集（Secondary Mushroom Dataset）合成生成，包含以下文件：

- **train.csv**：训练集，包含3,116,945条样本，22列（含`id`和`class`），其中`class`为目标变量（`e`=可食用，`p`=有毒）。
- **test.csv**：测试集，包含2,077,964条样本，21列（不含`class`）。
- **sample_submission.csv**：提交样例文件。

数据集包含20个特征字段，其中3个为数值型（`cap-diameter`、`stem-height`、`stem-width`），17个为类别型。类别型特征存在大量缺失值（以NaN表示），部分特征（如`veil-type`、`spore-print-color`）的缺失率极高。数据集的类别编码采用单字母缩写体系，例如`cap-shape`的`b`表示钟形（bell）、`x`表示凸面（convex）等，该编码体系与UCI原始数据集保持一致。

## 3. 需求分析

### 3.1 前景调研与市场分析

蘑菇中毒是全球性的公共卫生问题。据世界卫生组织统计，全球每年因误食毒蘑菇导致的中毒事件数以万计，其中相当比例的案例发生在缺乏专业真菌学知识的普通人群中。传统的蘑菇鉴定依赖专家经验，需要观察菌盖形状、菌褶颜色、菌柄特征、孢子印颜色等数十项形态学指标，专业门槛极高。

随着智能手机的普及和计算机视觉技术的发展，基于图像识别的蘑菇鉴定应用具有广阔的市场前景。然而，现有的蘑菇识别应用大多采用端到端的图像分类模型，存在以下局限：（1）分类结果缺乏可解释性，用户无法了解判定依据；（2）模型对图片质量敏感，在非标准拍摄条件下识别率显著下降；（3）无法处理模型未见过的蘑菇品种。

本系统采用"视觉特征提取+结构化分类"的两阶段架构，有效解决了上述问题。第一阶段通过VLM将图像转化为可解释的形态学特征，用户可直观理解并人工核验；第二阶段基于大规模数据训练的集成分类模型进行毒性判定，确保预测精度。这种设计在保持高准确率的同时，提供了良好的可解释性和人机交互性。

### 3.2 任务需求分析

本系统的核心任务是将用户上传的蘑菇图片转化为"可食用"或"有毒"的二分类判定结果，并给出中毒概率。任务可分解为以下子任务：

1. **图像特征提取**：从蘑菇图片中自动识别20项形态学特征，包括菌盖直径、菌盖形状、菌盖表面、菌盖颜色、变色反应、菌褶附着方式、菌褶间距、菌褶颜色、菌柄高度、菌柄宽度、菌柄根部形态、菌柄表面、菌柄颜色、菌幕类型、菌幕颜色、是否有环、菌环类型、孢子印颜色、生育地和季节。

2. **特征核验与修正**：提供用户界面展示AI提取的特征，允许用户对明显错误的识别结果进行手动修正，确保输入分类模型的特征数据准确可靠。

3. **毒性分类预测**：基于修正后的特征数据，通过Stacking集成模型计算中毒概率，给出最终判定结果。

4. **结果可视化展示**：以直观的方式展示预测结果，包括毒性判定、中毒概率和完整特征数据。

### 3.3 功能需求分析

基于上述任务分解，系统需实现以下功能模块：

| 模块 | 功能描述 | 实现方式 |
|------|---------|---------|
| 图片上传 | 支持JPG/JPEG/PNG格式图片上传 | Streamlit `file_uploader` |
| AI特征提取 | 调用VLM自动解析20项形态特征 | FastAPI + OpenAI SDK + LM Studio |
| 特征展示 | 以中英文对照表单展示提取结果 | Streamlit表单组件 |
| 人工修正 | 允许用户修改数值和类别特征 | Streamlit `number_input`/`selectbox` |
| 毒性预测 | 基于集成模型计算中毒概率 | FastAPI + XGBoost/LightGBM/LR |
| 结果展示 | 动态展示判定结果和概率 | Streamlit `success`/`error`/`progress` |
| 配置管理 | 端口、URL、模型路径等配置解耦 | python-dotenv + `.env`文件 |
| 缓存优化 | 避免重复调用VLM | Streamlit `@st.cache_data` |
| 单元测试 | 验证特征预处理逻辑正确性 | Pytest |

## 4. 设计内容和实现

### 4.1 总体设计与模块化处理

系统采用三层微服务架构，由三个独立进程组成，通过HTTP协议进行通信：

```
┌─────────────────────────────────────────────────────────┐
│                    用户界面层 (Streamlit)                  │
│                    front.py (端口 8501)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ 图片上传  │→│ 特征核验  │→│ 毒性预测  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└────────┬───────────────┬───────────────────────────────┘
         │               │
    HTTP POST        HTTP POST
    /analyze-image   /predict
         │               │
┌────────▼───────┐ ┌─────▼────────────────────────────────┐
│  VLM特征提取服务 │ │        分类器推理服务                   │
│  imgprocess_api │ │        classifier_api.py (端口 8000)  │
│  (端口 8001)    │ │  ┌──────┐ ┌──────┐ ┌──────────┐     │
│                 │ │  │LightGBM│ │XGBoost│ │MetaModel │     │
│  LM Studio ←── │ │  └──┬───┘ └──┬───┘ └─────┬────┘     │
│  (本地VLM)      │ │     └────┬────┘           │          │
└─────────────────┘ │          └────────┬───────┘          │
                    │    Stacking集成推理  │                  │
                    └────────────────────┘──────────────────┘
```

**模块间交互流程**：

1. 用户在Streamlit前端上传蘑菇图片，点击"AI提取特征"按钮。
2. 前端将图片编码为Base64字符串，通过HTTP POST请求发送至VLM API（`/analyze-image`端点）。
3. VLM API将Base64图片与结构化提示词组合，通过OpenAI SDK调用本地LM Studio部署的视觉语言模型。
4. VLM返回包含20项特征的JSON对象，前端解析并在表单中展示，支持用户手动修正。
5. 用户确认特征后点击"确认并预测毒性"，前端将特征数据通过HTTP POST发送至分类器API（`/predict`端点）。
6. 分类器API对输入数据进行预处理（数值转换、类别编码、空值映射），依次通过LightGBM和XGBoost基模型获取预测概率，再由逻辑回归元模型进行Stacking集成推理。
7. 分类器返回预测类别和中毒概率，前端动态展示结果。

该架构的优势在于：（1）服务解耦——三个进程独立部署和扩展，任一服务崩溃不影响其他服务；（2）技术异构——前端和后端可采用不同技术栈，通过标准HTTP协议通信；（3）灵活替换——VLM模型和分类模型可独立替换，无需修改其他模块代码。

### 4.2 多模态量化模型和提示词工程前置处理

#### 4.2.1 VLM API服务实现

VLM特征提取服务由`imgprocess_api.py`实现，核心功能是接收Base64编码的蘑菇图片，调用本地VLM提取形态学特征，返回结构化JSON结果。

**请求模型定义**：使用Pydantic的`BaseModel`定义`ImageRequest`类，仅包含一个`image_base64`字段（`str`类型），确保请求体的类型安全验证。

**VLM调用流程**：

```python
client = OpenAI(base_url=lm_studio_url, api_key="lm-studio")

response = client.chat.completions.create(
    model="local-model",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{request.image_base64}"}}
        ]
    }],
    max_tokens=800,
    temperature=0.1
)
```

关键参数说明：
- `model="local-model"`：LM Studio的通配模型名，自动路由至当前加载的模型。
- `max_tokens=800`：限制输出长度，20项特征的JSON输出通常在300-500 token范围内。
- `temperature=0.1`：极低温度值，抑制采样随机性，确保输出格式的一致性和稳定性。

**输出后处理**：VLM的原始输出可能包含Markdown代码块标记（如` ```json `）或前导说明文字，因此需要通过正则表达式`r"(\{.*\})"`提取其中的JSON对象部分。此外，VLM可能输出Python风格的`None`而非JSON标准的`null`，代码中通过字符串替换`clean_json_str.replace(": None", ": null")`进行兼容性修正。

**异常处理**：对JSON解析失败和VLM调用失败分别抛出`HTTPException`，返回500状态码和详细错误信息，确保前端能够捕获并展示错误提示。

#### 4.2.2 提示词工程设计

提示词是VLM特征提取质量的关键决定因素。本项目的提示词采用"任务描述+特征枚举表+输出格式约束+示例输出"四段式结构：

**任务描述段**：明确要求模型分析蘑菇图片并输出标准JSON对象，限定输出范围。

**特征枚举映射表**：以Markdown表格形式列出全部20项特征，每项包含字段序号、中文名称、键名（Key）和允许值（Allowed Values）。该设计确保模型输出的键名和值严格匹配训练数据的编码体系。例如：

| Field | Key | Allowed Values |
|:---|:---|:---|
| 1. 菌盖直径 | `cap-diameter` | Float (单位: cm) |
| 2. 菌盖形状 | `cap-shape` | b:bell, c:conical, x:convex, f:flat, k:knobbed, s:sunken |
| ... | ... | ... |

对于无法观察到的特征，提示词明确要求填入`null`，而非猜测或省略字段。

**输出格式约束**：要求"仅输出JSON对象，不输出任何解释文本"，避免模型添加额外说明干扰JSON解析。

**示例输出**：提供一个完整的JSON样例，包含数值字段的浮点格式、类别字段的编码值和空值的`null`表示，为模型提供明确的输出模板。

### 4.3 分类模型设计与流水线执行

#### 4.3.1 数据预处理

分类模型的训练数据预处理在`train_model.ipynb`中完成，主要包括以下步骤：

**目标变量编码**：将`class`列的`e`（可食用）映射为0，`p`（有毒）映射为1，形成二分类标签向量`y`。

**LabelEncoder编码**：对17个类别型特征使用scikit-learn的`LabelEncoder`进行编码。关键实现细节：编码器在训练集和测试集的合并数据上拟合（`le.fit(pd.concat([train_df[col], test_df[col]])`），确保编码空间覆盖所有可能出现的类别值。编码完成后，将所有编码器保存至`label_encoders.pkl`文件，供推理阶段使用。

**空值处理约定**：训练阶段通过`astype(str)`将NaN转换为字符串`"nan"`，这意味着`"nan"`被LabelEncoder视为一个独立的类别。这一约定在推理阶段必须严格遵循——所有`None`/`null`值必须显式转换为字符串`"nan"`后再进行编码，否则将导致编码错误或预测失效。

**数据划分**：采用`train_test_split`以80:20的比例划分训练集和验证集，使用`stratify=y`确保正负样本比例一致。最终训练集包含2,493,556条样本，验证集包含623,389条样本。

#### 4.3.2 超参数优化

本项目使用Optuna框架进行超参数优化。Optuna采用Tree-structured Parzen Estimator（TPE）算法，该算法属于贝叶斯优化方法，通过构建两个概率模型——l(x)（表现较好的试验参数分布）和g(x)（表现较差的试验参数分布）——来指导搜索方向，选择使l(x)/g(x)最大化的参数作为下一轮试验的候选值。相比网格搜索和随机搜索，TPE能在更少的试验次数内找到更优的参数组合。

**LightGBM超参数搜索空间**：

| 参数 | 搜索范围 | 说明 |
|------|---------|------|
| `learning_rate` | [0.01, 0.1] (log) | 学习率，控制每棵树的贡献权重 |
| `num_leaves` | [20, 255] | 叶节点数，控制模型复杂度 |
| `max_depth` | [3, 12] | 树的最大深度 |
| `min_child_samples` | [5, 100] | 叶节点最小样本数，防止过拟合 |
| `subsample` | [0.5, 1.0] | 样本采样比例 |
| `colsample_bytree` | [0.5, 1.0] | 特征采样比例 |

固定参数：`objective='binary'`、`metric='binary_logloss'`、`device='gpu'`、`n_estimators=500`。

**XGBoost超参数搜索空间**：

| 参数 | 搜索范围 | 说明 |
|------|---------|------|
| `learning_rate` | [0.01, 0.1] (log) | 学习率 |
| `max_depth` | [3, 12] | 树的最大深度 |
| `min_child_weight` | [1, 10] | 叶节点最小权重和 |
| `subsample` | [0.5, 1.0] | 样本采样比例 |
| `colsample_bytree` | [0.5, 1.0] | 特征采样比例 |
| `gamma` | [1e-8, 1.0] (log) | 分裂最小增益阈值 |

固定参数：`objective='binary:logistic'`、`eval_metric='logloss'`、`tree_method='hist'`、`device='cuda'`、`n_estimators=500`。

两个模型均进行50轮Optuna试验，优化目标为验证集上的MCC（Matthews Correlation Coefficient），方向为最大化。

**最优参数结果**：

LightGBM最优参数：`learning_rate=0.0457`、`num_leaves=254`、`max_depth=12`、`min_child_samples=98`、`subsample=0.8867`、`colsample_bytree=0.5005`，验证集MCC=0.9847。

XGBoost最优参数：`learning_rate=0.0547`、`max_depth=12`、`min_child_weight=2`、`subsample=0.8696`、`colsample_bytree=0.5018`、`gamma=5.13e-08`，验证集MCC=0.9848。

#### 4.3.3 Stacking集成模型训练

Stacking集成模型的训练采用Out-of-Fold（OOF）策略，流程如下：

**Step 1：基模型全量训练**。使用Optuna搜索到的最优参数，分别在完整训练集上训练LightGBM和XGBoost模型，保存为`lgb_model.pkl`和`xgb_model.pkl`。

**Step 2：OOF预测生成**。使用5折分层交叉验证（StratifiedKFold, `n_splits=5`, `random_state=42`），对每个基模型执行以下操作：
- 在每折中，使用4折数据训练模型，在剩余1折上生成预测概率。
- 将5折的验证集预测拼接为完整的OOF预测向量`oof_lgb`/`oof_xgb`（长度等于训练集样本数）。
- 同时在测试集上生成预测概率，取5折平均值作为测试集预测`test_lgb`/`test_xgb`。

OOF预测结果：
- LightGBM各折MCC：0.9822、0.9820、0.9822、0.9817、0.9821，整体OOF MCC=0.9820。
- XGBoost各折MCC：0.9847、0.9846、0.9846、0.9845、0.9847，整体OOF MCC=0.9846。

**Step 3：元模型训练**。将两个基模型的OOF预测概率按列拼接为元特征矩阵`X_meta = np.column_stack([oof_lgb, oof_xgb])`，使用逻辑回归（LogisticRegression）作为元模型在`X_meta`上训练。逻辑回归的权重向量为`[4.985, 5.770]`，截距为`-5.121`，表明XGBoost的预测概率在最终决策中获得了略高于LightGBM的权重。元模型保存为`meta_model.pkl`。

Stacking集成后的OOF MCC为0.9843，与单一XGBoost的0.9846相近，但Stacking通过融合两个异构模型的预测，在理论上有更好的泛化鲁棒性。

#### 4.3.4 推理阶段预处理

推理阶段的预处理逻辑在`classifier_api.py`的`preprocess_data`函数中实现，必须严格与训练阶段的预处理保持一致：

1. **列补全**：确保20个特征列全部存在，缺失列填充`None`。
2. **数值列处理**：对`cap-diameter`、`stem-height`、`stem-width`三个数值列使用`pd.to_numeric(errors='coerce')`强制转换为浮点数，无法转换的值变为NaN（XGBoost和LightGBM原生支持NaN）。
3. **类别列空值映射**：将所有类别列通过`astype(str)`转换为字符串，其中`None`→`"None"`、`NaN`→`"nan"`。随后执行统一映射：`replace({'None': 'nan', 'nan': 'nan', '<NA>': 'nan'})`，确保所有空值表示统一为`"nan"`字符串。
4. **未见类别处理**：对于LabelEncoder中未见过的新类别值，优先映射为`"nan"`（如果`"nan"`在编码器类别中），否则映射为编码器的第一个类别（`le.classes_[0]`）作为Fallback。
5. **编码与排序**：使用保存的LabelEncoder对类别列进行编码，最终按`feature_order`列表的严格顺序排列所有列。

#### 4.3.5 Stacking推理流程

分类器API的`/predict`端点实现Stacking推理：

```python
lgb_proba = lgb_model.predict_proba(X)[:, 1]
xgb_proba = xgb_model.predict_proba(X)[:, 1]
meta_features = np.column_stack([lgb_proba, xgb_proba])
final_proba = meta_model.predict_proba(meta_features)[:, 1]
final_pred = meta_model.predict(meta_features)
```

推理结果以JSON数组形式返回，每个元素包含`id`、`predicted_class`（`"e"`或`"p"`）和`probability_poisonous`（0-1之间的浮点数）。

### 4.4 后端API接口设计与前端展示设计

#### 4.4.1 分类器API（classifier_api.py）

分类器API基于FastAPI构建，提供以下端点：

**POST /predict**：接收JSON数组格式的蘑菇特征数据，返回毒性预测结果。请求体为`List[Dict[str, Any]]`类型，每个字典包含`id`和20项特征字段。响应体为包含`id`、`predicted_class`和`probability_poisonous`的JSON数组。

**GET /**：健康检查端点，返回服务状态和版本信息。

**模型加载机制**：使用FastAPI的`@app.on_event("startup")`生命周期钩子，在服务启动时通过`joblib.load()`加载四个模型文件（`lgb_model.pkl`、`xgb_model.pkl`、`meta_model.pkl`、`label_encoders.pkl`）。模型目录通过环境变量`MODEL_DIR`配置，默认为项目根目录下的`models`文件夹。

**全局变量**：`lgb_model`、`xgb_model`、`meta_model`、`label_encoders`和`feature_order`作为模块级全局变量，在启动时初始化，供所有请求共享。`feature_order`列表严格定义了20项特征的输入顺序，该顺序必须与训练阶段一致。

#### 4.4.2 VLM API（imgprocess_api.py）

VLM API同样基于FastAPI构建，提供以下端点：

**POST /analyze-image**：接收包含Base64编码图片的JSON请求体，返回VLM提取的20项形态特征。请求体通过Pydantic的`ImageRequest`模型验证，仅包含`image_base64`字段。

**核心处理流程**：组装多模态消息（文本提示词+图片）→调用VLM→正则提取JSON→兼容性修正→解析返回。

#### 4.4.3 前端界面（front.py）

前端界面基于Streamlit构建，采用宽布局（`layout="wide"`），主要包含以下交互区域：

**图片上传区**：使用`st.file_uploader`组件，限制文件类型为JPG/JPEG/PNG。上传后通过Pillow打开图片，非RGB模式自动转换为RGB。

**AI特征提取**：点击按钮后，图片通过`io.BytesIO`缓冲区保存为JPEG格式，再通过`base64.b64encode`编码为字符串。调用`get_vlm_analysis`函数发送请求，该函数使用`@st.cache_data`装饰器缓存结果，避免对同一图片重复调用VLM。提取结果存储在`st.session_state['ai_result']`中，跨重渲染保持状态。

**特征核验表单**：使用`st.form`组件构建，20项特征按双列布局排列。数值型特征（`cap-diameter`、`stem-height`、`stem-width`）使用`st.number_input`展示，类别型特征使用`st.selectbox`展示。`MUSHROOM_MAPPING`字典定义了完整的特征映射配置，键为特征英文名，值为中文描述字符串（数值列）或编码-中文映射字典（类别列）。类别列的下拉选项包含"-- 未观察到 (null) --"作为空值选项，选择该项时对应字段传入`None`。

**毒性预测展示**：预测结果根据判定类别动态展示——毒蘑菇使用`st.error`红色警示，可食用使用`st.success`绿色提示。中毒概率通过`st.progress`进度条和百分比文字双重展示。完整特征数据可通过`st.expander`折叠面板查看。

**侧边栏**：提供操作指南，简述四步使用流程。

#### 4.4.4 一键启动脚本

`run_services.bat`为Windows平台的一键启动脚本，依次在三个独立的CMD窗口中启动分类器API、VLM API和Streamlit前端。脚本首先检查Conda是否可用，然后激活`mushroom`环境，最后使用`start`命令在新窗口中分别运行三个服务进程。

## 5. 测试和实验

### 5.1 分类模型对比测试

#### 5.1.1 评估指标选择

本项目选用Matthews相关系数（MCC）作为核心评估指标。MCC是二分类任务中综合考虑真阳性（TP）、假阳性（FP）、真阴性（TN）和假阴性（FN）的平衡指标，其计算公式为：

$$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

MCC的取值范围为[-1, +1]，其中+1表示完美预测，0表示随机预测，-1表示完全相反预测。相比准确率（Accuracy）和F1分数，MCC在类别不平衡场景下更为可靠，因为它同时考虑了混淆矩阵的四个象限，不会因多数类的高准确率而掩盖少数类的低召回率。Kaggle S4E8竞赛官方同样采用MCC作为评估指标。

#### 5.1.2 基模型性能对比

在5折交叉验证的OOF预测中，两个基模型的性能如下：

| 模型 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | 整体OOF MCC |
|------|--------|--------|--------|--------|--------|------------|
| LightGBM | 0.9822 | 0.9820 | 0.9822 | 0.9817 | 0.9821 | 0.9820 |
| XGBoost | 0.9847 | 0.9846 | 0.9846 | 0.9845 | 0.9847 | 0.9846 |

XGBoost在各折上均略优于LightGBM，整体MCC差距约为0.0026。两者均达到了0.98以上的MCC，表明在311万条训练数据上训练的梯度提升树模型已具备极高的分类能力。

#### 5.1.3 Stacking集成效果

Stacking集成后的OOF MCC为0.9843，介于LightGBM（0.9820）和XGBoost（0.9846）之间。虽然Stacking未显著超越单一XGBoost，但其核心价值在于：（1）通过融合两个异构模型的预测，降低了单一模型偏差的风险；（2）逻辑回归元模型的可解释性权重（LightGBM: 4.985, XGBoost: 5.770）清晰反映了两个基模型的相对贡献。

#### 5.1.4 Optuna超参数优化效果

LightGBM在50轮Optuna试验中，MCC从初始的0.8531（Trial 0）逐步提升至0.9847（Trial 39），体现了TPE算法的高效搜索能力。XGBoost的优化过程类似，MCC从0.9559（Trial 0）提升至0.9848（Trial 23）。两个模型的搜索空间均在较优区域收敛，表明50轮试验对于该任务已足够。

### 5.2 系统整体测试

#### 5.2.1 API功能测试

使用`test.ipynb`对分类器API进行端到端功能测试。测试数据包含一条完整的蘑菇特征记录（ID=3116946），其中包含多个`None`空值字段。测试流程为：构造测试数据→调用`/predict`端点→解析响应→展示结果。

测试结果：预测类别为"毒蘑菇(p)"，中毒概率99.56%，与预期一致。该测试验证了API的以下功能：（1）正确接收JSON输入；（2）正确处理空值字段；（3）Stacking推理流程正常执行；（4）返回格式符合预期。

#### 5.2.2 单元测试

项目使用Pytest框架编写单元测试，测试文件为`tests/test_preprocessing.py`。该测试针对`classifier_api.py`中的`preprocess_data`函数，验证以下关键逻辑：

- **空值映射**：输入数据中`cap-shape`为`None`时，预处理后应正确转换为字符串`"nan"`并通过LabelEncoder编码，而非产生空值异常。
- **列补全**：输入数据仅包含4个字段时，预处理后应补全为20列。
- **数值转换**：数值列应正确转换为浮点类型。
- **无空值残留**：预处理后的DataFrame不应包含任何空值。

测试使用`monkeypatch`机制替换全局变量`label_encoders`为`MockEncoder`（模拟编码器，`transform`方法返回全0），避免依赖真实模型文件。`MockEncoder`的`classes_`属性包含`'nan'`类别，确保空值映射逻辑可被正确测试。

#### 5.2.3 VLM特征提取测试

通过Streamlit前端上传蘑菇图片进行VLM特征提取的集成测试。测试验证了以下功能：（1）图片上传和Base64编码正常工作；（2）VLM API成功调用并返回JSON结果；（3）正则表达式正确提取JSON对象；（4）前端表单正确展示提取结果；（5）`@st.cache_data`缓存机制生效，重复请求不触发新的VLM调用。

## 6. 总结和展望

### 6.1 工作总结

本项目成功构建了一个基于视觉大模型与集成学习的蘑菇毒性预测系统，实现了从图像输入到毒性判定的完整流程。主要贡献包括：

1. **两阶段架构设计**：将视觉特征提取与分类推理解耦，VLM负责图像理解，梯度提升树负责结构化数据分类，充分发挥各自优势。
2. **Stacking集成策略**：基于LightGBM和XGBoost的OOF预测构建元特征，通过逻辑回归元模型实现概率融合，在311万条数据上达到MCC 0.984的高性能。
3. **结构化提示词工程**：设计了包含特征枚举映射表和输出格式约束的提示词，确保VLM输出严格匹配分类模型的输入编码体系。
4. **人工核验闭环**：前端提供特征复核与修正功能，弥补VLM零样本识别的潜在误差，提升系统可靠性。
5. **工程化实践**：通过`.env`配置解耦、Streamlit计算缓存、Pytest单元测试等手段，保障系统的可维护性和稳定性。

### 6.2 未来展望

1. **预测解释性增强**：集成SHAP（SHapley Additive exPlanations）或LIME（Local Interpretable Model-agnostic Explanations）方法，在前端展示影响毒性判定的关键形态特征及其贡献度，提升结果的可解释性。
2. **提示词优化**：通过Few-shot示例和思维链（Chain-of-Thought）技术优化VLM提示词，提高形态提取的零样本识别精度，减少人工修正的需求。
3. **输入校验**：在API层增加对输入数值的物理范围校验（如菌盖直径应在合理范围内），过滤异常输入，提升系统鲁棒性。
4. **Docker容器化**：编写`docker-compose.yml`实现多服务一键编排，简化部署流程，提升跨平台兼容性。
5. **纠错闭环**：增加用户反馈收集机制，将人工修正的特征数据作为微调数据集，持续优化VLM的特征提取能力。
6. **模型更新**：随着更强大的开源VLM发布，可替换LM Studio中加载的模型，提升特征提取的准确率和覆盖范围。

## 7. 谢辞

感谢Kaggle平台提供的Playground Series S4E8竞赛数据集，感谢开源社区提供的XGBoost、LightGBM、FastAPI、Streamlit、Optuna等优秀工具，感谢LM Studio团队提供的本地模型部署方案，使本项目得以实现。

## 8. 参考文献

[1] Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System[C]. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016: 785-794.

[2] Ke G, Meng Q, Finley T, et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree[C]. Advances in Neural Information Processing Systems 30 (NIPS 2017), 2017: 3146-3154.

[3] Wolpert D H. Stacked Generalization[J]. Neural Networks, 1992, 5(2): 241-259.

[4] Radford A, Kim J W, Hallacy C, et al. Learning Transferable Visual Models From Natural Language Supervision[C]. International Conference on Machine Learning (ICML), 2021: 8748-8763.

[5] Liu H, Li C, Wu Q, et al. Visual Instruction Tuning[C]. Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 2023.

[6] Akiba T, Sano S, Yanase T, et al. Optuna: A Next-generation Hyperparameter Optimization Framework[C]. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2019: 2623-2631.

[7] Chicco D, Jurman G. The Advantages of the Matthews Correlation Coefficient (MCC) over F1 Score and Accuracy in Binary Classification Evaluation[J]. BMC Genomics, 2020, 21(1): 6.

[8] Ramírez J, Flores M. FastAPI: Modern, Fast Web Framework for Building APIs with Python 3.7+[EB/OL]. https://fastapi.tiangolo.com, 2023.

[9] Kaggle. Playground Series - Season 4, Episode 8: Binary Prediction of Poisonous Mushrooms[EB/OL]. https://www.kaggle.com/competitions/playground-series-s4e8, 2024.

[10] LM Studio. OpenAI Compatibility API[EB/OL]. https://lmstudio.ai/docs, 2025.
