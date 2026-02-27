import os
import json
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ----------------------------
# 1. 初始化 FastAPI 应用
# ----------------------------

app = FastAPI(
    title="Mushroom Edibility Classifier",
    description="API for predicting mushroom edibility (edible/poisonous)",
    version="1.0"
)

# ----------------------------
# 2. 全局变量：模型和编码器（启动时加载）
# ----------------------------
lgb_model = None
xgb_model = None
meta_model = None
label_encoders = None
feature_order = [
    'cap-diameter', 'cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed',
    'gill-attachment', 'gill-spacing', 'gill-color', 'stem-height', 'stem-width',
    'stem-root', 'stem-surface', 'stem-color', 'veil-type', 'veil-color',
    'has-ring', 'ring-type', 'spore-print-color', 'habitat', 'season'
]

@app.on_event("startup")
def load_models():
    global lgb_model, xgb_model, meta_model, label_encoders
    # 从环境变量读取模型目录，默认为项目根目录下的 models
    model_dir = os.getenv("MODEL_DIR", os.path.join(os.getcwd(), "models"))
    
    try:
        # 加载所有模型
        lgb_model = joblib.load(os.path.join(model_dir, "lgb_model.pkl"))
        xgb_model = joblib.load(os.path.join(model_dir, "xgb_model.pkl"))
        meta_model = joblib.load(os.path.join(model_dir, "meta_model.pkl"))
        label_encoders = joblib.load(os.path.join(model_dir, "label_encoders.pkl"))
        
        print(f"✅ Models loaded successfully from {model_dir}")
    except Exception as e:
        print(f"❌ Failed to load models from {model_dir}: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


# ----------------------------
# 3. 预处理函数 (修正版)
# ----------------------------
def preprocess_data(input_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """将 JSON 输入转换为预处理后的 DataFrame"""
    df = pd.DataFrame(input_data)
    
    # 1. 确保所有列存在
    for col in feature_order:
        if col not in df.columns:
            df[col] = None  # 产生 NaN/None

    # 2. 显式处理数值列 (防止传入字符串导致模型报错)
    numeric_cols = ['cap-diameter', 'stem-height', 'stem-width']
    for col in numeric_cols:
        # 强制转为 float，无法转换的变为 NaN (模型能处理数值 NaN)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. 处理类别列
    categorical_cols = [col for col in feature_order if col not in numeric_cols]
    
    for col in categorical_cols:
        le = label_encoders[col]
        
        # -------------------------------------------------------
        # 修正逻辑：模拟训练时的 astype(str) 行为
        # -------------------------------------------------------
        # 训练时: .astype(str) 将 NaN 变成了 "nan"
        # 因此，这里我们要把 None/NaN 显式变成 "nan" 字符串以匹配编码器
        df[col] = df[col].astype(str) 
        
        # 处理 python 的 'None' 字符串问题 (pd.DataFrame(None) -> None -> str() -> 'None')
        # 如果你的训练数据来源于 CSV，空值通常是 float NaN -> 'nan'
        # 如果 JSON 传入 null，pandas 可能是 None -> 'None'
        # 统一映射一下：
        df[col] = df[col].replace({'None': 'nan', 'nan': 'nan', '<NA>': 'nan'})

        # 处理未见过的类别 (Handle Unseen Labels)
        # 逻辑：如果是已知类别 -> 保持
        #       如果是未知类别 -> 尝试归类为 'nan' (假设训练集有空值)
        #       如果 'nan' 也不在编码器里 (训练集该列全满) -> 归类为第一个已知类别 (Fallback)
        
        fallback_value = 'nan' if 'nan' in le.classes_ else le.classes_[0]
        
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else fallback_value)
        
        # 编码
        df[col] = le.transform(df[col])
    
    # 保持特征顺序
    return df[feature_order]

# ----------------------------
# 4. 预测接口
# ----------------------------
@app.post("/predict")
async def predict(data: List[Dict[str, Any]]):
    """
    接收 JSON 数组，返回预测结果
    
    示例输入:
    [
      {
        "id": 3116945,
        "cap-diameter": 8.64,
        "cap-shape": "x",
        "cap-surface": null,
        "cap-color": "n",
        "does-bruise-or-bleed": "t",
        "gill-attachment": null,
        "gill-spacing": null,
        "gill-color": "w",
        "stem-height": 11.13,
        "stem-width": 17.12,
        "stem-root": "b",
        "stem-surface": null,
        "stem-color": "w",
        "veil-type": "u",
        "veil-color": "w",
        "has-ring": "t",
        "ring-type": "g",
        "spore-print-color": null,
        "habitat": "d",
        "season": "a"
      }
    ]
    """
    if not data:
        raise HTTPException(status_code=400, detail="Empty input data")
    
    # 预处理
    X = preprocess_data(data)
    
    # 基模型预测
    lgb_proba = lgb_model.predict_proba(X)[:, 1]
    xgb_proba = xgb_model.predict_proba(X)[:, 1]
    
    # 元模型输入
    meta_features = np.column_stack([lgb_proba, xgb_proba])
    
    # 最终预测
    final_proba = meta_model.predict_proba(meta_features)[:, 1]
    final_pred = meta_model.predict(meta_features)
    
    # 构建结果
    results = []
    for i, item in enumerate(data):
        results.append({
            "id": item["id"],
            "predicted_class": "p" if final_pred[i] == 1 else "e",
            "probability_poisonous": float(final_proba[i])
        })
    
    return results

# ----------------------------
# 5. 健康检查
# ----------------------------
@app.get("/")
async def root():
    return {"status": "mushroom_classifier_api_running", "version": "1.0"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("CLASSIFIER_PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
