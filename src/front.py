import streamlit as st
from PIL import Image
import io
import base64
import requests
import time

# --- 1. 配置项 ---
VLM_API_URL = "http://127.0.0.1:8001/analyze-image"
CLASSIFIER_API_URL = "http://127.0.0.1:8000/predict"

# --- 1. 完整的中英文特征映射配置表 ---
# 确保这里的 Key 和 Value 代码与分类器训练时的标签编码一致
MUSHROOM_MAPPING = {
    "cap-diameter": "菌盖直径 (cap-diameter) (cm)", 
    "cap-shape": {
        "b": "钟形 (bell)", 
        "c": "圆锥形 (conical)", 
        "x": "凸面 (convex)", 
        "f": "平面 (flat)", 
        "k": "凸顶 (knobbed)", 
        "s": "凹陷 (sunken)",
        "o": "其他 (other)"
    },
    "cap-surface": {
        "f": "纤维状 (fibrous)", 
        "g": "沟槽状 (grooves)", 
        "y": "鳞片状 (scaly)", 
        "s": "光滑 (smooth)",
        "o": "其他 (other)"
    },
    "cap-color": {
        "n": "棕色 (brown)", 
        "b": "浅黄色 (buff)", 
        "c": "肉桂色 (cinnamon)", 
        "g": "灰色 (gray)", 
        "r": "绿色 (green)", 
        "p": "粉色 (pink)", 
        "u": "紫色 (purple)", 
        "e": "红色 (red)", 
        "w": "白色 (white)", 
        "y": "黄色 (yellow)",
        "o": "其他 (other)"
    },
    "does-bruise-or-bleed": {
        "t": "是 (true)", 
        "f": "否 (false)",
        "o": "其他 (other)"
    },
    "gill-attachment": {
        "a": "生 (attached)", 
        "d": "延生 (descending)", 
        "f": "离生 (free)", 
        "n": "凹生 (notched)",
        "o": "其他 (other)"
    },
    "gill-spacing": {
        "c": "密集 (close)", 
        "w": "拥挤 (crowded)", 
        "d": "稀疏 (distant)",
        "o": "其他 (other)"
    },
    "gill-color": {
        "k": "黑色 (black)", 
        "n": "棕色 (brown)", 
        "b": "浅黄色 (buff)", 
        "h": "巧克力色 (chocolate)", 
        "g": "灰色 (gray)", 
        "r": "绿色 (green)", 
        "o": "橙色 (orange)", 
        "p": "粉色 (pink)", 
        "u": "紫色 (purple)", 
        "e": "红色 (red)", 
        "w": "白色 (white)", 
        "y": "黄色 (yellow)",
        "o": "其他 (other)"
    },
    "stem-height": "菌柄高度 (stem-height) (cm)",
    "stem-width": "菌柄宽度 (stem-width) (mm)",
    "stem-root": {
        "b": "球茎状 (bulbous)", 
        "c": "棒状 (club)", 
        "u": "杯状 (cup)", 
        "e": "等大 (equal)", 
        "z": "根状菌索 (rhizomorphs)", 
        "r": "生根 (rooted)",
        "o": "其他 (other)"
    },
    "stem-surface": {
        "f": "纤维状 (fibrous)", 
        "y": "鳞片状 (scaly)", 
        "k": "丝状 (silky)", 
        "s": "光滑 (smooth)",
        "o": "其他 (other)"
    },
    "stem-color": {
        "n": "棕色 (brown)", 
        "b": "浅黄色 (buff)", 
        "c": "肉桂色 (cinnamon)", 
        "g": "灰色 (gray)", 
        "o": "橙色 (orange)", 
        "p": "粉色 (pink)", 
        "e": "红色 (red)", 
        "w": "白色 (white)", 
        "y": "黄色 (yellow)",
        "o": "其他 (other)"
    },
    "veil-type": {
        "p": "内幕 (partial)", 
        "u": "外幕 (universal)",
        "o": "其他 (other)"
    },
    "veil-color": {
        "n": "棕色 (brown)", 
        "o": "橙色 (orange)", 
        "w": "白色 (white)", 
        "y": "黄色 (yellow)",
        "o": "其他 (other)"
    },
    "has-ring": {
        "t": "有 (true)", 
        "f": "无 (false)",
        "o": "其他 (other)"
    },
    "ring-type": {
        "c": "蛛网状 (cobwebby)", 
        "e": "易逝 (evanescent)", 
        "f": "外翻 (flaring)", 
        "l": "大型 (large)", 
        "n": "无 (none)", 
        "p": "悬垂 (pendant)", 
        "s": "鞘状 (sheathing)", 
        "z": "环带 (zone)",
        "o": "其他 (other)"
    },
    "spore-print-color": {
        "k": "黑色 (black)", 
        "n": "棕色 (brown)", 
        "b": "浅黄色 (buff)", 
        "h": "巧克力色 (chocolate)", 
        "r": "绿色 (green)", 
        "o": "橙色 (orange)", 
        "u": "紫色 (purple)", 
        "w": "白色 (white)", 
        "y": "黄色 (yellow)",
        "o": "其他 (other)"
    },
    "habitat": {
        "g": "草地 (grasses)", 
        "l": "树叶 (leaves)", 
        "m": "草甸 (meadows)", 
        "p": "路径 (paths)", 
        "u": "城市 (urban)", 
        "w": "废弃地 (waste)", 
        "d": "森林 (woods)",
        "o": "其他 (other)"
    },
    "season": {
        "a": "秋季 (autumn)", 
        "s": "春季 (spring)", 
        "u": "夏季 (summer)", 
        "w": "冬季 (winter)",
        "o": "其他 (other)"
    }
}

# --- 3. 辅助函数 ---
def predict_toxicity(data_dict: dict):
    """调用分类器 API 预测毒性"""
    # 构造 API 要求的 List[Dict] 格式，并补上 id
    payload = [{**data_dict, "id": int(time.time())}]
    try:
        resp = requests.post(CLASSIFIER_API_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()[0] # 返回第一个预测结果
    except Exception as e:
        st.error(f"分类器调用失败: {e}")
        return None

# --- 4. Streamlit 页面布局 ---
st.set_page_config(page_title="蘑菇毒性全流程检测", layout="wide")
st.title("🍄 蘑菇特征识别与毒性智能预测")
st.markdown("---")

uploaded_file = st.file_uploader("第一步：上传蘑菇照片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    col_img, col_form = st.columns([1, 2])
    
    with col_img:
        st.image(image, caption="待分析样本", use_container_width=True)
        analyze_btn = st.button("🚀 第二步：AI 提取特征", use_container_width=True)

    if analyze_btn:
        with st.spinner("视觉模型正在解析形态..."):
            try:
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                response = requests.post(
                    VLM_API_URL,
                    json={"image_base64": img_str},
                    timeout=60
                )

                if response.status_code == 200:
                    st.session_state['ai_result'] = response.json()
                    st.toast("特征提取成功！请在右侧核对。", icon="✨")
                else:
                    st.error(f"VLM API 错误: {response.text}")
            except Exception as e:
                st.error(f"无法连接到视觉模型服务器: {e}")

    # --- 5. 人工复核与毒性预测 ---
    if 'ai_result' in st.session_state:
        ai_res = st.session_state['ai_result']
        
        with col_form:
            st.subheader("📝 第三步：特征核对与毒性检测")
            with st.form("refine_and_predict"):
                final_data = {}
                f_col1, f_col2 = st.columns(2)
                
                for i, (key, config) in enumerate(MUSHROOM_MAPPING.items()):
                    target_col = f_col1 if i % 2 == 0 else f_col2
                    current_ai_val = ai_res.get(key)

                    if isinstance(config, str):
                        # 处理数值列
                        val = target_col.number_input(
                            f"{key} ({config})", 
                            value=float(current_ai_val) if current_ai_val else 0.0
                        )
                        final_data[key] = val
                    else:
                        # 处理类别列
                        options_map = config
                        display_list = ["-- 未观察到 (null) --"] + list(options_map.values())
                        
                        default_idx = 0
                        if current_ai_val in options_map:
                            default_idx = display_list.index(options_map[current_ai_val])
                        
                        chosen_text = target_col.selectbox(f"{key}", options=display_list, index=default_idx)
                        
                        if chosen_text == "-- 未观察到 (null) --":
                            final_data[key] = None # 传输时会自动转为 JSON null
                        else:
                            code = [k for k, v in options_map.items() if v == chosen_text][0]
                            final_data[key] = code

                st.markdown("---")
                predict_btn = st.form_submit_button("🔥 确认并预测毒性", use_container_width=True)

                if predict_btn:
                    with st.spinner("正在综合多模型进行毒性评估..."):
                        # 调用分类器 API
                        prediction = predict_toxicity(final_data)
                        
                        if prediction:
                            st.markdown("### 🏆 预测结果")
                            prob = prediction["probability_poisonous"]
                            is_poisonous = prediction["predicted_class"] == "p"

                            # 动态展示 UI
                            if is_poisonous:
                                st.error(f"**判定结果：毒蘑菇 (Poisonous)**")
                                st.progress(prob)
                                st.write(f"中毒概率：{prob:.2%}")
                            else:
                                st.success(f"**判定结果：可食用 (Edible)**")
                                st.progress(prob)
                                st.write(f"中毒概率：{prob:.2%}")
                            
                            with st.expander("查看完整特征数据"):
                                st.json(final_data)

st.sidebar.markdown("""
### 操作指南
1. **上传图片**：选择清晰的蘑菇全貌图。
2. **AI 分析**：提取菌盖、菌柄等 20 项形态特征。
3. **人工复核**：由于视觉模型可能存在误差，请手动修正明显错误的特征。
4. **毒性预测**：点击按钮，后端分类器将基于集成学习模型给出毒性判断。
""")