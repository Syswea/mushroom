import os
import json
import re
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

app = FastAPI()

# 从环境变量读取 LM Studio 地址，默认为本地 1234 端口
lm_studio_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
client = OpenAI(base_url=lm_studio_url, api_key="lm-studio")


class ImageRequest(BaseModel):
    image_base64: str

# 优化后的 Prompt：修正了示例中的 null 格式，并增强了约束
prompt = """
## Task
请分析上传的蘑菇图片，并识别其形态学特征。必须输出一个标准的 JSON 对象。

## Feature Mapping (Enumeration)
如果特征无法观察到，请务必填入 null。

| Field | Key | Allowed Values / Description |
| :--- | :--- | :--- |
| 1. 菌盖直径 | `cap-diameter` | Float (单位: cm) |
| 2. 菌盖形状 | `cap-shape` | b:bell, c:conical, x:convex, f:flat, k:knobbed, s:sunken |
| 3. 菌盖表面 | `cap-surface` | f:fibrous, g:grooves, y:scaly, s:smooth |
| 4. 菌盖颜色 | `cap-color` | n:brown, b:buff, c:cinnamon, g:gray, r:green, p:pink, u:purple, e:red, w:white, y:yellow |
| 5. 变色反应 | `does-bruise-or-bleed` | t:true, f:false |
| 6. 菌褶附着 | `gill-attachment` | a:attached, d:descending, f:free, n:notched |
| 7. 菌褶间距 | `gill-spacing` | c:close, w:crowded, d:distant |
| 8. 菌褶颜色 | `gill-color` | k:black, n:brown, b:buff, h:chocolate, g:gray, r:green, o:orange, p:pink, u:purple, e:red, w:white, y:yellow |
| 9. 菌柄高度 | `stem-height` | Float (单位: cm) |
| 10. 菌柄宽度 | `stem-width` | Float (单位: mm) |
| 11. 菌柄根部 | `stem-root` | b:bulbous, c:club, u:cup, e:equal, z:rhizomorphs, r:rooted, ?:missing |
| 12. 菌柄表面 | `stem-surface` | f:fibrous, y:scaly, k:silky, s:smooth |
| 13. 菌柄颜色 | `stem-color` | n:brown, b:buff, c:cinnamon, g:gray, o:orange, p:pink, e:red, w:white, y:yellow |
| 14. 菌幕类型 | `veil-type` | p:partial, u:universal |
| 15. 菌幕颜色 | `veil-color` | n:brown, o:orange, w:white, y:yellow |
| 16. 是否有环 | `has-ring` | t:true, f:false |
| 17. 菌环类型 | `ring-type` | c:cobwebby, e:evanescent, f:flaring, l:large, n:none, p:pendant, s:sheathing, z:zone |
| 18. 孢子印颜色 | `spore-print-color` | k:black, n:brown, b:buff, h:chocolate, g:green, o:orange, u:purple, w:white, y:yellow |
| 19. 生育地 | `habitat` | g:grasses, l:leaves, m:meadows, p:paths, u:urban, w:waste, d:woods |
| 20. 季节 | `season` | a:autumn, s:spring, u:summer, w:winter |

## Output Format
仅输出 JSON 对象，不输出任何解释文本。

## Example Output (JSON only)
{
    "cap-diameter": 5.0,
    "cap-shape": "x",
    "cap-surface": "s",
    "cap-color": "n",
    "does-bruise-or-bleed": "f",
    "gill-attachment": "f",
    "gill-spacing": null,
    "gill-color": "n",
    "stem-height": 10.0,
    "stem-width": 15.0,
    "stem-root": null,
    "stem-surface": "s",
    "stem-color": "b",
    "veil-type": null,
    "veil-color": "w",
    "has-ring": "t",
    "ring-type": "l",
    "spore-print-color": "n",
    "habitat": "d",
    "season": "a"
}
"""

@app.post("/analyze-image")
async def analyze_image(request: ImageRequest):
    try:
        response = client.chat.completions.create(
            model="local-model", 
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{request.image_base64}"}
                        },
                    ]
                }
            ],
            max_tokens=800,
            temperature=0.1, # 降低随机性，保证格式稳定
        )

        raw_content = response.choices[0].message.content.strip()

        # 使用正则表达式提取内容（处理模型可能输出的 Markdown 代码块或前导文字）
        json_match = re.search(r"(\{.*\})", raw_content, re.DOTALL)
        
        if json_match:
            clean_json_str = json_match.group(1)
            # 兼容性处理：防止模型输出 Python 的 None 而不是 JSON 的 null
            clean_json_str = clean_json_str.replace(": None", ": null")
            
            result_dict = json.loads(clean_json_str)
            return result_dict
        else:
            raise ValueError(f"Model failed to generate a valid JSON object. Raw: {raw_content}")

    except json.JSONDecodeError as je:
        raise HTTPException(status_code=500, detail=f"JSON Parsing Error: {str(je)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("VLM_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
