import pytest
import pandas as pd
import numpy as np
from src.classifier_api import preprocess_data

class MockEncoder:
    def __init__(self, classes):
        self.classes_ = classes
    def transform(self, data):
        return [0] * len(data) # 简化模拟

def test_preprocess_data_nan_handling(monkeypatch):
    # 模拟 label_encoders
    mock_encoders = {
        col: MockEncoder(['nan', 'x', 'y']) 
        for col in [
            'cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed',
            'gill-attachment', 'gill-spacing', 'gill-color', 'stem-root', 
            'stem-surface', 'stem-color', 'veil-type', 'veil-color',
            'has-ring', 'ring-type', 'spore-print-color', 'habitat', 'season'
        ]
    }
    
    # 使用 monkeypatch 设置全局变量
    import src.classifier_api
    monkeypatch.setattr(src.classifier_api, "label_encoders", mock_encoders)
    
    input_data = [{
        "cap-diameter": 5.0,
        "cap-shape": None, # 测试 None 转 "nan"
        "stem-height": 10.0,
        "stem-width": 15.0
    }]
    
    df = preprocess_data(input_data)
    
    # 验证所有的列都存在
    assert len(df.columns) == 20
    # 验证数值列已转换
    assert df['cap-diameter'].iloc[0] == 5.0
    # 验证类别列已处理 (由于 MockEncoder 返回 0，我们主要看 preprocess_data 内部是否报错)
    assert not df.isnull().any().any()
