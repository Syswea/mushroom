# %%
import joblib
import pandas as pd

# %%
# 加载三个模型 lgb xgb meta
lgb_model = joblib.load('models/lgb_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')
meta_model = joblib.load('models/meta_model.pkl')

# %%
# 给出一组数据并进行预测
# id,cap-diameter,cap-shape,cap-surface,cap-color,does-bruise-or-bleed,gill-attachment,gill-spacing,gill-color,stem-height,stem-width,stem-root,stem-surface,stem-color,veil-type,veil-color,has-ring,ring-type,spore-print-color,habitat,season
# 3116945,8.64,x,,n,t,,,w,11.13,17.12,b,,w,u,w,t,g,,d,a
data = pd.DataFrame([{
    'id': 3116945,
    'cap-diameter': 8.64,
    'cap-shape': 'x',
    'cap-surface': '',
    'cap-color': 'n',
    'does-bruise-or-bleed': 't',
    'gill-attachment': '',
    'gill-spacing': '',
    'gill-color': 'w',
    'stem-height': 11.13,
    'stem-width': 17.12,
    'stem-root': 'b',
    'stem-surface': '',
    'stem-color': 'w',
    'veil-type': 'u',
    'veil-color': 'w',
    'has-ring': 't',
    'ring-type': 'g',
    'spore-print-color': '',
    'habitat': 'd',
    'season': 'a'
}])

# %%
# 先用 lgb 和 xgb 分别预测
lgb_pred = lgb_model.predict_proba(data)[:, 1]
xgb_pred = xgb_model.predict_proba(data)[:, 1]
print(f'LGB Prediction: {lgb_pred}')
print(f'XGB Prediction: {xgb_pred}')

# %%
# 将两个模型的预测结果作为 meta 模型的输入进行最终预测
meta_input = pd.DataFrame({
    'lgb_pred': lgb_pred,
    'xgb_pred': xgb_pred
})
final_pred = meta_model.predict_proba(meta_input)[:, 1]
print(f'Final Ensemble Prediction: {final_pred}')

# %%
# 将最终预测结果保存到文件中
with open('predictions.txt', 'w') as f:
    for pred in final_pred:
        f.write(f'{pred}\n')