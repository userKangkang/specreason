import pickle
import pandas as pd

# 读取pickle文件
with open('results/60/2.pickle', 'rb') as f:
    data = pickle.load(f)

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 提取需要的列并计算总和
# small_model_time_total = df['small_model_time'].sum()
base_model_time_total = df['base_model_time'].sum()
# eval_time_total = df['eval_time'].sum()
step_time_total = df['step_time'].sum()

# print(f"small_model_time总和: {small_model_time_total}")
print(f"base_model_time总和: {base_model_time_total}")
# print(f"eval_time总和: {eval_time_total}")
print(f"step_time总和: {step_time_total}")

# 如果只想保存特定的列
df[[
    # 'small_model_time',
    'base_model_time',
    # 'eval_time',
    'step_time'
    ]].to_csv('results/60/2_time_data.csv', index=False)


### specreason(0): 
### small_model_time总和: 12.278065960997083
### base_model_time总和: 17.509580694999386
### eval_time总和: 8.367939907999244
### step_time总和: 38.15558656399571
###
### bigmodel(1):
### base_model_time总和: 86.00351376500475
### step_time总和: 86.00351376500475
### 
### smallmodel(2):
### base_model_time总和: 7.12147454800288
### step_time总和: 7.12147454800288