import pickle
import pandas as pd
import os
import glob
from datasets import load_dataset, load_from_disk

# 定义要处理的文件夹列表
folders = ['aime', 'math', 'gpqa']
models = ['Qwen-32B_deepseek-1.5B']
base_path = 'results_cot/greedy_9'

# 存储所有结果的列表
summary_data = []

def check_accuracy(step_str, ground_truth):
    """检查step_str是否包含正确答案"""
    if not step_str or not ground_truth:
        return False
    # 转换为字符串并清理
    step_str_clean = str(step_str).strip().lower()
    ground_truth_clean = str(ground_truth).strip().lower()
    
    # 检查step_str是否包含正确答案
    return ("{" + ground_truth_clean + "}") in step_str_clean or (ground_truth_clean + ")") in step_str_clean or ("answer is " + ground_truth_clean) in step_str_clean

for folder in folders:
    if folder == "aime":
        dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
    elif folder == "math":
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
    elif folder == "gpqa":
        if os.getenv("HF_HUB_OFFLINE", "0") == "1":
            dataset = load_from_disk("/scratch/gpfs/rp2773/hf_cache/datasets/gpqa")
        else:    
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
    for model in models:
        
        folder_step_time_totals = []
        folder_base_time_totals = []
        correct_count = 0
        finished_count = 0
        budget_count = 0
        total_files = 0
        
        for n in range(60, 81):
            folder_path = os.path.join(base_path, folder, model, str(n))
            
            if not os.path.exists(folder_path):
                print(f"目录不存在，跳过: {folder_path}")
                continue

            # 查找文件夹中的所有pickle文件
            pickle_files = (glob.glob(os.path.join(folder_path, '1.pickle')) + 
                glob.glob(os.path.join(folder_path, '0.pickle')))

            if not pickle_files:
                print(f"在 {folder_path} 中没有找到pickle文件")
                continue

            for pickle_file in pickle_files:
                try:
                    # 读取pickle文件
                    with open(pickle_file, 'rb') as f:
                        data = pickle.load(f)

                    # 将数据转换为DataFrame
                    df = pd.DataFrame(data)

                    filename = os.path.basename(pickle_file)
                    print(f"处理文件: {folder}/{filename}")

                    # 计算各列总和
                    base_model_time_total = df['base_model_time'].sum()
                    step_time_total = df['step_time'].sum()

                    print(f"  base_model_time总和: {base_model_time_total}")
                    print(f"  step_time总和: {step_time_total}")

                    # 为每个文件导出单独的CSV文件
                    output_csv = os.path.join(folder_path, f'{os.path.splitext(filename)[0]}_time_data.csv')
                    df[['base_model_time', 'step_time']].to_csv(output_csv, index=False)
                    print(f"  已导出: {output_csv}")

                    # 收集每个文件的数据
                    folder_step_time_totals.append(step_time_total)
                    folder_base_time_totals.append(base_model_time_total)
                    
                    if len(df) > 0:
                        last_row = df.iloc[-1]
                        step_str = last_row.get('step_str', '')
                        step_reason = last_row.get('stop_reason', '')
                        
                        question_index = n
                        if folder == "aime":
                            ground_truth = dataset["answer"][n - 60]
                            options = None
                        elif folder == "math":
                            ground_truth = dataset["answer"][n]
                            options = None
                        elif folder == "gpqa":
                            ground_truth = "A"
                            
                        ground_truth = ground_truth
                        
                        # 检查准确性
                        is_correct = check_accuracy(step_str, ground_truth)
                        if is_correct:
                            correct_count += 1
                        
                        # 统计完成状态
                        if step_reason == 'finished':
                            finished_count += 1
                        elif step_reason == 'budget':
                            budget_count += 1
                        
                        total_files += 1
                        
                        print(f"  准确率检查: 正确答案='{ground_truth}', 预测包含='{step_str[:50]}...', 正确={is_correct}")
                        print(f"  完成状态: {step_reason}")

                except Exception as e:
                    print(f"处理文件 {pickle_file} 时出错: {e}")
                    continue
                
        # 计算文件夹内所有文件的平均值
        if folder_step_time_totals and total_files > 0:
            avg_step_time = sum(folder_step_time_totals) / len(folder_step_time_totals)
            avg_base_time = sum(folder_base_time_totals) / len(folder_base_time_totals)
            total_step_time = sum(folder_step_time_totals)
            total_base_time = sum(folder_base_time_totals)
            
            # 计算准确率和比例
            accuracy = correct_count / total_files if total_files > 0 else 0
            finished_ratio = finished_count / total_files if total_files > 0 else 0
            budget_ratio = budget_count / total_files if total_files > 0 else 0
            
            summary_data.append({
                'dataset': f"{folder}_{model}",
                'file_count': len(folder_step_time_totals),
                'total_step_time': total_step_time,
                'total_base_model_time': total_base_time,
                'avg_step_time': avg_step_time,
                'avg_base_model_time': avg_base_time,
                'accuracy': accuracy,
                'finished_count': finished_count,
                'budget_count': budget_count,
                'finished_ratio': finished_ratio,
                'budget_ratio': budget_ratio,
                'correct_count': correct_count,
                'total_questions': total_files
            })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    
    # 计算总体平均值
    overall_avg_step = summary_df['avg_step_time'].mean()
    overall_avg_base = summary_df['avg_base_model_time'].mean()
    overall_accuracy = summary_df['accuracy'].mean()
    overall_finished_ratio = summary_df['finished_ratio'].mean()
    overall_budget_ratio = summary_df['budget_ratio'].mean()
    
    # 添加总体平均行
    overall_row = pd.DataFrame({
        'dataset': ['OVERALL_AVERAGE'],
        'file_count': [summary_df['file_count'].sum()],
        'total_step_time': [summary_df['total_step_time'].sum()],
        'total_base_model_time': [summary_df['total_base_model_time'].sum()],
        'avg_step_time': [overall_avg_step],
        'avg_base_model_time': [overall_avg_base],
        'accuracy': [overall_accuracy],
        'finished_count': [summary_df['finished_count'].sum()],
        'budget_count': [summary_df['budget_count'].sum()],
        'finished_ratio': [overall_finished_ratio],
        'budget_ratio': [overall_budget_ratio],
        'correct_count': [summary_df['correct_count'].sum()],
        'total_questions': [summary_df['total_questions'].sum()]
    })
    
    summary_df = pd.concat([summary_df, overall_row], ignore_index=True)
    
    # 导出汇总CSV文件
    summary_csv_path = os.path.join(base_path, 'time_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n汇总文件已导出: {summary_csv_path}")
    
    # 打印汇总结果
    print("\n=== 汇总结果 ===")
    print(summary_df.to_string(index=False))
    
    # 打印重要指标
    print("\n=== 关键指标 ===")
    for _, row in summary_df.iterrows():
        if row['dataset'] != 'OVERALL_AVERAGE':
            print(f"{row['dataset']}: 准确率={row['accuracy']:.3f}, 完成率={row['finished_ratio']:.3f}, 预算耗尽率={row['budget_ratio']:.3f}")

else:
    print("没有找到任何可处理的数据")