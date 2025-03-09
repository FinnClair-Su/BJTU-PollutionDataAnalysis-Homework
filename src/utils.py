import os
import matplotlib.pyplot as plt


def setup_environment():
    """设置项目环境"""
    # 创建必要的目录
    directories = ['data', 'plots', 'output', 'plots/model_comparison']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # 设置matplotlib参数
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    print("环境设置完成")


def save_model_comparison(model_results, target_column):
    """保存模型比较结果"""
    # 确保输出目录存在
    os.makedirs('output', exist_ok=True)

    # 保存CSV格式
    model_results.to_csv(f'output/{target_column}_model_comparison.csv', index=False)

    # 保存为易读的文本格式
    with open(f'output/{target_column}_model_comparison.txt', 'w') as f:
        f.write(f"# {target_column} 预测模型性能对比\n\n")
        f.write(model_results.to_string(index=False))

    print(f"{target_column} 模型比较结果已保存")
