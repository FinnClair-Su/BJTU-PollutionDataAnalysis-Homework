import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import traceback

# 添加src目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.data_loader import load_data, preprocess_data, export_data
from src.data_visualizer import visualize_data, correlation_analysis, compare_pearson_correlation
from src.feature_engineering import normalize_data
from src.model_trainer import train_models
from src.utils import setup_environment, save_model_comparison


def main():
    try:
        # 设置环境
        setup_environment()

        # 1. 加载数据
        data = load_data('data/data.csv')

        # 2. 数据预处理
        processed_data, numeric_columns, original_data = preprocess_data(data)

        # 检查是否还有缺失值
        missing_values = processed_data[numeric_columns].isnull().sum()
        if missing_values.sum() > 0:
            print(f"警告：处理后数据仍有缺失值:\n{missing_values[missing_values > 0]}")
            print("尝试再次填充缺失值...")
            for col in numeric_columns:
                if processed_data[col].isnull().any():
                    processed_data[col] = processed_data[col].fillna(processed_data[col].median())

        # 3. 数据可视化
        visualize_data(processed_data, numeric_columns)

        # 4. 相关性分析以及数据清洗前后的对比
        correlation_matrix = correlation_analysis(processed_data, numeric_columns)
        try:
            correlation_comparison = compare_pearson_correlation(original_data, processed_data, numeric_columns)
        except Exception as e:
            print(f"计算Pearson相关系数对比时出错: {str(e)}")
            print("继续执行其他分析...")

        # 5. 数据归一化
        data_minmax, data_standard = normalize_data(processed_data, numeric_columns)

        # 6. 多模型构建与比较 (针对PM2.5)
        try:
            model_results = train_models(processed_data, 'PM25')
            save_model_comparison(model_results, 'PM25')
        except Exception as e:
            print(f"模型训练过程中出错: {str(e)}")
            traceback.print_exc()
            print("跳过模型训练步骤...")

        # 7. 导出处理后的数据
        export_data(processed_data, 'processed_data.csv')
        export_data(data_minmax, 'normalized_minmax_data.csv')
        export_data(data_standard, 'normalized_standard_data.csv')

        print("\n分析完成！所有图表和结果已保存。")

    except Exception as e:
        print(f"程序执行过程中出现错误: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
