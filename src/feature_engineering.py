import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize_data(data, numeric_columns):
    """使用两种归一化方法处理数据"""
    print("开始数据归一化...")

    # 1. Min-Max归一化
    min_max_scaler = MinMaxScaler()
    data_minmax = pd.DataFrame(
        min_max_scaler.fit_transform(data[numeric_columns]),
        columns=numeric_columns
    )

    # 2. Z-score标准化 (Standard Scaler)
    standard_scaler = StandardScaler()
    data_standard = pd.DataFrame(
        standard_scaler.fit_transform(data[numeric_columns]),
        columns=numeric_columns
    )

    # 可视化归一化效果
    plt.figure(figsize=(15, 10))
    # 原始数据
    plt.subplot(3, 1, 1)
    sns.boxplot(data=data[numeric_columns[:6]])
    plt.title('原始数据')
    plt.xticks(rotation=45)

    # Min-Max归一化后
    plt.subplot(3, 1, 2)
    sns.boxplot(data=data_minmax[numeric_columns[:6]])
    plt.title('Min-Max归一化后')
    plt.xticks(rotation=45)

    # Z-score标准化后
    plt.subplot(3, 1, 3)
    sns.boxplot(data=data_standard[numeric_columns[:6]])
    plt.title('Z-score标准化后')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('plots/normalization_comparison.png')
    plt.close()

    return data_minmax, data_standard
