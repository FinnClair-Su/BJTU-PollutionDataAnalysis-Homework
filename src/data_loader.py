import pandas as pd
import numpy as np
import os


def load_data(file_path):
    """加载数据"""
    print("加载数据...")
    data = pd.read_csv(file_path)
    print(f"加载完成, 共 {data.shape[0]} 行, {data.shape[1]} 列")
    return data


def preprocess_data(data):
    """数据预处理"""
    print("开始数据预处理...")

    # 保存原始数据的副本用于后续比较
    original_data = data.copy()

    # 删除站点编码列，因为都是同一个站点
    if 'STATIONCODE' in data.columns:
        data = data.drop('STATIONCODE', axis=1)

    # 处理时间列
    if 'RECEIVETIME' in data.columns:
        data['RECEIVETIME'] = pd.to_datetime(data['RECEIVETIME'])
        data['Hour'] = data['RECEIVETIME'].dt.hour
        data['Day'] = data['RECEIVETIME'].dt.day
        data['Month'] = data['RECEIVETIME'].dt.month

    # 检查并处理缺失值
    missing_values = data.isnull().sum()
    print(f"缺失值统计:\n{missing_values[missing_values > 0]}")

    # 选择我们需要的数值型列进行分析
    numeric_columns = ['CO', 'NO2', 'SO2', 'O3', 'PM25', 'PM10', 'TEMPERATURE',
                       'HUMIDITY', 'PM05N', 'PM1N', 'PM25N', 'PM10N',
                       'WindSpeed', 'heating season']

    # 确保所有选择的列都存在
    numeric_columns = [col for col in numeric_columns if col in data.columns]

    # 清洗异常值 (使用3个标准差法) - 修复链式赋值问题
    for column in numeric_columns:
        # 计算均值和标准差
        mean_val = data[column].mean()
        std_val = data[column].std()

        # 创建掩码标识异常值
        upper_mask = data[column] > mean_val + 3 * std_val
        lower_mask = data[column] < mean_val - 3 * std_val

        # 复制数据列并替换异常值为NaN
        temp_series = data[column].copy()
        temp_series[upper_mask | lower_mask] = np.nan

        # 将修改后的列赋值回原数据集
        data[column] = temp_series

    # 填充缺失值 - 修复链式赋值问题
    for column in numeric_columns:
        data[column] = data[column].fillna(data[column].median())

    print("数据预处理完成")
    return data, numeric_columns, original_data


def export_data(data, filename):
    """导出数据到CSV文件"""
    # 确保输出目录存在
    os.makedirs('output', exist_ok=True)

    filepath = os.path.join('output', filename)
    data.to_csv(filepath, index=False)
    print(f"数据已导出到 {filepath}")
