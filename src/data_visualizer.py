import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def visualize_data(data, numeric_columns):
    """不同维度的数据可视化"""
    print("开始数据可视化...")

    # 创建保存图表的目录
    os.makedirs('plots', exist_ok=True)

    # 1. 时间序列图 - 展示主要污染物随时间变化
    plt.figure(figsize=(15, 10))
    for i, pollutant in enumerate(['PM25', 'PM10', 'CO', 'NO2', 'SO2', 'O3']):
        if pollutant in data.columns:
            plt.subplot(3, 2, i + 1)
            plt.plot(data['RECEIVETIME'], data[pollutant])
            plt.title(f'{pollutant}随时间变化')
            plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/time_series.png')
    plt.close()

    # 2. 密度分布图 - 查看数据分布情况
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(numeric_columns[:6]):
        plt.subplot(2, 3, i + 1)
        sns.histplot(data[column], kde=True)
        plt.title(f'{column}分布')
    plt.tight_layout()
    plt.savefig('plots/distributions.png')
    plt.close()

    # 3. 箱型图 - 查看数据的分布和异常值
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=data[numeric_columns[:6]])
    plt.title('主要污染物箱型图')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/boxplots.png')
    plt.close()

    # 4. 热力图 - 按小时和日期展示PM2.5浓度
    if 'PM25' in data.columns and 'Hour' in data.columns and 'Day' in data.columns:
        pivot_data = data.pivot_table(index='Hour', columns='Day', values='PM25', aggfunc='mean')
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, cmap='YlOrRd', annot=False)
        plt.title('PM2.5浓度热力图 (小时 vs 日期)')
        plt.tight_layout()
        plt.savefig('plots/pm25_heatmap.png')
        plt.close()

    # 5. 多变量关系图
    sns.pairplot(data[numeric_columns[:5]])
    plt.savefig('plots/pairplot.png')
    plt.close()

    print("数据可视化完成并保存到plots文件夹")


def correlation_analysis(data, numeric_columns):
    """分析数值列之间的相关性"""
    print("进行相关性分析...")

    # 计算相关系数矩阵
    corr_matrix = data[numeric_columns].corr()

    # 绘制相关性热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('属性间相关性热力图')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()

    # 找出相关性高的特征对 (|r|>0.7)
    high_corr = []
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr.append((numeric_columns[i], numeric_columns[j], corr_matrix.iloc[i, j]))

    print("\n强相关性特征对 (|r|>0.7):")
    for item in high_corr:
        print(f"{item[0]} 和 {item[1]}: {item[2]:.4f}")

    # 可视化几个高相关性特征对
    if high_corr:
        plt.figure(figsize=(15, 10))
        for i, (feat1, feat2, _) in enumerate(high_corr[:6]):  # 最多展示6个
            if i >= 6:
                break
            plt.subplot(2, 3, i + 1)
            plt.scatter(data[feat1], data[feat2], alpha=0.5)
            plt.title(f'{feat1} vs {feat2}')
            plt.xlabel(feat1)
            plt.ylabel(feat2)
        plt.tight_layout()
        plt.savefig('plots/high_correlation_pairs.png')
        plt.close()

    return corr_matrix


def compare_pearson_correlation(original_data, cleaned_data, numeric_columns):
    """比较数据清洗前后的Pearson相关系数"""
    print("比较数据清洗前后的Pearson相关系数...")

    # 确保原始数据中存在所有所需的列
    orig_columns = [col for col in numeric_columns if col in original_data.columns]

    # 创建一个DataFrame用于存储相关系数的对比
    comparison_data = []

    for i in range(len(orig_columns)):
        for j in range(i + 1, len(orig_columns)):
            col1, col2 = orig_columns[i], orig_columns[j]

            try:
                # 计算原始数据的相关系数 - 使用共同的非缺失值
                # 首先找出两列都非缺失的索引
                common_indices_orig = original_data[[col1, col2]].dropna().index
                if len(common_indices_orig) > 1:  # 需要至少两个点才能计算相关系数
                    orig_corr, orig_p = pearsonr(
                        original_data.loc[common_indices_orig, col1],
                        original_data.loc[common_indices_orig, col2]
                    )
                else:
                    # 如果没有足够的共同非缺失值，设置为NaN
                    orig_corr, orig_p = np.nan, np.nan

                # 计算清洗后数据的相关系数
                # 清洗后数据应该没有缺失值，但为了保险起见仍然使用dropna
                common_indices_cleaned = cleaned_data[[col1, col2]].dropna().index
                if len(common_indices_cleaned) > 1:
                    clean_corr, clean_p = pearsonr(
                        cleaned_data.loc[common_indices_cleaned, col1],
                        cleaned_data.loc[common_indices_cleaned, col2]
                    )
                else:
                    clean_corr, clean_p = np.nan, np.nan

                # 添加到比较数据中
                comparison_data.append({
                    'Feature1': col1,
                    'Feature2': col2,
                    'Original_Correlation': orig_corr,
                    'Cleaned_Correlation': clean_corr,
                    'Difference': clean_corr - orig_corr if not np.isnan(orig_corr) and not np.isnan(
                        clean_corr) else np.nan,
                    'Original_p_value': orig_p,
                    'Cleaned_p_value': clean_p
                })
            except Exception as e:
                print(f"计算 {col1} 和 {col2} 的相关系数时出错: {str(e)}")
                # 添加错误记录到数据中
                comparison_data.append({
                    'Feature1': col1,
                    'Feature2': col2,
                    'Original_Correlation': np.nan,
                    'Cleaned_Correlation': np.nan,
                    'Difference': np.nan,
                    'Original_p_value': np.nan,
                    'Cleaned_p_value': np.nan
                })

    # 转换为DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    # 过滤掉包含NaN的行
    comparison_df = comparison_df.dropna()

    if len(comparison_df) > 0:
        # 按相关系数差异绝对值排序
        comparison_df['Abs_Difference'] = comparison_df['Difference'].abs()
        comparison_df = comparison_df.sort_values('Abs_Difference', ascending=False)
        comparison_df = comparison_df.drop('Abs_Difference', axis=1)

        # 创建一个更美观的表格展示 - 使用前15个差异最大的或全部（如果不足15个）
        top_diff = comparison_df.head(min(15, len(comparison_df)))

        # 可视化对比表格
        plt.figure(figsize=(14, 10))
        plt.axis('off')

        # 创建表格
        table_data = []
        headers = ['特征对', '清洗前相关系数', '清洗后相关系数', '差异']

        for _, row in top_diff.iterrows():
            feature_pair = f"{row['Feature1']} - {row['Feature2']}"
            orig_corr = f"{row['Original_Correlation']:.4f}"
            clean_corr = f"{row['Cleaned_Correlation']:.4f}"
            diff = f"{row['Difference']:.4f}"
            table_data.append([feature_pair, orig_corr, clean_corr, diff])

        # 绘制表格
        table = plt.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center',
            colWidths=[0.3, 0.2, 0.2, 0.2]
        )

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)  # 调整表格大小

        plt.title('数据清洗前后Pearson相关系数对比（差异最大的特征对）', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('plots/pearson_correlation_comparison.png', bbox_inches='tight')
        plt.close()

        # 导出完整比较数据
        os.makedirs('output', exist_ok=True)
        comparison_df.to_csv('output/correlation_comparison.csv', index=False)

        print("相关系数对比完成并保存")
    else:
        print("无法计算相关系数对比：数据不足或全为NaN")

    return comparison_df

