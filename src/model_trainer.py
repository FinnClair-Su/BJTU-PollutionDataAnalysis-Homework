import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def prepare_data(data, target_column='PM25'):
    """准备模型训练和测试数据集"""
    print(f"准备{target_column}预测模型的数据...")

    # 准备特征和目标变量
    feature_columns = [col for col in data.columns if col != target_column
                       and col != 'RECEIVETIME' and pd.api.types.is_numeric_dtype(data[col])]
    X = data[feature_columns]
    y = data[target_column]

    # 检查并处理缺失值
    if X.isnull().any().any() or y.isnull().any():
        print("警告：数据中仍然存在缺失值，正在进行额外的缺失值处理...")

        # 处理特征缺失值
        for col in X.columns:
            if X[col].isnull().any():
                print(f"列 {col} 中有 {X[col].isnull().sum()} 个缺失值")
                X[col] = X[col].fillna(X[col].median())

        # 处理目标变量缺失值
        if y.isnull().any():
            print(f"目标变量 {target_column} 中有 {y.isnull().sum()} 个缺失值")
            y = y.fillna(y.median())

    # 再次检查缺失值
    if X.isnull().any().any() or y.isnull().any():
        print("错误：处理后数据仍然存在缺失值，将移除包含缺失值的行...")
        # 确定非缺失值的索引
        valid_indices = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test, feature_columns


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """训练模型并评估性能"""
    print(f"训练和评估 {name} 模型...")

    # 确保没有缺失值
    if X_train.isnull().any().any() or y_train.isnull().any():
        raise ValueError(f"训练数据中存在缺失值，请先处理缺失值")

    if X_test.isnull().any().any() or y_test.isnull().any():
        raise ValueError(f"测试数据中存在缺失值，请先处理缺失值")

    # 计时开始
    start_time = time.time()

    # 训练模型
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"训练 {name} 模型时出错: {str(e)}")
        # 返回空结果
        return {
            'Model': name,
            'MSE': np.nan,
            'RMSE': np.nan,
            'MAE': np.nan,
            'R^2': np.nan,
            'CV R^2': np.nan,
            'Training Time': np.nan
        }, None, None

    # 计时结束
    train_time = time.time() - start_time

    # 预测
    y_pred = model.predict(X_test)

    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 交叉验证
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_r2 = np.mean(cv_scores)
    except Exception as e:
        print(f"执行交叉验证时出错: {str(e)}")
        cv_r2 = np.nan

    result = {
        'Model': name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R^2': r2,
        'CV R^2': cv_r2,
        'Training Time': train_time
    }

    return result, y_pred, model


def train_deep_learning_model(X_train, X_test, y_train, y_test):
    """训练深度学习模型作为SOTA模型"""
    print("训练深度学习模型 (SOTA)...")

    # 创建目录
    os.makedirs('plots/model_comparison', exist_ok=True)

    # 计时开始
    start_time = time.time()

    # 构建模型
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # 编译模型
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # 早停策略
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    # 计时结束
    train_time = time.time() - start_time

    # 预测
    y_pred = model.predict(X_test).flatten()

    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 可视化训练历史
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('模型损失')
    plt.ylabel('损失')
    plt.xlabel('Epoch')
    plt.legend(['训练集', '验证集'], loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('平均绝对误差')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['训练集', '验证集'], loc='upper right')

    plt.tight_layout()
    plt.savefig('plots/model_comparison/deep_learning_history.png')
    plt.close()

    result = {
        'Model': 'Deep Learning (SOTA)',
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R^2': r2,
        'CV R^2': None,  # 深度学习模型没有使用CV
        'Training Time': train_time
    }

    return result, y_pred, model


def plot_feature_importance(model, feature_columns, model_name, target_column):
    """绘制特征重要性图"""
    if hasattr(model, 'feature_importances_'):
        # 创建目录
        os.makedirs('plots/model_comparison', exist_ok=True)

        # 获取特征重要性
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        # 绘制特征重要性图
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance[:10])  # 展示前10个重要特征
        plt.title(f'{model_name} - 预测{target_column}的特征重要性')
        plt.tight_layout()
        plt.savefig(f'plots/model_comparison/{target_column}_{model_name}_feature_importance.png')
        plt.close()

        return feature_importance
    return None


def plot_predictions(y_test, y_pred, model_name, target_column):
    """绘制预测值与实际值对比图"""
    # 创建目录
    os.makedirs('plots/model_comparison', exist_ok=True)

    # 绘制预测对比图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'{model_name} - {target_column}预测值与实际值对比')
    plt.tight_layout()
    plt.savefig(f'plots/model_comparison/{target_column}_{model_name}_predictions.png')
    plt.close()


def train_models(data, target_column='PM25'):
    """训练和对比不同的模型"""
    print(f"\n开始为{target_column}构建和评估多种模型...")

    # 准备数据
    X_train, X_test, y_train, y_test, feature_columns = prepare_data(data, target_column)

    # 定义模型
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GBDT': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'MLP (Neural Network)': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=100, random_state=42)
    }

    # 评估模型
    results = []
    trained_models = {}

    for name, model in models.items():
        # 训练和评估模型
        result, y_pred, trained_model = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(result)
        trained_models[name] = trained_model

        # 绘制预测对比图
        plot_predictions(y_test, y_pred, name, target_column)

        # 绘制特征重要性（如果模型支持）
        plot_feature_importance(trained_model, feature_columns, name, target_column)

    # 添加深度学习模型（SOTA）
    try:
        dl_result, dl_pred, dl_model = train_deep_learning_model(X_train, X_test, y_train, y_test)
        results.append(dl_result)
        trained_models['Deep Learning (SOTA)'] = dl_model

        # 绘制预测对比图
        plot_predictions(y_test, dl_pred, 'Deep Learning (SOTA)', target_column)
    except Exception as e:
        print(f"深度学习模型训练失败: {str(e)}")

    # 创建结果DataFrame
    results_df = pd.DataFrame(results)

    # 可视化模型性能对比
    metrics = ['MSE', 'RMSE', 'MAE', 'R^2', 'Training Time']

    plt.figure(figsize=(15, 12))
    for i, metric in enumerate(metrics):
        plt.subplot(3, 2, i + 1)

        # 暂时去掉可能含有None的CV R^2
        temp_df = results_df.dropna(subset=[metric])

        # 根据指标确定绘图方向（越小越好或越大越好）
        if metric in ['MSE', 'RMSE', 'MAE', 'Training Time']:
            # 这些指标越低越好
            sns.barplot(x='Model', y=metric, data=temp_df, palette='Blues_r')
        else:
            # 这些指标越高越好
            sns.barplot(x='Model', y=metric, data=temp_df, palette='Blues')

        plt.title(f'模型 {metric} 比较')
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f'plots/model_comparison/{target_column}_model_metrics_comparison.png')
    plt.close()

    # 输出最佳模型
    best_r2_idx = results_df['R^2'].idxmax()
    best_model_name = results_df.loc[best_r2_idx]['Model']
    best_r2 = results_df.loc[best_r2_idx]['R^2']

    print(f"\n最佳模型: {best_model_name}, R^2: {best_r2:.4f}")

    return results_df
