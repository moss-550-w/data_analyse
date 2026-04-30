"""
消费趋势预测：STL季节性分解 + SARIMA组合模型
功能：基于历史消费数据进行日度消费金额预测
数据来源：intermediate_data_for_review.xlsx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置文件路径（相对路径）
INPUT_FILE = r'..\..\intermediate_data_for_review.xlsx'
OUTPUT_FILE = 'prediction_results.csv'
FORECAST_DAYS = 30  # 预测未来30天


def load_and_preprocess_data(file_path):
    """
    步骤1：加载数据并按日期聚合
    """
    print("=" * 60)
    print("步骤1：数据加载与预处理")
    print("=" * 60)
    
    # 读取数据
    df = pd.read_excel(file_path)
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    
    print(f"✓ 成功加载数据，共 {len(df)} 条交易记录")
    print(f"时间范围: {df['transaction_time'].min().date()} 至 {df['transaction_time'].max().date()}")
    
    # 按日期聚合每日消费金额
    df['date'] = df['transaction_time'].dt.date
    daily_spending = df.groupby('date')['transaction_amount'].sum().reset_index()
    daily_spending['date'] = pd.to_datetime(daily_spending['date'])
    daily_spending = daily_spending.sort_values('date')
    daily_spending.set_index('date', inplace=True)
    
    # 填充缺失日期（如果有）
    date_range = pd.date_range(start=daily_spending.index.min(), 
                               end=daily_spending.index.max(), 
                               freq='D')
    daily_spending = daily_spending.reindex(date_range, fill_value=0)
    daily_spending.index.name = 'date'
    daily_spending.columns = ['spending']
    
    print(f"✓ 日度聚合完成，共 {len(daily_spending)} 天")
    print(f"日均消费: {daily_spending['spending'].mean():.2f}元")
    print(f"日消费范围: {daily_spending['spending'].min():.2f} - {daily_spending['spending'].max():.2f}元")
    
    return daily_spending


def adf_test(timeseries, title):
    """
    ADF平稳性检验
    """
    result = adfuller(timeseries.dropna())
    print(f"\n【{title}】ADF检验结果:")
    print(f"  ADF统计量: {result[0]:.4f}")
    print(f"  p值: {result[1]:.4f}")
    print(f"  临界值(1%): {result[4]['1%']:.4f}")
    print(f"  临界值(5%): {result[4]['5%']:.4f}")
    
    if result[1] <= 0.05:
        print(f"  ✓ 序列平稳 (p <= 0.05)")
        return True
    else:
        print(f"  ✗ 序列非平稳 (p > 0.05)")
        return False


def stl_decomposition(ts_data):
    """
    步骤2：STL季节性分解
    将时间序列分解为趋势项、季节项和残差项
    """
    print("\n" + "=" * 60)
    print("步骤2：STL季节性分解")
    print("=" * 60)
    
    # STL分解参数设置
    # period=7: 周度季节性（7天周期）
    # seasonal=7: 季节平滑窗口
    # trend=15: 趋势平滑窗口
    # robust=True: 鲁棒性加权，抵抗异常值
    
    stl = STL(ts_data['spending'], 
              period=7,
              seasonal=7,
              trend=15,
              robust=True)
    
    result = stl.fit()
    
    # 提取各分量
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid
    
    print(f"✓ STL分解完成")
    print(f"  趋势项均值: {trend.mean():.2f}元")
    print(f"  季节项范围: {seasonal.min():.2f} 至 {seasonal.max():.2f}元")
    print(f"  残差项标准差: {residual.std():.2f}元")
    
    # 保存分解结果
    decomposition_df = pd.DataFrame({
        'original': ts_data['spending'],
        'trend': trend,
        'seasonal': seasonal,
        'residual': residual
    })
    
    return decomposition_df, result


def visualize_stl(decomposition_df, stl_result):
    """
    可视化STL分解结果
    """
    print("\n  生成STL分解可视化...")
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # 原始序列
    axes[0].plot(decomposition_df.index, decomposition_df['original'], 
                 color='steelblue', linewidth=1.5)
    axes[0].set_title('原始消费序列', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('消费金额(元)')
    axes[0].grid(True, alpha=0.3)
    
    # 趋势项
    axes[1].plot(decomposition_df.index, decomposition_df['trend'], 
                 color='red', linewidth=2)
    axes[1].set_title('趋势项 (Trend)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('消费金额(元)')
    axes[1].grid(True, alpha=0.3)
    
    # 季节项
    axes[2].plot(decomposition_df.index, decomposition_df['seasonal'], 
                 color='green', linewidth=1.5)
    axes[2].set_title('季节项 (Seasonal) - 7天周期', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('消费金额(元)')
    axes[2].grid(True, alpha=0.3)
    
    # 残差项
    axes[3].plot(decomposition_df.index, decomposition_df['residual'], 
                 color='purple', linewidth=1, alpha=0.7)
    axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[3].set_title('残差项 (Residual)', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('消费金额(元)')
    axes[3].set_xlabel('日期')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stl_decomposition.png', dpi=300, bbox_inches='tight')
    print("  ✓ STL分解图已保存: stl_decomposition.png")
    
    plt.close()


def train_sarima(component, component_name, seasonal_period=7):
    """
    训练SARIMA模型
    """
    print(f"\n【{component_name}】SARIMA建模:")
    
    # 1. 平稳性检验
    is_stationary = adf_test(component, component_name)
    
    # 2. 确定差分阶数
    d = 0 if is_stationary else 1
    D = 0  # 假设季节性已经通过STL提取
    
    # 3. 尝试不同参数组合，选择AIC最小的模型
    best_aic = np.inf
    best_model = None
    best_order = None
    
    # 简化的参数搜索（实际应用中可以扩大搜索范围）
    p_range = range(0, 3)
    q_range = range(0, 3)
    P_range = range(0, 2)
    Q_range = range(0, 2)
    
    for p in p_range:
        for q in q_range:
            for P in P_range:
                for Q in Q_range:
                    try:
                        model = SARIMAX(component.dropna(),
                                       order=(p, d, q),
                                       seasonal_order=(P, D, Q, seasonal_period),
                                       enforce_stationarity=False,
                                       enforce_invertibility=False)
                        fitted = model.fit(disp=False)
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_model = fitted
                            best_order = (p, d, q, P, D, Q)
                    except:
                        continue
    
    if best_model is not None:
        print(f"  ✓ 最优模型: SARIMA{best_order[:3]}{best_order[3:]}_7")
        print(f"  ✓ AIC: {best_aic:.2f}")
        return best_model
    else:
        # 如果搜索失败，使用默认参数
        print("  ⚠ 参数搜索失败，使用默认参数")
        model = SARIMAX(component.dropna(),
                       order=(1, d, 1),
                       seasonal_order=(1, 0, 1, seasonal_period))
        return model.fit(disp=False)


def white_noise_test(residuals, title):
    """
    残差白噪声检验 (Ljung-Box检验)
    """
    lb_test = acorr_ljungbox(residuals.dropna(), lags=10, return_df=True)
    p_value = lb_test['lb_pvalue'].iloc[-1]
    
    print(f"\n【{title}】Ljung-Box白噪声检验:")
    print(f"  p值: {p_value:.4f}")
    
    if p_value > 0.05:
        print(f"  ✓ 残差为白噪声 (p > 0.05)，无需进一步建模")
        return True
    else:
        print(f"  ✗ 残差存在自相关 (p <= 0.05)，需要建立ARIMA模型")
        return False


def predict_and_reconstruct(decomposition_df, forecast_days=30):
    """
    步骤3：对各分量进行预测并重构结果
    """
    print("\n" + "=" * 60)
    print("步骤3：SARIMA预测与结果重构")
    print("=" * 60)
    
    # 3.1 趋势项预测
    print("\n1. 趋势项预测:")
    trend_model = train_sarima(decomposition_df['trend'], "趋势项")
    trend_forecast = trend_model.get_forecast(steps=forecast_days)
    trend_pred = trend_forecast.predicted_mean
    trend_conf = trend_forecast.conf_int()
    
    # 3.2 季节项预测
    print("\n2. 季节项预测:")
    # 季节项通常较为稳定，可以周期性延续
    seasonal_values = decomposition_df['seasonal'].values
    last_week_seasonal = seasonal_values[-7:]  # 取最后一周的季节项
    
    # 周期性延续
    seasonal_pred = []
    for i in range(forecast_days):
        seasonal_pred.append(last_week_seasonal[i % 7])
    seasonal_pred = pd.Series(seasonal_pred, index=trend_pred.index)
    
    print(f"  ✓ 采用周期性延续法预测季节项")
    print(f"  未来一周季节因子: {last_week_seasonal.round(2)}")
    
    # 3.3 残差项处理
    print("\n3. 残差项处理:")
    residuals = decomposition_df['residual']
    is_white_noise = white_noise_test(residuals, "残差项")
    
    if is_white_noise:
        # 如果是白噪声，直接使用均值（通常为0）
        residual_pred = pd.Series([0] * forecast_days, index=trend_pred.index)
        residual_conf_lower = pd.Series([0] * forecast_days, index=trend_pred.index)
        residual_conf_upper = pd.Series([0] * forecast_days, index=trend_pred.index)
        print(f"  ✓ 残差项取均值0作为预测值")
    else:
        # 如果不是白噪声，建立ARIMA模型
        residual_model = train_sarima(residuals, "残差项")
        residual_forecast = residual_model.get_forecast(steps=forecast_days)
        residual_pred = residual_forecast.predicted_mean
        residual_conf = residual_forecast.conf_int()
        residual_conf_lower = residual_conf.iloc[:, 0]
        residual_conf_upper = residual_conf.iloc[:, 1]
    
    # 3.4 结果重构 (加法模型: Y = Trend + Seasonal + Residual)
    final_prediction = trend_pred + seasonal_pred + residual_pred
    final_lower = trend_conf.iloc[:, 0] + seasonal_pred + residual_conf_lower
    final_upper = trend_conf.iloc[:, 1] + seasonal_pred + residual_conf_upper
    
    # 确保预测值为非负
    final_prediction = final_prediction.clip(lower=0)
    final_lower = final_lower.clip(lower=0)
    final_upper = final_upper.clip(lower=0)
    
    print("\n4. 预测结果汇总:")
    print(f"  预测天数: {forecast_days}天")
    print(f"  预测总金额: {final_prediction.sum():.2f}元")
    print(f"  预测日均消费: {final_prediction.mean():.2f}元")
    print(f"  95%置信区间: [{final_lower.sum():.2f}元, {final_upper.sum():.2f}元]")
    
    # 构建预测结果DataFrame
    forecast_df = pd.DataFrame({
        'date': trend_pred.index,
        'trend': trend_pred.values,
        'seasonal': seasonal_pred.values,
        'residual': residual_pred.values,
        'prediction': final_prediction.values,
        'lower_bound': final_lower.values,
        'upper_bound': final_upper.values
    })
    
    return forecast_df


def visualize_prediction(decomposition_df, forecast_df):
    """
    步骤4：可视化预测结果
    """
    print("\n" + "=" * 60)
    print("步骤4：生成预测可视化")
    print("=" * 60)
    
    # 合并历史数据和预测数据
    historical = decomposition_df['original']
    prediction_dates = forecast_df['date']
    prediction_values = forecast_df['prediction']
    lower_bound = forecast_df['lower_bound']
    upper_bound = forecast_df['upper_bound']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 图1：整体预测图
    ax1 = axes[0]
    ax1.plot(historical.index, historical.values, 
             color='steelblue', linewidth=1.5, label='历史数据')
    ax1.plot(prediction_dates, prediction_values, 
             color='red', linewidth=2, linestyle='--', label='预测值')
    ax1.fill_between(prediction_dates, lower_bound, upper_bound, 
                      color='red', alpha=0.2, label='95%置信区间')
    ax1.axvline(x=historical.index[-1], color='gray', 
                linestyle=':', alpha=0.7, label='预测起点')
    
    ax1.set_title('消费趋势预测 (STL+SARIMA组合模型)', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('消费金额(元)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 图2：预测期详细图
    ax2 = axes[1]
    ax2.plot(prediction_dates, prediction_values, 
             color='red', linewidth=2, marker='o', markersize=4, label='预测值')
    ax2.fill_between(prediction_dates, lower_bound, upper_bound, 
                      color='red', alpha=0.2, label='95%置信区间')
    
    # 标注周末
    for i, date in enumerate(prediction_dates):
        if date.weekday() >= 5:  # 周六或周日
            ax2.axvspan(date, date + pd.Timedelta(days=1), 
                       alpha=0.1, color='green')
    
    ax2.set_title('未来30天消费预测详情 (绿色背景为周末)', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel('消费金额(元)')
    ax2.set_xlabel('日期')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 旋转x轴日期标签
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('consumption_forecast.png', dpi=300, bbox_inches='tight')
    print("✓ 预测结果图已保存: consumption_forecast.png")
    
    plt.close()


def evaluate_model(decomposition_df):
    """
    步骤5：模型评估
    """
    print("\n" + "=" * 60)
    print("步骤5：模型评估")
    print("=" * 60)
    
    # 使用最后7天作为测试集
    train_data = decomposition_df['original'][:-7]
    test_data = decomposition_df['original'][-7:]
    
    # 简单基准模型：用历史均值预测
    baseline_pred = [train_data.mean()] * 7
    
    # 计算评估指标
    mae_baseline = mean_absolute_error(test_data, baseline_pred)
    rmse_baseline = np.sqrt(mean_squared_error(test_data, baseline_pred))
    mape_baseline = np.mean(np.abs((test_data - baseline_pred) / test_data)) * 100
    
    print("\n基准模型（历史均值）评估:")
    print(f"  MAE: {mae_baseline:.2f}元")
    print(f"  RMSE: {rmse_baseline:.2f}元")
    print(f"  MAPE: {mape_baseline:.2f}%")
    
    print("\n说明:")
    print("  - MAE: 平均绝对误差，预测值与真实值的平均差距")
    print("  - RMSE: 均方根误差，对大额误差更敏感")
    print("  - MAPE: 平均绝对百分比误差，相对误差指标")
    
    return {
        'mae': mae_baseline,
        'rmse': rmse_baseline,
        'mape': mape_baseline
    }


def save_results(forecast_df, decomposition_df):
    """
    步骤6：保存预测结果
    """
    print("\n" + "=" * 60)
    print("步骤6：保存预测结果")
    print("=" * 60)
    
    # 保存预测结果
    forecast_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"✓ 预测结果已保存: {OUTPUT_FILE}")
    
    # 生成预测报告
    report = []
    report.append("=" * 60)
    report.append("消费趋势预测报告 (STL+SARIMA组合模型)")
    report.append("=" * 60)
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    report.append("\n" + "-" * 60)
    report.append("历史数据统计")
    report.append("-" * 60)
    report.append(f"数据天数: {len(decomposition_df)}天")
    report.append(f"历史总消费: {decomposition_df['original'].sum():.2f}元")
    report.append(f"日均消费: {decomposition_df['original'].mean():.2f}元")
    report.append(f"消费标准差: {decomposition_df['original'].std():.2f}元")
    
    report.append("\n" + "-" * 60)
    report.append("预测结果汇总")
    report.append("-" * 60)
    report.append(f"预测天数: {len(forecast_df)}天")
    report.append(f"预测日期范围: {forecast_df['date'].min().date()} 至 {forecast_df['date'].max().date()}")
    report.append(f"预测总消费: {forecast_df['prediction'].sum():.2f}元")
    report.append(f"预测日均消费: {forecast_df['prediction'].mean():.2f}元")
    report.append(f"95%置信区间: [{forecast_df['lower_bound'].sum():.2f}元, {forecast_df['upper_bound'].sum():.2f}元]")
    
    report.append("\n" + "-" * 60)
    report.append("每日预测详情")
    report.append("-" * 60)
    
    for _, row in forecast_df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        weekday = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'][row['date'].weekday()]
        report.append(f"{date_str} ({weekday}): {row['prediction']:.2f}元 "
                     f"[置信区间: {row['lower_bound']:.2f}-{row['upper_bound']:.2f}元]")
    
    report.append("\n" + "=" * 60)
    
    report_text = '\n'.join(report)
    
    with open('forecast_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ 预测报告已保存: forecast_report.txt")
    print("\n输出文件列表:")
    print(f"  1. {OUTPUT_FILE} - 预测结果数据")
    print(f"  2. forecast_report.txt - 文字预测报告")
    print(f"  3. stl_decomposition.png - STL分解可视化")
    print(f"  4. consumption_forecast.png - 消费趋势预测图")


def main():
    """
    主函数：执行完整的消费趋势预测流程
    """
    print("\n" + "=" * 60)
    print("消费趋势预测工具")
    print("=" * 60)
    print("模型: STL季节性分解 + SARIMA组合模型")
    print("=" * 60)
    
    try:
        # 步骤1：数据加载与预处理
        daily_data = load_and_preprocess_data(INPUT_FILE)
        
        # 步骤2：STL季节性分解
        decomposition_df, stl_result = stl_decomposition(daily_data)
        visualize_stl(decomposition_df, stl_result)
        
        # 步骤3：SARIMA预测与结果重构
        forecast_df = predict_and_reconstruct(decomposition_df, FORECAST_DAYS)
        
        # 步骤4：可视化预测结果
        visualize_prediction(decomposition_df, forecast_df)
        
        # 步骤5：模型评估
        metrics = evaluate_model(decomposition_df)
        
        # 步骤6：保存结果
        save_results(forecast_df, decomposition_df)
        
        print("\n" + "=" * 60)
        print("预测分析完成！")
        print("=" * 60)
        
        return forecast_df, decomposition_df
        
    except Exception as e:
        print(f"\n✗ 预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == '__main__':
    forecast_df, decomposition_df = main()
