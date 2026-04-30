"""
消费影响因素分析：随机森林特征重要性+双重差分法组合模型
功能：识别影响消费金额的关键因素，并估计特定事件的因果效应
数据来源：intermediate_data_for_review.xlsx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置文件路径（相对路径）
INPUT_FILE = r'..\..\intermediate_data_for_review.xlsx'

# 随机森林参数
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 8,
    'min_samples_split': 5,
    'min_samples_leaf': 3,
    'random_state': 42
}


def load_and_prepare_data(file_path):
    """
    步骤1：加载数据并构建特征
    """
    print("=" * 70)
    print("步骤1：数据加载与特征工程")
    print("=" * 70)
    
    # 读取数据
    df = pd.read_excel(file_path)
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    
    print(f"✓ 成功加载数据，共 {len(df)} 条交易记录")
    print(f"时间范围: {df['transaction_time'].min().date()} 至 {df['transaction_time'].max().date()}")
    
    # 构建特征
    df = build_features(df)
    
    print(f"\n✓ 特征工程完成，共构建 {len([c for c in df.columns if c not in ['transaction_id', 'transaction_time', '人工审核结果', '备注']])} 个特征")
    
    return df


def build_features(df):
    """
    构建影响消费金额的特征
    """
    # 时间特征
    df['month'] = df['transaction_time'].dt.month
    df['weekday'] = df['transaction_time'].dt.weekday  # 0=周一, 6=周日
    df['day'] = df['transaction_time'].dt.day
    df['hour'] = df['transaction_time'].dt.hour
    
    # 时段特征 (1=早:6-11, 2=中:12-17, 3=晚:18-22, 4=深夜:23-5)
    df['time_period'] = df['hour'].apply(
        lambda h: 1 if 6 <= h < 12 else (2 if 12 <= h < 18 else (3 if 18 <= h < 23 else 4))
    )
    
    # 是否周末
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    # 是否月初/月末 (假设每月15日为发薪日，前后3天为发薪日前后)
    df['is_payday_near'] = ((df['day'] >= 12) & (df['day'] <= 18)).astype(int)
    
    # 是否月初 (1-7日)
    df['is_month_start'] = (df['day'] <= 7).astype(int)
    
    # 是否月末 (25-31日)
    df['is_month_end'] = (df['day'] >= 25).astype(int)
    
    # 消费类目编码
    le_category = LabelEncoder()
    df['category_encoded'] = le_category.fit_transform(df['category'])
    
    # 是否必要消费（超市购物、餐饮美食为必要消费）
    necessary_categories = ['超市购物', '餐饮美食', '交通出行']
    df['is_necessary'] = df['category'].isin(necessary_categories).astype(int)
    
    # 是否高频小额（超市购物）
    df['is_small_freq'] = (df['category'] == '超市购物').astype(int)
    
    # 是否低频大额（其他、服饰美容）
    df['is_large_infreq'] = df['category'].isin(['其他', '服饰美容']).astype(int)
    
    # 前N天平均消费（历史特征）
    df = df.sort_values('transaction_time').reset_index(drop=True)
    df['avg_3d'] = 0.0
    df['avg_7d'] = 0.0
    
    for i in range(len(df)):
        current_time = df.loc[i, 'transaction_time']
        
        # 前3天平均
        past_3d = df[(df['transaction_time'] < current_time) & 
                     (df['transaction_time'] >= current_time - pd.Timedelta(days=3))]
        df.loc[i, 'avg_3d'] = past_3d['transaction_amount'].mean() if len(past_3d) > 0 else 0
        
        # 前7天平均
        past_7d = df[(df['transaction_time'] < current_time) & 
                     (df['transaction_time'] >= current_time - pd.Timedelta(days=7))]
        df.loc[i, 'avg_7d'] = past_7d['transaction_amount'].mean() if len(past_7d) > 0 else 0
    
    # 是否为异常值
    df['is_outlier'] = df['is_outlier_iqr'] | df['is_outlier_3sigma']
    df['is_outlier'] = df['is_outlier'].astype(int)
    
    return df


def random_forest_feature_importance(df):
    """
    步骤2：随机森林特征重要性分析
    """
    print("\n" + "=" * 70)
    print("步骤2：随机森林特征重要性分析")
    print("=" * 70)
    
    # 选择特征列
    feature_cols = [
        'month', 'weekday', 'day', 'hour', 'time_period',
        'is_weekend', 'is_payday_near', 'is_month_start', 'is_month_end',
        'category_encoded', 'is_necessary', 'is_small_freq', 'is_large_infreq',
        'avg_3d', 'avg_7d', 'is_outlier'
    ]
    
    X = df[feature_cols]
    y = df['transaction_amount']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练随机森林模型
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)
    
    # 模型评估
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n模型性能:")
    print(f"  训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}")
    print(f"  MSE: {mse:.2f}")
    print(f"  R²: {r2:.3f}")
    
    # 计算置换重要性
    perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    
    # 整理特征重要性结果
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std,
        'gini_importance': rf.feature_importances_
    })
    
    importance_df = importance_df.sort_values('importance_mean', ascending=False)
    importance_df['importance_pct'] = importance_df['importance_mean'] / importance_df['importance_mean'].sum() * 100
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    # 映射特征名称为中文
    feature_name_map = {
        'month': '月份',
        'weekday': '星期',
        'day': '日期',
        'hour': '小时',
        'time_period': '时段',
        'is_weekend': '是否周末',
        'is_payday_near': '是否发薪日前后',
        'is_month_start': '是否月初',
        'is_month_end': '是否月末',
        'category_encoded': '消费类目',
        'is_necessary': '是否必要消费',
        'is_small_freq': '是否高频小额',
        'is_large_infreq': '是否低频大额',
        'avg_3d': '前3天平均消费',
        'avg_7d': '前7天平均消费',
        'is_outlier': '是否异常消费'
    }
    importance_df['feature_cn'] = importance_df['feature'].map(feature_name_map)
    
    print(f"\n特征重要性排序 (置换重要性):")
    print(f"{'排名':<6}{'特征名称':<16}{'重要性':<12}{'贡献度(%)':<12}")
    print("-" * 50)
    for _, row in importance_df.head(10).iterrows():
        print(f"{row['rank']:<6}{row['feature_cn']:<16}{row['importance_mean']:<12.4f}{row['importance_pct']:<12.1f}")
    
    return rf, importance_df, X, y


def visualize_feature_importance(importance_df):
    """
    可视化特征重要性
    """
    print("\n  生成特征重要性可视化...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 图1：置换重要性条形图
    top_features = importance_df.head(10)
    bars1 = axes[0].barh(range(len(top_features)), top_features['importance_mean'], 
                         xerr=top_features['importance_std'], capsize=3,
                         color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['feature_cn'])
    axes[0].set_xlabel('置换重要性', fontsize=12)
    axes[0].set_ylabel('特征', fontsize=12)
    axes[0].set_title('随机森林置换重要性 (TOP10)', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # 图2：基尼重要性条形图
    top_gini = importance_df.sort_values('gini_importance', ascending=False).head(10)
    bars2 = axes[1].barh(range(len(top_gini)), top_gini['gini_importance'],
                         color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_yticks(range(len(top_gini)))
    axes[1].set_yticklabels(top_gini['feature_cn'])
    axes[1].set_xlabel('基尼重要性', fontsize=12)
    axes[1].set_ylabel('特征', fontsize=12)
    axes[1].set_title('随机森林基尼重要性 (TOP10)', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("  ✓ 特征重要性图已保存: feature_importance.png")
    plt.close()


def did_analysis(df, event_col='is_payday_near', event_name='发薪日前后'):
    """
    步骤3：双重差分法(DID)分析
    """
    print("\n" + "=" * 70)
    print(f"步骤3：双重差分法(DID)分析 - 事件: {event_name}")
    print("=" * 70)
    
    # 构建DID变量
    # Treat: 处理组虚拟变量（是否处于特定时间段）
    # Post: 事件后虚拟变量（这里简化为是否处于特定日期）
    
    # 为了进行DID分析，我们需要定义处理组和对照组
    # 处理组：发薪日前后3天的消费
    # 对照组：其他时间的消费
    
    df['Treat'] = df[event_col]
    
    # 定义Post变量：以数据中间时间点为界
    mid_date = df['transaction_time'].median()
    df['Post'] = (df['transaction_time'] >= mid_date).astype(int)
    
    # 构建交互项
    df['Treat_Post'] = df['Treat'] * df['Post']
    
    # 准备回归变量
    X_cols = ['Treat', 'Post', 'Treat_Post', 'is_weekend', 'category_encoded']
    X = df[X_cols]
    X = sm.add_constant(X)  # 添加常数项
    y = df['transaction_amount']
    
    # 运行OLS回归
    model = sm.OLS(y, X).fit()
    
    print(f"\nDID回归结果:")
    print(model.summary().tables[1])
    
    # 提取关键系数
    treat_coef = model.params['Treat']
    post_coef = model.params['Post']
    did_coef = model.params['Treat_Post']
    did_pvalue = model.pvalues['Treat_Post']
    
    print(f"\n核心结果解读:")
    print(f"  Treat系数 (处理组vs对照组事前差异): {treat_coef:.2f}")
    print(f"  Post系数 (时间趋势): {post_coef:.2f}")
    print(f"  DID系数 (事件净效应): {did_coef:.2f} (p={did_pvalue:.4f})")
    
    if did_pvalue < 0.01:
        significance = "极显著 (p<0.01)"
    elif did_pvalue < 0.05:
        significance = "显著 (p<0.05)"
    else:
        significance = "不显著 (p>=0.05)"
    
    print(f"  显著性水平: {significance}")
    
    return model, did_coef, did_pvalue


def parallel_trend_test(df, event_col='is_payday_near'):
    """
    步骤4：平行趋势检验
    """
    print("\n" + "=" * 70)
    print("步骤4：平行趋势检验")
    print("=" * 70)
    
    # 按日期分组计算处理组和对照组的平均消费
    df['date'] = df['transaction_time'].dt.date
    daily_stats = df.groupby(['date', event_col])['transaction_amount'].mean().reset_index()
    daily_stats.columns = ['date', 'treat', 'avg_amount']
    
    # 转换为透视表
    pivot = daily_stats.pivot(index='date', columns='treat', values='avg_amount').fillna(0)
    if len(pivot.columns) < 2:
        print("  ⚠ 数据不足以进行平行趋势检验（处理组或对照组缺失）")
        return None
    
    pivot.columns = ['control', 'treat']
    pivot['diff'] = pivot['treat'] - pivot['control']
    
    # 绘制趋势图
    fig, ax = plt.subplots(figsize=(14, 6))
    
    dates = pd.to_datetime(pivot.index)
    ax.plot(dates, pivot['control'], 'b-o', label='对照组（其他时间）', linewidth=2, markersize=6)
    ax.plot(dates, pivot['treat'], 'r-s', label='处理组（发薪日前后）', linewidth=2, markersize=6)
    
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('平均消费金额(元)', fontsize=12)
    ax.set_title('平行趋势检验：处理组vs对照组消费趋势', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parallel_trend.png', dpi=300, bbox_inches='tight')
    print("  ✓ 平行趋势图已保存: parallel_trend.png")
    plt.close()
    
    # 计算趋势差异的统计检验
    pre_period = int(len(pivot) * 0.5)  # 前半段为事前
    pre_diff = pivot['diff'].iloc[:pre_period]
    
    if len(pre_diff) > 1:
        from scipy import stats
        t_stat, p_val = stats.ttest_1samp(pre_diff, 0)
        print(f"\n  事前趋势差异检验: t={t_stat:.2f}, p={p_val:.4f}")
        if p_val > 0.05:
            print("  ✓ 平行趋势假设成立（事前差异不显著）")
        else:
            print("  ⚠ 平行趋势假设可能不成立（事前差异显著）")
    
    return pivot


def analyze_by_category(df):
    """
    步骤5：按消费类目的详细分析
    """
    print("\n" + "=" * 70)
    print("步骤5：按消费类目的影响因素分析")
    print("=" * 70)
    
    category_analysis = []
    
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        
        analysis = {
            'category': category,
            'count': len(cat_data),
            'avg_amount': cat_data['transaction_amount'].mean(),
            'weekend_pct': cat_data['is_weekend'].mean() * 100,
            'payday_pct': cat_data['is_payday_near'].mean() * 100,
            'necessary_pct': cat_data['is_necessary'].mean() * 100
        }
        category_analysis.append(analysis)
    
    cat_df = pd.DataFrame(category_analysis)
    cat_df = cat_df.sort_values('avg_amount', ascending=False)
    
    print(f"\n各类目消费特征:")
    print(f"{'类目':<12}{'笔数':<8}{'均价(元)':<12}{'周末占比(%)':<14}{'发薪日占比(%)':<14}")
    print("-" * 65)
    for _, row in cat_df.iterrows():
        print(f"{row['category']:<12}{int(row['count']):<8}{row['avg_amount']:<12.2f}"
              f"{row['weekend_pct']:<14.1f}{row['payday_pct']:<14.1f}")
    
    return cat_df


def generate_report(df, importance_df, did_results, category_df):
    """
    步骤6：生成分析报告
    """
    print("\n" + "=" * 70)
    print("步骤6：生成分析报告")
    print("=" * 70)
    
    model, did_coef, did_pvalue = did_results
    
    # 保存特征重要性
    importance_export = importance_df[['rank', 'feature_cn', 'importance_mean', 'importance_pct', 'gini_importance']].copy()
    importance_export.columns = ['排名', '特征名称', '置换重要性', '贡献度(%)', '基尼重要性']
    importance_export.to_csv('feature_importance.csv', index=False, encoding='utf-8-sig')
    print("✓ 特征重要性结果已保存: feature_importance.csv")
    
    # 保存DID结果
    did_summary = pd.DataFrame({
        '变量': ['Treat', 'Post', 'Treat_Post(DID)', 'is_weekend', 'category_encoded'],
        '系数': [model.params.get(v, 0) for v in ['Treat', 'Post', 'Treat_Post', 'is_weekend', 'category_encoded']],
        '标准误': [model.bse.get(v, 0) for v in ['Treat', 'Post', 'Treat_Post', 'is_weekend', 'category_encoded']],
        'p值': [model.pvalues.get(v, 1) for v in ['Treat', 'Post', 'Treat_Post', 'is_weekend', 'category_encoded']],
        '显著性': ['***' if model.pvalues.get(v, 1) < 0.01 else ('**' if model.pvalues.get(v, 1) < 0.05 else ('*' if model.pvalues.get(v, 1) < 0.1 else '')) 
                   for v in ['Treat', 'Post', 'Treat_Post', 'is_weekend', 'category_encoded']]
    })
    did_summary.to_csv('did_results.csv', index=False, encoding='utf-8-sig')
    print("✓ DID结果已保存: did_results.csv")
    
    # 保存类目分析
    category_df.to_csv('category_analysis.csv', index=False, encoding='utf-8-sig')
    print("✓ 类目分析已保存: category_analysis.csv")
    
    # 生成文字报告
    report = []
    report.append("=" * 70)
    report.append("消费影响因素分析报告 (随机森林+DID)")
    report.append("=" * 70)
    report.append(f"\n生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"数据时间范围: {df['transaction_time'].min().date()} 至 {df['transaction_time'].max().date()}")
    report.append(f"样本量: {len(df)} 笔交易记录")
    
    # 特征重要性部分
    report.append("\n" + "-" * 70)
    report.append("一、随机森林特征重要性分析")
    report.append("-" * 70)
    
    top3_features = importance_df.head(3)
    report.append(f"\nTOP3重要特征:")
    for i, (_, row) in enumerate(top3_features.iterrows(), 1):
        report.append(f"  {i}. {row['feature_cn']}: 重要性={row['importance_mean']:.4f}, 贡献度={row['importance_pct']:.1f}%")
    
    # DID部分
    report.append("\n" + "-" * 70)
    report.append("二、双重差分法(DID)因果效应估计")
    report.append("-" * 70)
    report.append(f"\n事件: 发薪日前后3天")
    report.append(f"DID系数 (平均处理效应): {did_coef:.2f}元")
    report.append(f"p值: {did_pvalue:.4f}")
    report.append(f"显著性: {'显著' if did_pvalue < 0.05 else '不显著'} (α=0.05)")
    
    # 类目分析部分
    report.append("\n" + "-" * 70)
    report.append("三、各类目消费特征")
    report.append("-" * 70)
    for _, row in category_df.iterrows():
        report.append(f"\n{row['category']}:")
        report.append(f"  交易笔数: {int(row['count'])}")
        report.append(f"  平均金额: {row['avg_amount']:.2f}元")
        report.append(f"  周末占比: {row['weekend_pct']:.1f}%")
        report.append(f"  发薪日附近占比: {row['payday_pct']:.1f}%")
    
    report.append("\n" + "=" * 70)
    report.append("注: *p<0.1, **p<0.05, ***p<0.01")
    report.append("=" * 70)
    
    report_text = '\n'.join(report)
    
    with open('influencing_factors_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ 分析报告已保存: influencing_factors_report.txt")
    print("\n输出文件列表:")
    print("  1. feature_importance.csv - 特征重要性结果")
    print("  2. did_results.csv - DID回归结果")
    print("  3. category_analysis.csv - 类目分析")
    print("  4. influencing_factors_report.txt - 文字分析报告")
    print("  5. feature_importance.png - 特征重要性图")
    print("  6. parallel_trend.png - 平行趋势检验图")


def main():
    """
    主函数：执行完整的消费影响因素分析流程
    """
    print("\n" + "=" * 70)
    print("消费影响因素分析工具")
    print("=" * 70)
    print("模型: 随机森林特征重要性 + 双重差分法(DID)")
    print("=" * 70)
    
    try:
        # 步骤1：数据加载与特征工程
        df = load_and_prepare_data(INPUT_FILE)
        
        # 步骤2：随机森林特征重要性分析
        rf_model, importance_df, X, y = random_forest_feature_importance(df)
        visualize_feature_importance(importance_df)
        
        # 步骤3：DID分析
        did_results = did_analysis(df, event_col='is_payday_near', event_name='发薪日前后')
        
        # 步骤4：平行趋势检验
        parallel_trend_test(df, event_col='is_payday_near')
        
        # 步骤5：类目分析
        category_df = analyze_by_category(df)
        
        # 步骤6：生成报告
        generate_report(df, importance_df, did_results, category_df)
        
        print("\n" + "=" * 70)
        print("消费影响因素分析完成！")
        print("=" * 70)
        
        return df, importance_df, did_results
        
    except Exception as e:
        print(f"\n✗ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == '__main__':
    df, importance_df, did_results = main()
