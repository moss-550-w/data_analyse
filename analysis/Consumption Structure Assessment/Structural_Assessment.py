"""
消费结构评估：洛伦兹曲线+基尼系数+方差分析组合模型
功能：评估消费支出的分布均衡性和影响因素显著性
数据来源：intermediate_data_for_review.xlsx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置文件路径（相对路径）
INPUT_FILE = r'..\..\intermediate_data_for_review.xlsx'
OUTPUT_FILE = 'assessment_results.csv'


def load_and_preprocess_data(file_path):
    """
    步骤1：加载数据并进行预处理
    """
    print("=" * 60)
    print("步骤1：数据加载与预处理")
    print("=" * 60)
    
    # 读取数据
    df = pd.read_excel(file_path)
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    
    print(f"✓ 成功加载数据，共 {len(df)} 条交易记录")
    print(f"时间范围: {df['transaction_time'].min().date()} 至 {df['transaction_time'].max().date()}")
    
    # 提取时间特征
    df['month'] = df['transaction_time'].dt.month
    df['weekday'] = df['transaction_time'].dt.weekday  # 0=周一, 6=周日
    df['weekday_name'] = df['transaction_time'].dt.day_name()
    
    # 映射星期名称（中文）
    weekday_map = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 
                   4: '周五', 5: '周六', 6: '周日'}
    df['weekday_cn'] = df['weekday'].map(weekday_map)
    
    # 类目消费统计
    category_stats = df.groupby('category')['transaction_amount'].agg(['sum', 'count', 'mean']).reset_index()
    category_stats.columns = ['category', 'total_amount', 'count', 'avg_amount']
    category_stats = category_stats.sort_values('total_amount')
    category_stats['amount_pct'] = category_stats['total_amount'] / category_stats['total_amount'].sum()
    
    print(f"\n类目消费分布:")
    for _, row in category_stats.iterrows():
        print(f"  {row['category']}: 金额={row['total_amount']:.2f}元 "
              f"({row['amount_pct']*100:.1f}%), 笔数={int(row['count'])}, 均价={row['avg_amount']:.2f}元")
    
    return df, category_stats


def calculate_lorenz_curve(category_stats):
    """
    步骤2：计算洛伦兹曲线和基尼系数
    """
    print("\n" + "=" * 60)
    print("步骤2：洛伦兹曲线与基尼系数计算")
    print("=" * 60)
    
    # 按消费金额从小到大排序
    sorted_stats = category_stats.sort_values('total_amount').reset_index(drop=True)
    n = len(sorted_stats)
    
    # 计算累计类目占比和累计金额占比
    sorted_stats['cumulative_amount'] = sorted_stats['total_amount'].cumsum()
    sorted_stats['cumulative_pct'] = sorted_stats['cumulative_amount'] / sorted_stats['total_amount'].sum()
    sorted_stats['category_pct'] = (np.arange(n) + 1) / n
    
    # 添加起点(0,0)
    category_pcts = [0] + sorted_stats['category_pct'].tolist()
    amount_pcts = [0] + sorted_stats['cumulative_pct'].tolist()
    
    # 使用梯形法计算基尼系数
    # G = 1 - sum((p_k - p_{k-1}) * (L(p_k) + L(p_{k-1})))
    gini = 0
    for i in range(1, len(category_pcts)):
        gini += (category_pcts[i] - category_pcts[i-1]) * (amount_pcts[i] + amount_pcts[i-1])
    gini = 1 - gini
    
    print(f"\n✓ 基尼系数计算完成: G = {gini:.3f}")
    
    # 基尼系数解读
    if gini < 0.20:
        level = "消费高度分散，各类目支出均衡"
        evaluation = "过于分散，可能存在不必要的零散消费"
    elif gini < 0.40:
        level = "消费相对分散，结构较为合理"
        evaluation = "合理，兼顾必要消费和多元化需求"
    elif gini < 0.60:
        level = "消费相对集中，少数类目占比较高"
        evaluation = "需关注，重点分析高占比类目的合理性"
    elif gini < 0.80:
        level = "消费高度集中，结构单一化"
        evaluation = "不合理，存在较大优化空间"
    else:
        level = "消费极度集中，几乎全部支出在1-2个类目"
        evaluation = "严重不合理，需立即调整消费结构"
    
    print(f"  消费结构特征: {level}")
    print(f"  合理性评价: {evaluation}")
    
    # 输出TOP3类目占比
    top3 = sorted_stats.tail(3)
    print(f"\nTOP3消费类目:")
    for _, row in top3.iloc[::-1].iterrows():
        print(f"  {row['category']}: {row['total_amount']:.2f}元 ({row['amount_pct']*100:.1f}%)")
    
    return gini, category_pcts, amount_pcts, sorted_stats


def visualize_lorenz_curve(gini, category_pcts, amount_pcts, sorted_stats):
    """
    绘制洛伦兹曲线
    """
    print("\n  生成洛伦兹曲线...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制绝对平等线
    ax.plot([0, 1], [0, 1], 'k--', label='绝对平等线', linewidth=1.5)
    
    # 绘制洛伦兹曲线
    ax.plot(category_pcts, amount_pcts, 'b-', linewidth=2.5, label=f'洛伦兹曲线 (G={gini:.3f})')
    
    # 填充曲线下方面积
    ax.fill_between(category_pcts, amount_pcts, alpha=0.3, color='blue')
    
    # 标注关键点
    for i in range(1, len(category_pcts)):
        if i == len(category_pcts) - 1 or i % max(1, (len(category_pcts)-1)//3) == 0:
            cat_name = sorted_stats.iloc[i-1]['category']
            ax.annotate(f'{cat_name}\n({amount_pcts[i]*100:.1f}%)', 
                       xy=(category_pcts[i], amount_pcts[i]),
                       xytext=(category_pcts[i]+0.05, amount_pcts[i]-0.1),
                       fontsize=9, ha='left',
                       arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    
    ax.set_xlabel('累计消费类目占比', fontsize=12)
    ax.set_ylabel('累计消费金额占比', fontsize=12)
    ax.set_title('消费支出洛伦兹曲线', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('lorenz_curve.png', dpi=300, bbox_inches='tight')
    print("  ✓ 洛伦兹曲线已保存: lorenz_curve.png")
    
    plt.close()


def one_way_anova(df, factor_col, value_col='transaction_amount'):
    """
    步骤3：单因素方差分析
    """
    print(f"\n" + "=" * 60)
    print(f"步骤3：单因素方差分析 - 因素: {factor_col}")
    print("=" * 60)
    
    # 获取各组数据
    groups = df.groupby(factor_col)[value_col].apply(list).to_dict()
    group_names = list(groups.keys())
    group_data = [groups[name] for name in group_names]
    
    # 描述性统计
    print(f"\n描述性统计:")
    desc_stats = df.groupby(factor_col)[value_col].agg(['count', 'mean', 'std']).reset_index()
    desc_stats.columns = [factor_col, '样本数', '均值(元)', '标准差(元)']
    print(desc_stats.to_string(index=False))
    
    # 正态性检验 (Shapiro-Wilk，对每个组分别检验)
    print(f"\n正态性检验 (Shapiro-Wilk):")
    all_normal = True
    for name, data in groups.items():
        if len(data) >= 3:  # 至少需要3个样本
            stat, p_value = stats.shapiro(data)
            normal = "✓ 正态" if p_value > 0.05 else "✗ 非正态"
            print(f"  {name}: W={stat:.4f}, p={p_value:.4f} {normal}")
            if p_value <= 0.05:
                all_normal = False
        else:
            print(f"  {name}: 样本数不足，跳过检验")
    
    # 方差齐性检验 (Levene检验)
    print(f"\n方差齐性检验 (Levene):")
    stat, p_value = stats.levene(*group_data)
    equal_var = "✓ 方差齐性" if p_value > 0.05 else "✗ 方差不齐"
    print(f"  W={stat:.4f}, p={p_value:.4f} {equal_var}")
    
    # 单因素方差分析
    print(f"\n方差分析结果:")
    f_stat, p_value = stats.f_oneway(*group_data)
    
    # 计算平方和与自由度
    all_data = df[value_col].values
    grand_mean = np.mean(all_data)
    
    # 组间平方和
    ss_between = sum(len(groups[name]) * (np.mean(groups[name]) - grand_mean)**2 
                     for name in group_names)
    df_between = len(group_names) - 1
    ms_between = ss_between / df_between if df_between > 0 else 0
    
    # 组内平方和
    ss_within = sum(sum((x - np.mean(groups[name]))**2 for x in groups[name]) 
                    for name in group_names)
    df_within = len(all_data) - len(group_names)
    ms_within = ss_within / df_within if df_within > 0 else 0
    
    # 总平方和
    ss_total = sum((x - grand_mean)**2 for x in all_data)
    
    print(f"\n方差分析表:")
    print(f"{'变异来源':<12} {'平方和':<12} {'自由度':<8} {'均方':<12} {'F值':<10} {'p值':<10}")
    print(f"{'-'*70}")
    print(f"{'组间':<12} {ss_between:<12.2f} {df_between:<8} {ms_between:<12.2f} {f_stat:<10.2f} {p_value:<10.4f}")
    print(f"{'组内':<12} {ss_within:<12.2f} {df_within:<8} {ms_within:<12.2f}")
    print(f"{'总计':<12} {ss_total:<12.2f} {len(all_data)-1:<8}")
    
    # 结果解读
    alpha = 0.05
    if p_value < alpha:
        significance = "具有统计学显著性"
        print(f"\n✓ 结论: p={p_value:.4f} < {alpha}，拒绝原假设")
        print(f"  不同{factor_col}的消费金额存在显著差异 ({significance})")
    else:
        print(f"\n✗ 结论: p={p_value:.4f} >= {alpha}，不拒绝原假设")
        print(f"  不同{factor_col}的消费金额无显著差异")
    
    return f_stat, p_value, group_names, group_data


def tukey_hsd_test(df, factor_col, value_col='transaction_amount'):
    """
    步骤4：Tukey HSD多重比较检验
    """
    print(f"\n" + "=" * 60)
    print(f"步骤4：Tukey HSD多重比较检验")
    print("=" * 60)
    
    # 执行Tukey HSD检验
    tukey = pairwise_tukeyhsd(endog=df[value_col], 
                               groups=df[factor_col], 
                               alpha=0.05)
    
    print(f"\n{tukey}")
    
    # 提取显著差异的组对
    results = tukey.summary().data[1:]  # 跳过表头
    significant_pairs = []
    
    for row in results:
        group1, group2, meandiff, p_adj, lower, upper, reject = row
        if reject:
            significant_pairs.append((group1, group2, meandiff, p_adj))
    
    if significant_pairs:
        print(f"\n✓ 存在显著差异的组对:")
        for g1, g2, diff, p in significant_pairs:
            print(f"  {g1} vs {g2}: 均值差={diff:.2f}元, p={p:.4f}")
    else:
        print(f"\n✗ 未发现显著差异的组对")
    
    return tukey, significant_pairs


def visualize_anova_results(df, factor_col, value_col='transaction_amount'):
    """
    可视化方差分析结果
    """
    print(f"\n  生成方差分析可视化...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 图1：箱线图
    df.boxplot(column=value_col, by=factor_col, ax=axes[0])
    axes[0].set_title(f'不同{factor_col}的消费金额分布', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(factor_col, fontsize=12)
    axes[0].set_ylabel('消费金额(元)', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 图2：柱状图（均值+误差线）
    means = df.groupby(factor_col)[value_col].mean()
    stds = df.groupby(factor_col)[value_col].std()
    counts = df.groupby(factor_col)[value_col].count()
    sems = stds / np.sqrt(counts)  # 标准误
    
    bars = axes[1].bar(range(len(means)), means.values, yerr=sems.values, 
                       capsize=5, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].set_xticks(range(len(means)))
    axes[1].set_xticklabels(means.index, rotation=45, ha='right')
    axes[1].set_xlabel(factor_col, fontsize=12)
    axes[1].set_ylabel('平均消费金额(元)', fontsize=12)
    axes[1].set_title(f'不同{factor_col}的平均消费金额（带标准误）', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上标注数值
    for i, (bar, mean) in enumerate(zip(bars, means.values)):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + sems.iloc[i],
                    f'{mean:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'anova_{factor_col}.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 方差分析图已保存: anova_{factor_col}.png")
    
    plt.close()


def save_results(gini, category_stats, anova_results_weekday, anova_results_month, tukey_weekday):
    """
    步骤5：保存结果
    """
    print("\n" + "=" * 60)
    print("步骤5：保存结果")
    print("=" * 60)
    
    # 保存基尼系数和类目统计
    gini_data = {
        '指标': ['基尼系数', '消费结构评价', '类目数量', '总消费金额'],
        '值': [f'{gini:.3f}', get_gini_evaluation(gini), len(category_stats), 
               f'{category_stats["total_amount"].sum():.2f}元']
    }
    gini_df = pd.DataFrame(gini_data)
    gini_df.to_csv('gini_coefficient.csv', index=False, encoding='utf-8-sig')
    print("✓ 基尼系数结果已保存: gini_coefficient.csv")
    
    # 保存类目统计
    category_stats_export = category_stats[['category', 'total_amount', 'count', 'avg_amount', 'amount_pct']].copy()
    category_stats_export.columns = ['消费类目', '总金额(元)', '消费笔数', '平均金额(元)', '金额占比']
    category_stats_export['金额占比'] = category_stats_export['金额占比'].apply(lambda x: f'{x*100:.1f}%')
    category_stats_export.to_csv('category_statistics.csv', index=False, encoding='utf-8-sig')
    print("✓ 类目统计已保存: category_statistics.csv")
    
    # 生成分析报告
    report = []
    report.append("=" * 60)
    report.append("消费结构评估报告 (洛伦兹曲线+基尼系数+ANOVA)")
    report.append("=" * 60)
    report.append(f"\n生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 基尼系数部分
    report.append("\n" + "-" * 60)
    report.append("一、消费结构集中度分析（洛伦兹曲线+基尼系数）")
    report.append("-" * 60)
    report.append(f"\n基尼系数: {gini:.3f}")
    report.append(f"评价: {get_gini_evaluation(gini)}")
    
    report.append(f"\n消费类目分布:")
    sorted_stats = category_stats.sort_values('total_amount', ascending=False)
    cumulative_pct = 0
    for i, (_, row) in enumerate(sorted_stats.iterrows(), 1):
        cumulative_pct += row['amount_pct']
        report.append(f"  {i}. {row['category']}: {row['total_amount']:.2f}元 "
                     f"({row['amount_pct']*100:.1f}%), 累计{cumulative_pct*100:.1f}%")
    
    # 方差分析部分
    if anova_results_weekday:
        report.append("\n" + "-" * 60)
        report.append("二、星期因素方差分析")
        report.append("-" * 60)
        f_stat, p_value, _, _ = anova_results_weekday
        report.append(f"\nF统计量: {f_stat:.2f}")
        report.append(f"p值: {p_value:.4f}")
        report.append(f"结论: {'显著' if p_value < 0.05 else '不显著'} (α=0.05)")
    
    if anova_results_month:
        report.append("\n" + "-" * 60)
        report.append("三、月份因素方差分析")
        report.append("-" * 60)
        f_stat, p_value, _, _ = anova_results_month
        report.append(f"\nF统计量: {f_stat:.2f}")
        report.append(f"p值: {p_value:.4f}")
        report.append(f"结论: {'显著' if p_value < 0.05 else '不显著'} (α=0.05)")
    
    report.append("\n" + "=" * 60)
    
    report_text = '\n'.join(report)
    
    with open('assessment_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ 分析报告已保存: assessment_report.txt")
    print("\n输出文件列表:")
    print("  1. gini_coefficient.csv - 基尼系数结果")
    print("  2. category_statistics.csv - 类目统计")
    print("  3. assessment_report.txt - 文字分析报告")
    print("  4. lorenz_curve.png - 洛伦兹曲线图")
    print("  5. anova_*.png - 方差分析可视化图")


def get_gini_evaluation(gini):
    """
    获取基尼系数评价
    """
    if gini < 0.20:
        return "消费高度分散，过于零散"
    elif gini < 0.40:
        return "消费相对分散，结构合理"
    elif gini < 0.60:
        return "消费相对集中，需关注"
    elif gini < 0.80:
        return "消费高度集中，结构单一"
    else:
        return "消费极度集中，需立即调整"


def main():
    """
    主函数：执行完整的消费结构评估流程
    """
    print("\n" + "=" * 60)
    print("消费结构评估工具")
    print("=" * 60)
    print("模型: 洛伦兹曲线+基尼系数+方差分析")
    print("=" * 60)
    
    try:
        # 步骤1：数据加载与预处理
        df, category_stats = load_and_preprocess_data(INPUT_FILE)
        
        # 步骤2：洛伦兹曲线与基尼系数
        gini, category_pcts, amount_pcts, sorted_stats = calculate_lorenz_curve(category_stats)
        visualize_lorenz_curve(gini, category_pcts, amount_pcts, sorted_stats)
        
        # 步骤3&4：方差分析（按星期）
        anova_results_weekday = None
        tukey_weekday = None
        if df['weekday_cn'].nunique() >= 2:
            anova_results_weekday = one_way_anova(df, 'weekday_cn')
            if anova_results_weekday[1] < 0.05:  # 如果显著，进行Tukey检验
                tukey_weekday, _ = tukey_hsd_test(df, 'weekday_cn')
            visualize_anova_results(df, 'weekday_cn')
        
        # 步骤3&4：方差分析（按月份）
        anova_results_month = None
        if df['month'].nunique() >= 2:
            df['month_str'] = df['month'].astype(str) + '月'
            anova_results_month = one_way_anova(df, 'month_str')
            visualize_anova_results(df, 'month_str')
        
        # 步骤5：保存结果
        save_results(gini, category_stats, anova_results_weekday, anova_results_month, tukey_weekday)
        
        print("\n" + "=" * 60)
        print("消费结构评估完成！")
        print("=" * 60)
        
        return gini, category_stats
        
    except Exception as e:
        print(f"\n✗ 评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == '__main__':
    gini, category_stats = main()
