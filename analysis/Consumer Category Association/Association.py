"""
消费类目关联分析：Apriori关联规则模型
功能：挖掘消费类目之间的关联关系
数据来源：intermediate_data_for_review.xlsx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置文件路径（相对路径）
INPUT_FILE = r'..\..\intermediate_data_for_review.xlsx'
OUTPUT_FILE = 'association_rules.csv'

# 模型参数
MIN_SUPPORT = 0.03      # 最小支持度 3%
MIN_CONFIDENCE = 0.30   # 最小置信度 30%
MIN_LIFT = 1.0          # 最小提升度 1.0


def load_and_preprocess_data(file_path):
    """
    步骤1：加载数据并转换为事务-类目格式
    将同一天的所有消费类目作为一个事务
    """
    print("=" * 60)
    print("步骤1：数据加载与预处理")
    print("=" * 60)
    
    # 读取数据
    df = pd.read_excel(file_path)
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    
    print(f"✓ 成功加载数据，共 {len(df)} 条交易记录")
    print(f"时间范围: {df['transaction_time'].min().date()} 至 {df['transaction_time'].max().date()}")
    
    # 查看类目分布
    category_counts = df['category'].value_counts()
    print(f"\n类目分布:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count}笔 ({count/len(df)*100:.1f}%)")
    
    # 按日期聚合，将同一天的所有类目作为一个事务
    df['date'] = df['transaction_time'].dt.date
    
    # 构建事务数据库
    transactions = []
    for date, group in df.groupby('date'):
        # 获取该日期所有的消费类目（去重）
        categories = set(group['category'].tolist())
        if len(categories) > 0:
            transactions.append({
                'date': date,
                'items': categories,
                'item_count': len(categories)
            })
    
    print(f"\n✓ 事务构建完成，共 {len(transactions)} 个事务（按天聚合）")
    
    # 统计每个事务的类目数量
    item_counts = [t['item_count'] for t in transactions]
    print(f"每个事务平均包含 {np.mean(item_counts):.2f} 个类目")
    print(f"事务包含类目数范围: {min(item_counts)} - {max(item_counts)}")
    
    return transactions, df


def calculate_support(itemset, transactions):
    """
    计算项集的支持度
    """
    count = 0
    for transaction in transactions:
        if itemset.issubset(transaction['items']):
            count += 1
    return count / len(transactions)


def generate_frequent_1_itemsets(transactions, min_support):
    """
    生成频繁1-项集
    """
    print("\n" + "=" * 60)
    print("步骤2：生成频繁项集")
    print("=" * 60)
    
    # 统计每个项的出现次数
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction['items']:
            item_counts[item] += 1
    
    total_transactions = len(transactions)
    
    # 筛选满足最小支持度的1-项集
    frequent_1 = {}
    for item, count in item_counts.items():
        support = count / total_transactions
        if support >= min_support:
            frequent_1[frozenset([item])] = support
    
    print(f"\n频繁1-项集 (支持度 >= {min_support*100:.0f}%):")
    for itemset, support in sorted(frequent_1.items(), key=lambda x: x[1], reverse=True):
        print(f"  {set(itemset)}: 支持度={support:.3f} ({support*100:.1f}%)")
    
    return frequent_1


def apriori_gen(frequent_k_minus_1, k):
    """
    连接步：从频繁(k-1)-项集生成候选k-项集
    """
    candidates = []
    itemsets = list(frequent_k_minus_1.keys())
    
    for i in range(len(itemsets)):
        for j in range(i + 1, len(itemsets)):
            # 获取两个项集
            itemset1 = list(itemsets[i])
            itemset2 = list(itemsets[j])
            
            # 对项集进行排序以便比较
            itemset1.sort()
            itemset2.sort()
            
            # 如果前k-2个项相同，则可以连接
            if itemset1[:k-2] == itemset2[:k-2]:
                # 生成候选k-项集
                candidate = frozenset(itemsets[i]) | frozenset(itemsets[j])
                if len(candidate) == k:
                    candidates.append(candidate)
    
    return candidates


def has_infrequent_subset(candidate, frequent_k_minus_1):
    """
    剪枝步：检查候选k-项集是否有非频繁的(k-1)-子集
    """
    k = len(candidate)
    # 生成所有(k-1)-子集
    subsets = list(combinations(candidate, k - 1))
    
    for subset in subsets:
        if frozenset(subset) not in frequent_k_minus_1:
            return True
    
    return False


def generate_frequent_k_itemsets(transactions, frequent_k_minus_1, k, min_support):
    """
    生成频繁k-项集
    """
    # 连接步：生成候选k-项集
    candidates = apriori_gen(frequent_k_minus_1, k)
    
    if not candidates:
        return {}
    
    # 剪枝步：删除有非频繁子集的候选项集
    pruned_candidates = []
    for candidate in candidates:
        if not has_infrequent_subset(candidate, frequent_k_minus_1):
            pruned_candidates.append(candidate)
    
    if not pruned_candidates:
        return {}
    
    # 扫描数据库，计算支持度
    frequent_k = {}
    total_transactions = len(transactions)
    
    for candidate in pruned_candidates:
        support = calculate_support(candidate, transactions)
        if support >= min_support:
            frequent_k[candidate] = support
    
    return frequent_k


def run_apriori(transactions, min_support):
    """
    执行完整的Apriori算法
    """
    all_frequent_itemsets = {}
    
    # 生成频繁1-项集
    frequent_1 = generate_frequent_1_itemsets(transactions, min_support)
    all_frequent_itemsets.update(frequent_1)
    
    if not frequent_1:
        print("\n✗ 未找到频繁1-项集，请降低最小支持度")
        return {}
    
    # 迭代生成频繁k-项集 (k >= 2)
    k = 2
    frequent_k_minus_1 = frequent_1
    
    while frequent_k_minus_1:
        frequent_k = generate_frequent_k_itemsets(transactions, frequent_k_minus_1, k, min_support)
        
        if frequent_k:
            print(f"\n频繁{k}-项集 (支持度 >= {min_support*100:.0f}%):")
            for itemset, support in sorted(frequent_k.items(), key=lambda x: x[1], reverse=True):
                print(f"  {set(itemset)}: 支持度={support:.3f} ({support*100:.1f}%)")
            all_frequent_itemsets.update(frequent_k)
            frequent_k_minus_1 = frequent_k
            k += 1
        else:
            break
    
    print(f"\n✓ 频繁项集挖掘完成，共找到 {len(all_frequent_itemsets)} 个频繁项集")
    
    return all_frequent_itemsets


def generate_association_rules(frequent_itemsets, transactions, min_confidence, min_lift):
    """
    步骤3：从频繁项集中生成关联规则
    """
    print("\n" + "=" * 60)
    print("步骤3：生成关联规则")
    print("=" * 60)
    
    rules = []
    
    # 只从包含2个或更多项的频繁项集中生成规则
    for itemset, support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        
        # 生成所有非空真子集作为前项
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                
                if not consequent:
                    continue
                
                # 计算置信度
                antecedent_support = frequent_itemsets.get(antecedent, 0)
                if antecedent_support == 0:
                    antecedent_support = calculate_support(antecedent, transactions)
                
                confidence = support / antecedent_support if antecedent_support > 0 else 0
                
                # 计算后项支持度
                consequent_support = frequent_itemsets.get(consequent, 0)
                if consequent_support == 0:
                    consequent_support = calculate_support(consequent, transactions)
                
                # 计算提升度
                lift = confidence / consequent_support if consequent_support > 0 else 0
                
                # 筛选满足条件的规则
                if confidence >= min_confidence and lift >= min_lift:
                    rules.append({
                        'antecedent': set(antecedent),
                        'consequent': set(consequent),
                        'antecedent_support': antecedent_support,
                        'consequent_support': consequent_support,
                        'support': support,
                        'confidence': confidence,
                        'lift': lift
                    })
    
    # 按提升度排序
    rules.sort(key=lambda x: x['lift'], reverse=True)
    
    print(f"\n生成 {len(rules)} 条满足条件的关联规则:")
    print(f"  (置信度 >= {min_confidence*100:.0f}%, 提升度 >= {min_lift})")
    
    return rules


def visualize_results(rules, frequent_itemsets):
    """
    步骤4：可视化关联规则
    """
    print("\n" + "=" * 60)
    print("步骤4：生成可视化")
    print("=" * 60)
    
    if not rules:
        print("✗ 无可视化的关联规则")
        return
    
    # 图1：关联规则散点图（支持度 vs 置信度，气泡大小表示提升度）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    supports = [r['support'] for r in rules]
    confidences = [r['confidence'] for r in rules]
    lifts = [r['lift'] for r in rules]
    
    # 散点图
    scatter = axes[0].scatter(supports, confidences, 
                              s=[l*50 for l in lifts], 
                              c=lifts, cmap='YlOrRd', 
                              alpha=0.6, edgecolors='black')
    axes[0].set_xlabel('支持度 (Support)', fontsize=12)
    axes[0].set_ylabel('置信度 (Confidence)', fontsize=12)
    axes[0].set_title('关联规则散点图\n(气泡大小表示提升度)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=MIN_CONFIDENCE, color='blue', linestyle='--', alpha=0.5, label=f'最小置信度={MIN_CONFIDENCE}')
    axes[0].axvline(x=MIN_SUPPORT, color='green', linestyle='--', alpha=0.5, label=f'最小支持度={MIN_SUPPORT}')
    axes[0].legend()
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label('提升度 (Lift)', fontsize=10)
    
    # 图2：TOP规则柱状图
    top_n = min(10, len(rules))
    top_rules = rules[:top_n]
    
    rule_labels = [f"{set(r['antecedent'])} →\n{set(r['consequent'])}" for r in top_rules]
    rule_lifts = [r['lift'] for r in top_rules]
    
    bars = axes[1].barh(range(len(rule_labels)), rule_lifts, color='steelblue')
    axes[1].set_yticks(range(len(rule_labels)))
    axes[1].set_yticklabels(rule_labels, fontsize=9)
    axes[1].set_xlabel('提升度 (Lift)', fontsize=12)
    axes[1].set_title(f'TOP {top_n} 关联规则（按提升度排序）', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # 在柱状图上标注数值
    for i, (bar, lift) in enumerate(zip(bars, rule_lifts)):
        axes[1].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{lift:.2f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('association_rules_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ 关联规则可视化图已保存: association_rules_visualization.png")
    
    plt.close()


def print_detailed_rules(rules):
    """
    打印详细的关联规则
    """
    print("\n" + "=" * 60)
    print("关联规则详情")
    print("=" * 60)
    
    if not rules:
        print("\n✗ 未找到满足条件的关联规则")
        print("建议: 尝试降低最小支持度、最小置信度或最小提升度")
        return
    
    for i, rule in enumerate(rules, 1):
        antecedent = ', '.join(rule['antecedent'])
        consequent = ', '.join(rule['consequent'])
        
        print(f"\n规则 R{i}: {{{antecedent}}} → {{{consequent}}}")
        print(f"  支持度 (Support): {rule['support']:.3f} ({rule['support']*100:.1f}%)")
        print(f"  置信度 (Confidence): {rule['confidence']:.3f} ({rule['confidence']*100:.1f}%)")
        print(f"  提升度 (Lift): {rule['lift']:.3f}")
        
        # 业务解读
        if rule['lift'] > 2:
            strength = "强关联"
        elif rule['lift'] > 1.5:
            strength = "中等关联"
        else:
            strength = "弱关联"
        
        print(f"  关联强度: {strength}")
        print(f"  业务含义: 当出现'{antecedent}'消费时，")
        print(f"           有{rule['confidence']*100:.1f}%的概率同时出现'{consequent}'消费")
        print(f"           关联强度是平均水平的{rule['lift']:.2f}倍")


def save_results(rules, frequent_itemsets, transactions):
    """
    步骤5：保存结果
    """
    print("\n" + "=" * 60)
    print("步骤5：保存结果")
    print("=" * 60)
    
    # 保存关联规则到CSV
    if rules:
        rules_df = pd.DataFrame([
            {
                '规则编号': f'R{i+1}',
                '前项(X)': ', '.join(r['antecedent']),
                '后项(Y)': ', '.join(r['consequent']),
                '前项支持度': f"{r['antecedent_support']:.3f}",
                '后项支持度': f"{r['consequent_support']:.3f}",
                '规则支持度': f"{r['support']:.3f}",
                '置信度': f"{r['confidence']:.3f}",
                '提升度': f"{r['lift']:.3f}"
            }
            for i, r in enumerate(rules)
        ])
        rules_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"✓ 关联规则已保存: {OUTPUT_FILE}")
    
    # 保存频繁项集
    frequent_df = pd.DataFrame([
        {
            '项集': ', '.join(itemset),
            '项集大小': len(itemset),
            '支持度': f"{support:.3f}",
            '支持度(%)': f"{support*100:.1f}%"
        }
        for itemset, support in sorted(frequent_itemsets.items(), key=lambda x: x[1], reverse=True)
    ])
    frequent_df.to_csv('frequent_itemsets.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 频繁项集已保存: frequent_itemsets.csv")
    
    # 生成分析报告
    report = []
    report.append("=" * 60)
    report.append("消费类目关联分析报告 (Apriori算法)")
    report.append("=" * 60)
    report.append(f"\n分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    report.append("\n" + "-" * 60)
    report.append("数据概览")
    report.append("-" * 60)
    report.append(f"事务总数: {len(transactions)} 天")
    report.append(f"最小支持度: {MIN_SUPPORT*100:.0f}%")
    report.append(f"最小置信度: {MIN_CONFIDENCE*100:.0f}%")
    report.append(f"最小提升度: {MIN_LIFT}")
    
    report.append("\n" + "-" * 60)
    report.append("频繁项集统计")
    report.append("-" * 60)
    report.append(f"频繁项集总数: {len(frequent_itemsets)}")
    
    # 按项集大小分类统计
    size_counts = Counter([len(itemset) for itemset in frequent_itemsets.keys()])
    for size in sorted(size_counts.keys()):
        report.append(f"  {size}-项集: {size_counts[size]} 个")
    
    report.append("\n" + "-" * 60)
    report.append("关联规则统计")
    report.append("-" * 60)
    report.append(f"关联规则总数: {len(rules)}")
    
    if rules:
        avg_lift = np.mean([r['lift'] for r in rules])
        max_lift = max([r['lift'] for r in rules])
        report.append(f"平均提升度: {avg_lift:.2f}")
        report.append(f"最大提升度: {max_lift:.2f}")
    
    report.append("\n" + "-" * 60)
    report.append("TOP 10 关联规则")
    report.append("-" * 60)
    
    for i, rule in enumerate(rules[:10], 1):
        antecedent = ', '.join(rule['antecedent'])
        consequent = ', '.join(rule['consequent'])
        report.append(f"\nR{i}: {{{antecedent}}} → {{{consequent}}}")
        report.append(f"    支持度={rule['support']:.3f}, "
                     f"置信度={rule['confidence']:.3f}, "
                     f"提升度={rule['lift']:.3f}")
    
    report.append("\n" + "=" * 60)
    
    report_text = '\n'.join(report)
    
    with open('association_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ 分析报告已保存: association_report.txt")
    print("\n输出文件列表:")
    print(f"  1. {OUTPUT_FILE} - 关联规则数据")
    print(f"  2. frequent_itemsets.csv - 频繁项集数据")
    print(f"  3. association_report.txt - 文字分析报告")
    print(f"  4. association_rules_visualization.png - 关联规则可视化图")


def main():
    """
    主函数：执行完整的关联分析流程
    """
    print("\n" + "=" * 60)
    print("消费类目关联分析工具")
    print("=" * 60)
    print("算法: Apriori关联规则挖掘")
    print("=" * 60)
    
    try:
        # 步骤1：数据加载与预处理
        transactions, raw_df = load_and_preprocess_data(INPUT_FILE)
        
        if len(transactions) < 10:
            print("\n✗ 事务数量过少，无法进行关联分析")
            return None, None
        
        # 步骤2：生成频繁项集
        frequent_itemsets = run_apriori(transactions, MIN_SUPPORT)
        
        if not frequent_itemsets:
            print("\n✗ 未找到频繁项集，请调整参数后重试")
            return None, None
        
        # 步骤3：生成关联规则
        rules = generate_association_rules(frequent_itemsets, transactions, 
                                          MIN_CONFIDENCE, MIN_LIFT)
        
        # 打印详细规则
        print_detailed_rules(rules)
        
        # 步骤4：可视化
        visualize_results(rules, frequent_itemsets)
        
        # 步骤5：保存结果
        save_results(rules, frequent_itemsets, transactions)
        
        print("\n" + "=" * 60)
        print("关联分析完成！")
        print("=" * 60)
        
        return rules, frequent_itemsets
        
    except Exception as e:
        print(f"\n✗ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == '__main__':
    rules, frequent_itemsets = main()
