"""
消费行为数学建模分析
功能：基于PCA降维和K-Means聚类的消费行为分析
数据来源：intermediate_data_for_review.xlsx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置文件路径（相对路径）
INPUT_FILE = r'..\..\intermediate_data_for_review.xlsx'
OUTPUT_FILE = 'clustering_results.csv'


def load_data(file_path):
    """
    加载中间数据文件
    """
    print("=" * 60)
    print("步骤1：加载数据")
    print("=" * 60)
    
    df = pd.read_excel(file_path)
    print(f"✓ 成功加载数据，共 {len(df)} 条记录")
    print(f"列名: {list(df.columns)}")
    return df


def extract_features(df):
    """
    步骤2：提取消费特征
    根据模型要求提取8个量化消费特征
    """
    print("\n" + "=" * 60)
    print("步骤2：提取消费特征")
    print("=" * 60)
    
    # 转换时间为datetime格式
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    
    # 提取时间特征
    df['hour'] = df['transaction_time'].dt.hour
    df['weekday'] = df['transaction_time'].dt.weekday  # 0=周一, 6=周日
    df['is_weekday'] = df['weekday'] < 5
    df['is_daytime'] = (df['hour'] >= 6) & (df['hour'] < 22)
    
    # 按月份分组计算各类目占比
    df['month'] = df['transaction_time'].dt.to_period('M')
    
    # 初始化特征DataFrame
    features_list = []
    
    for idx, row in df.iterrows():
        # 获取同月份的所有数据
        month_data = df[df['month'] == row['month']]
        month_total = month_data['transaction_amount'].sum()
        
        if month_total == 0:
            month_total = 1  # 避免除零
        
        # 特征1：单笔消费金额
        x1 = row['transaction_amount']
        
        # 特征2：餐饮消费占比（该笔所在月份）
        food_amount = month_data[month_data['category'] == '餐饮美食']['transaction_amount'].sum()
        x2 = food_amount / month_total * 100
        
        # 特征3：交通消费占比
        transport_amount = month_data[month_data['category'] == '交通出行']['transaction_amount'].sum()
        x3 = transport_amount / month_total * 100
        
        # 特征4：购物消费占比（超市购物+服饰美容+日用百货）
        shopping_categories = ['超市购物', '服饰美容', '日用百货']
        shopping_amount = month_data[month_data['category'].isin(shopping_categories)]['transaction_amount'].sum()
        x4 = shopping_amount / month_total * 100
        
        # 特征5：娱乐消费占比
        ent_categories = ['休闲娱乐']
        ent_amount = month_data[month_data['category'].isin(ent_categories)]['transaction_amount'].sum()
        x5 = ent_amount / month_total * 100
        
        # 特征6：工作日消费占比
        weekday_amount = month_data[month_data['is_weekday']]['transaction_amount'].sum()
        x6 = weekday_amount / month_total * 100
        
        # 特征7：白天消费占比
        daytime_amount = month_data[month_data['is_daytime']]['transaction_amount'].sum()
        x7 = daytime_amount / month_total * 100
        
        # 特征8：微信支付占比（假设都是微信支付）
        x8 = 100.0
        
        features_list.append({
            'transaction_id': row['transaction_id'],
            'x1_amount': x1,
            'x2_food_ratio': x2,
            'x3_transport_ratio': x3,
            'x4_shopping_ratio': x4,
            'x5_entertainment_ratio': x5,
            'x6_weekday_ratio': x6,
            'x7_daytime_ratio': x7,
            'x8_wechat_ratio': x8,
            'category': row['category'],
            'transaction_amount': row['transaction_amount'],
            'transaction_time': row['transaction_time'],
            'is_weekday': row['is_weekday'],
            'is_daytime': row['is_daytime']
        })
    
    features_df = pd.DataFrame(features_list)
    
    print(f"✓ 特征提取完成")
    print(f"特征矩阵形状: {features_df.shape}")
    print("\n特征说明:")
    print("  x1_amount: 单笔消费金额(元)")
    print("  x2_food_ratio: 餐饮消费占比(%)")
    print("  x3_transport_ratio: 交通消费占比(%)")
    print("  x4_shopping_ratio: 购物消费占比(%)")
    print("  x5_entertainment_ratio: 娱乐消费占比(%)")
    print("  x6_weekday_ratio: 工作日消费占比(%)")
    print("  x7_daytime_ratio: 白天消费占比(%)")
    print("  x8_wechat_ratio: 微信支付占比(%)")
    
    return features_df


def pca_analysis(features_df):
    """
    步骤3：PCA降维分析
    """
    print("\n" + "=" * 60)
    print("步骤3：PCA主成分分析")
    print("=" * 60)
    
    # 选择特征列
    feature_cols = ['x1_amount', 'x2_food_ratio', 'x3_transport_ratio', 
                    'x4_shopping_ratio', 'x5_entertainment_ratio',
                    'x6_weekday_ratio', 'x7_daytime_ratio', 'x8_wechat_ratio']
    
    X = features_df[feature_cols].values
    
    # Z-score标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("✓ 数据标准化完成 (Z-score)")
    
    # PCA分析
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # 计算方差贡献率
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print("\n主成分方差贡献率:")
    print("-" * 50)
    print(f"{'主成分':<10}{'特征值':<12}{'方差贡献率(%)':<15}{'累计方差贡献率(%)':<15}")
    print("-" * 50)
    
    for i in range(len(explained_variance_ratio)):
        eigenvalue = pca.explained_variance_[i]
        ratio = explained_variance_ratio[i] * 100
        cum_ratio = cumulative_variance[i] * 100
        print(f"PC{i+1:<9}{eigenvalue:<12.2f}{ratio:<15.1f}{cum_ratio:<15.1f}")
        
        # 找出满足85%累计方差的最小主成分数
        if cumulative_variance[i] >= 0.85 and i < 3:
            n_components = i + 1
    
    # 使用前3个主成分（根据模型要求）
    n_components = 3
    print(f"\n✓ 选择前{n_components}个主成分，累计方差贡献率: {cumulative_variance[n_components-1]*100:.1f}%")
    
    # 获取主成分载荷矩阵
    loadings = pd.DataFrame(
        pca.components_[:n_components].T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_cols
    )
    
    print("\n主成分载荷矩阵:")
    print(loadings.round(2))
    
    # 解释主成分含义
    print("\n主成分含义解读:")
    print("-" * 50)
    
    # PC1解释
    pc1_loadings = loadings['PC1'].abs().sort_values(ascending=False)
    print(f"PC1（大额消费因子）: 主要与 {pc1_loadings.index[0]}({loadings.loc[pc1_loadings.index[0], 'PC1']:.2f})、"
          f"{pc1_loadings.index[1]}({loadings.loc[pc1_loadings.index[1], 'PC1']:.2f}) 相关")
    
    # PC2解释
    pc2_loadings = loadings['PC2'].abs().sort_values(ascending=False)
    print(f"PC2（日常消费因子）: 主要与 {pc2_loadings.index[0]}({loadings.loc[pc2_loadings.index[0], 'PC2']:.2f})、"
          f"{pc2_loadings.index[1]}({loadings.loc[pc2_loadings.index[1], 'PC2']:.2f}) 相关")
    
    # PC3解释
    pc3_loadings = loadings['PC3'].abs().sort_values(ascending=False)
    print(f"PC3（支付习惯因子）: 主要与 {pc3_loadings.index[0]}({loadings.loc[pc3_loadings.index[0], 'PC3']:.2f})、"
          f"{pc3_loadings.index[1]}({loadings.loc[pc3_loadings.index[1], 'PC3']:.2f}) 相关")
    
    # 使用指定主成分数重新计算
    pca_final = PCA(n_components=n_components)
    X_pca_final = pca_final.fit_transform(X_scaled)
    
    # 将PCA结果添加到DataFrame
    for i in range(n_components):
        features_df[f'PC{i+1}'] = X_pca_final[:, i]
    
    return features_df, pca_final, scaler


def elbow_method(features_df):
    """
    步骤4：肘部法则确定最佳K值
    """
    print("\n" + "=" * 60)
    print("步骤4：肘部法则确定最佳K值")
    print("=" * 60)
    
    # 使用PCA后的3个主成分
    X = features_df[['PC1', 'PC2', 'PC3']].values
    
    # 计算不同K值的SSE
    sse = []
    k_range = range(1, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    
    # 绘制肘部法则图
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('聚类数 K', fontsize=12)
    plt.ylabel('SSE (簇内平方误差和)', fontsize=12)
    plt.title('肘部法则确定最佳K值', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 标记K=4的位置
    plt.axvline(x=4, color='r', linestyle='--', alpha=0.7, label='K=4 (推荐值)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
    print("✓ 肘部法则图已保存: elbow_method.png")
    
    # 根据模型选择K=4
    optimal_k = 4
    print(f"\n✓ 根据肘部法则，选择最佳K值: {optimal_k}")
    
    return optimal_k, sse


def kmeans_clustering(features_df, k):
    """
    步骤5：K-Means聚类分析
    """
    print("\n" + "=" * 60)
    print("步骤5：K-Means聚类分析")
    print("=" * 60)
    
    # 使用PCA后的3个主成分进行聚类
    X = features_df[['PC1', 'PC2', 'PC3']].values
    
    # 执行K-Means聚类
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(X)
    
    # 添加聚类标签
    features_df['cluster'] = cluster_labels
    
    # 计算轮廓系数
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"✓ 聚类完成，平均轮廓系数: {silhouette_avg:.3f}")
    print(f"  （轮廓系数范围[-1, 1]，越接近1表示聚类效果越好）")
    
    # 统计各簇的信息
    print("\n各聚类簇统计:")
    print("-" * 80)
    print(f"{'簇':<6}{'样本数':<10}{'占比(%)':<10}{'平均消费(元)':<15}{'主要消费类目':<20}")
    print("-" * 80)
    
    cluster_summary = []
    
    for i in range(k):
        cluster_data = features_df[features_df['cluster'] == i]
        count = len(cluster_data)
        ratio = count / len(features_df) * 100
        avg_amount = cluster_data['transaction_amount'].mean()
        
        # 找出主要消费类目
        main_category = cluster_data['category'].mode()
        if len(main_category) > 0:
            main_cat = main_category.iloc[0]
        else:
            main_cat = 'N/A'
        
        print(f"{i+1:<6}{count:<10}{ratio:<10.1f}{avg_amount:<15.2f}{main_cat:<20}")
        
        cluster_summary.append({
            'cluster': i + 1,
            'count': count,
            'ratio': ratio,
            'avg_amount': avg_amount,
            'main_category': main_cat
        })
    
    # 详细分析每个簇
    print("\n各聚类簇详细分析:")
    print("=" * 80)
    
    for i in range(k):
        cluster_data = features_df[features_df['cluster'] == i]
        print(f"\n【簇{i+1}】")
        print(f"  样本数: {len(cluster_data)} ({len(cluster_data)/len(features_df)*100:.1f}%)")
        print(f"  平均单笔消费: {cluster_data['transaction_amount'].mean():.2f}元")
        print(f"  消费金额范围: {cluster_data['transaction_amount'].min():.2f} - {cluster_data['transaction_amount'].max():.2f}元")
        print(f"  主要消费类目: {cluster_data['category'].mode().iloc[0] if len(cluster_data['category'].mode()) > 0 else 'N/A'}")
        
        # 统计各类目占比
        cat_dist = cluster_data['category'].value_counts(normalize=True) * 100
        print(f"  类目分布: {dict(cat_dist.head(3).round(1))}")
        
        # 工作日消费比例
        weekday_ratio = cluster_data['is_weekday'].mean() * 100
        print(f"  工作日消费比例: {weekday_ratio:.1f}%")
        
        # 白天消费比例
        daytime_ratio = cluster_data['is_daytime'].mean() * 100
        print(f"  白天消费比例: {daytime_ratio:.1f}%")
    
    return features_df, kmeans, silhouette_avg


def visualize_results(features_df, pca, kmeans):
    """
    步骤6：可视化聚类结果
    """
    print("\n" + "=" * 60)
    print("步骤6：生成可视化图表")
    print("=" * 60)
    
    # 1. 3D散点图 - 主成分空间中的聚类结果
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    markers = ['o', 's', '^', 'D']
    
    for i in range(4):
        cluster_data = features_df[features_df['cluster'] == i]
        ax.scatter(cluster_data['PC1'], cluster_data['PC2'], cluster_data['PC3'],
                   c=colors[i], marker=markers[i], s=100, alpha=0.7,
                   label=f'簇{i+1} (n={len(cluster_data)})')
    
    # 绘制聚类中心
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
               c='black', marker='*', s=300, edgecolors='white', linewidths=2,
               label='聚类中心')
    
    ax.set_xlabel('PC1 (大额消费因子)', fontsize=11)
    ax.set_ylabel('PC2 (日常消费因子)', fontsize=11)
    ax.set_zlabel('PC3 (支付习惯因子)', fontsize=11)
    ax.set_title('K-Means聚类结果 (PCA三维空间)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clustering_3d.png', dpi=300, bbox_inches='tight')
    print("✓ 3D聚类图已保存: clustering_3d.png")
    
    # 2. 各簇消费金额分布箱线图
    plt.figure(figsize=(12, 6))
    
    cluster_data_list = [features_df[features_df['cluster'] == i]['transaction_amount'].values 
                         for i in range(4)]
    
    bp = plt.boxplot(cluster_data_list, labels=[f'簇{i+1}' for i in range(4)],
                       patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    plt.xlabel('聚类簇', fontsize=12)
    plt.ylabel('消费金额 (元)', fontsize=12)
    plt.title('各聚类簇消费金额分布', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('cluster_amount_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ 消费金额分布图已保存: cluster_amount_distribution.png")
    
    # 3. 聚类结果热力图 - 各簇特征均值
    plt.figure(figsize=(12, 8))
    
    feature_cols = ['x1_amount', 'x2_food_ratio', 'x3_transport_ratio', 
                    'x4_shopping_ratio', 'x5_entertainment_ratio',
                    'x6_weekday_ratio', 'x7_daytime_ratio']
    
    cluster_means = []
    for i in range(4):
        cluster_data = features_df[features_df['cluster'] == i]
        means = cluster_data[feature_cols].mean().values
        cluster_means.append(means)
    
    cluster_means = np.array(cluster_means)
    
    # Z-score标准化热力图数据
    from scipy import stats
    cluster_means_z = stats.zscore(cluster_means, axis=0)
    
    feature_labels = ['单笔金额', '餐饮占比', '交通占比', '购物占比', 
                      '娱乐占比', '工作日占比', '白天占比']
    
    im = plt.imshow(cluster_means_z, cmap='RdYlBu_r', aspect='auto', vmin=-2, vmax=2)
    plt.colorbar(im, label='Z-score标准化值')
    
    plt.xticks(range(len(feature_labels)), feature_labels, rotation=45, ha='right')
    plt.yticks(range(4), [f'簇{i+1}' for i in range(4)])
    
    # 添加数值标注
    for i in range(4):
        for j in range(len(feature_labels)):
            text = plt.text(j, i, f'{cluster_means[i, j]:.1f}',
                           ha="center", va="center", color="black", fontsize=9)
    
    plt.title('各聚类簇特征均值热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cluster_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ 特征热力图已保存: cluster_heatmap.png")
    
    # 4. PCA方差贡献率图
    plt.figure(figsize=(10, 6))
    
    pca_full = PCA()
    pca_full.fit(features_df[['x1_amount', 'x2_food_ratio', 'x3_transport_ratio', 
                               'x4_shopping_ratio', 'x5_entertainment_ratio',
                               'x6_weekday_ratio', 'x7_daytime_ratio', 'x8_wechat_ratio']])
    
    explained_var = pca_full.explained_variance_ratio_ * 100
    cumulative_var = np.cumsum(explained_var)
    
    x = range(1, len(explained_var) + 1)
    
    plt.bar(x, explained_var, alpha=0.7, color='steelblue', label='单个方差贡献率')
    plt.plot(x, cumulative_var, 'ro-', linewidth=2, markersize=8, label='累计方差贡献率')
    
    plt.axhline(y=85, color='g', linestyle='--', alpha=0.7, label='85%阈值')
    plt.axvline(x=3, color='r', linestyle='--', alpha=0.7, label='选择3个主成分')
    
    plt.xlabel('主成分', fontsize=12)
    plt.ylabel('方差贡献率 (%)', fontsize=12)
    plt.title('PCA方差贡献率分析', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_variance.png', dpi=300, bbox_inches='tight')
    print("✓ PCA方差贡献率图已保存: pca_variance.png")


def save_results(features_df, silhouette_avg):
    """
    步骤7：保存分析结果
    """
    print("\n" + "=" * 60)
    print("步骤7：保存分析结果")
    print("=" * 60)
    
    # 保存聚类结果
    output_cols = ['transaction_id', 'transaction_time', 'transaction_amount',
                   'category', 'PC1', 'PC2', 'PC3', 'cluster']
    
    result_df = features_df[output_cols].copy()
    result_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"✓ 聚类结果已保存: {OUTPUT_FILE}")
    
    # 生成分析报告
    report = []
    report.append("=" * 60)
    report.append("消费行为数学建模分析报告")
    report.append("=" * 60)
    report.append(f"\n分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"数据样本数: {len(features_df)}")
    report.append(f"平均轮廓系数: {silhouette_avg:.3f}")
    
    report.append("\n" + "-" * 60)
    report.append("聚类结果摘要")
    report.append("-" * 60)
    
    for i in range(4):
        cluster_data = features_df[features_df['cluster'] == i]
        report.append(f"\n【簇{i+1}】")
        report.append(f"  样本数: {len(cluster_data)} ({len(cluster_data)/len(features_df)*100:.1f}%)")
        report.append(f"  平均单笔消费: {cluster_data['transaction_amount'].mean():.2f}元")
        report.append(f"  消费金额中位数: {cluster_data['transaction_amount'].median():.2f}元")
        report.append(f"  主要消费类目: {cluster_data['category'].mode().iloc[0] if len(cluster_data['category'].mode()) > 0 else 'N/A'}")
    
    report.append("\n" + "=" * 60)
    
    report_text = '\n'.join(report)
    
    with open('clustering_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ 分析报告已保存: clustering_report.txt")
    print("\n分析结果文件:")
    print(f"  1. {OUTPUT_FILE} - 聚类结果数据")
    print(f"  2. clustering_report.txt - 文字分析报告")
    print(f"  3. elbow_method.png - 肘部法则图")
    print(f"  4. clustering_3d.png - 3D聚类可视化")
    print(f"  5. cluster_amount_distribution.png - 消费金额分布")
    print(f"  6. cluster_heatmap.png - 特征热力图")
    print(f"  7. pca_variance.png - PCA方差贡献率")


def main():
    """
    主函数：执行完整的消费行为建模分析
    """
    print("\n" + "=" * 60)
    print("消费行为数学建模分析工具")
    print("=" * 60)
    print("模型: PCA降维 + K-Means聚类")
    print("=" * 60)
    
    try:
        # 步骤1：加载数据
        df = load_data(INPUT_FILE)
        
        # 步骤2：提取特征
        features_df = extract_features(df)
        
        # 步骤3：PCA分析
        features_df, pca, scaler = pca_analysis(features_df)
        
        # 步骤4：肘部法则
        optimal_k, sse = elbow_method(features_df)
        
        # 步骤5：K-Means聚类
        features_df, kmeans, silhouette_avg = kmeans_clustering(features_df, optimal_k)
        
        # 步骤6：可视化
        visualize_results(features_df, pca, kmeans)
        
        # 步骤7：保存结果
        save_results(features_df, silhouette_avg)
        
        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)
        
        return features_df
        
    except Exception as e:
        print(f"\n✗ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    result_df = main()
