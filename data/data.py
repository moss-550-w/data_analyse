"""
微信支付账单数据预处理脚本
功能：根据数据清洗要求对微信账单进行数据清洗、脱敏、异常值检测和特征工程
"""

import pandas as pd
import numpy as np
import hashlib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 配置文件路径（相对路径，相对于data.py所在目录）
INPUT_FILE = r'..\微信支付账单流水文件(20260130-20260430)_20260430132019.xlsx'
OUTPUT_FILE = 'cleaned_wechat_data.csv'
INTERMEDIATE_FILE = 'intermediate_data_for_review.csv'  # 中间数据，用于人工判断

# 微信账单原始列名映射（根据微信账单标准格式）
COLUMN_MAPPING = {
    '交易时间': 'transaction_time',
    '交易类型': 'transaction_type_raw',
    '交易对方': 'counterparty',
    '商品': 'product',
    '收/支': 'transaction_direction',
    '金额(元)': 'transaction_amount',
    '支付方式': 'payment_method_raw',
    '当前状态': 'transaction_status_raw',
    '交易单号': 'transaction_id_raw',
    '商户单号': 'merchant_id'
}

# 消费类目映射规则（将商户名称映射为行业类目）
CATEGORY_MAPPING = {
    # 餐饮美食
    '肯德基': '餐饮美食',
    '麦当劳': '餐饮美食',
    '星巴克': '餐饮美食',
    '必胜客': '餐饮美食',
    '海底捞': '餐饮美食',
    '餐厅': '餐饮美食',
    '饭店': '餐饮美食',
    '火锅': '餐饮美食',
    '烧烤': '餐饮美食',
    '奶茶': '餐饮美食',
    '咖啡': '餐饮美食',
    '快餐': '餐饮美食',
    '面馆': '餐饮美食',
    '寿司': '餐饮美食',
    '披萨': '餐饮美食',
    '汉堡': '餐饮美食',
    '外卖': '餐饮美食',
    '美团': '餐饮美食',
    '饿了么': '餐饮美食',
    
    # 超市购物
    '超市': '超市购物',
    '沃尔玛': '超市购物',
    '家乐福': '超市购物',
    '永辉': '超市购物',
    '大润发': '超市购物',
    '盒马': '超市购物',
    '山姆': '超市购物',
    '便利店': '超市购物',
    '全家': '超市购物',
    '罗森': '超市购物',
    '7-11': '超市购物',
    '华润万家': '超市购物',
    '物美': '超市购物',
    
    # 交通出行
    '滴滴': '交通出行',
    '打车': '交通出行',
    '出租车': '交通出行',
    '地铁': '交通出行',
    '公交': '交通出行',
    '高铁': '交通出行',
    '火车': '交通出行',
    '机票': '交通出行',
    '加油站': '交通出行',
    '停车': '交通出行',
    '高速': '交通出行',
    'ETC': '交通出行',
    '共享单车': '交通出行',
    '哈啰': '交通出行',
    '摩拜': '交通出行',
    
    # 日用百货
    '京东': '日用百货',
    '淘宝': '日用百货',
    '天猫': '日用百货',
    '拼多多': '日用百货',
    '唯品会': '日用百货',
    '苏宁易购': '日用百货',
    '国美': '日用百货',
    '无印良品': '日用百货',
    '名创优品': '日用百货',
    
    # 休闲娱乐
    '电影': '休闲娱乐',
    '影院': '休闲娱乐',
    'KTV': '休闲娱乐',
    '网吧': '休闲娱乐',
    '游戏': '休闲娱乐',
    '腾讯': '休闲娱乐',
    '爱奇艺': '休闲娱乐',
    '优酷': '休闲娱乐',
    'B站': '休闲娱乐',
    '哔哩哔哩': '休闲娱乐',
    '网易云': '休闲娱乐',
    'QQ音乐': '休闲娱乐',
    '视频会员': '休闲娱乐',
    '音乐': '休闲娱乐',
    '演出': '休闲娱乐',
    '门票': '休闲娱乐',
    
    # 医疗健康
    '医院': '医疗健康',
    '药店': '医疗健康',
    '诊所': '医疗健康',
    '体检': '医疗健康',
    '医保': '医疗健康',
    
    # 通讯费用
    '话费': '通讯费用',
    '流量': '通讯费用',
    '移动': '通讯费用',
    '联通': '通讯费用',
    '电信': '通讯费用',
    '宽带': '通讯费用',
    
    # 服饰美容
    '服装': '服饰美容',
    '鞋': '服饰美容',
    '化妆品': '服饰美容',
    '美容': '服饰美容',
    '美发': '服饰美容',
    '美甲': '服饰美容',
    '优衣库': '服饰美容',
    'ZARA': '服饰美容',
    'H&M': '服饰美容',
    '耐克': '服饰美容',
    '阿迪达斯': '服饰美容',
    
    # 居住缴费
    '房租': '居住缴费',
    '物业': '居住缴费',
    '水电': '居住缴费',
    '燃气': '居住缴费',
    '供暖': '居住缴费',
    
    # 学习办公
    '书籍': '学习办公',
    '文具': '学习办公',
    '打印': '学习办公',
    '复印': '学习办公',
    '当当': '学习办公',
    '课程': '学习办公',
    '培训': '学习办公',
    
    # 人情社交
    '红包': '人情社交',
    '转账': '人情社交',
    '礼品': '人情社交',
    '鲜花': '人情社交'
}

# 必要消费类目定义
NECESSARY_CATEGORIES = ['餐饮美食', '交通出行', '日用百货', '医疗健康']

# 节假日列表（2026年）
HOLIDAYS_2026 = [
    '2026-01-01',  # 元旦
    '2026-02-17', '2026-02-18', '2026-02-19', '2026-02-20', '2026-02-21', '2026-02-22',  # 春节
    '2026-04-04', '2026-04-05', '2026-04-06',  # 清明节
    '2026-05-01', '2026-05-02', '2026-05-03',  # 劳动节
    '2026-06-19', '2026-06-20', '2026-06-21',  # 端午节
    '2026-09-25', '2026-09-26', '2026-09-27',  # 中秋节
    '2026-10-01', '2026-10-02', '2026-10-03', '2026-10-04', '2026-10-05', '2026-10-06', '2026-10-07',  # 国庆节
]


def load_data(file_path):
    """
    步骤1：读取微信账单Excel文件
    微信账单格式：前17行为表头信息，第18行为列名，第19行开始为数据
    """
    print("=" * 60)
    print("步骤1：读取原始数据")
    print("=" * 60)
    
    try:
        # 读取Excel文件，跳过前17行（微信账单表头信息），第18行作为列名
        df = pd.read_excel(file_path, skiprows=17)
        print(f"✓ 成功读取数据，共 {len(df)} 行")
        print(f"原始列名: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"✗ 读取文件失败: {e}")
        raise


def standardize_columns(df):
    """
    标准化列名
    """
    print("\n" + "=" * 60)
    print("步骤2：标准化列名和格式")
    print("=" * 60)
    
    # 重命名列
    df = df.rename(columns=COLUMN_MAPPING)
    
    # 确保必要列存在
    required_cols = ['transaction_time', 'transaction_amount', 'transaction_direction']
    for col in required_cols:
        if col not in df.columns:
            print(f"⚠ 警告：缺少必要列 {col}")
    
    print(f"✓ 列名标准化完成")
    print(f"标准化后列名: {list(df.columns)}")
    return df


def filter_and_clean(df):
    """
    步骤3：筛选支出记录并清洗数据
    """
    print("\n" + "=" * 60)
    print("步骤3：筛选支出记录并清洗")
    print("=" * 60)
    
    initial_count = len(df)
    
    # 3.1 转换时间格式
    if 'transaction_time' in df.columns:
        df['transaction_time'] = pd.to_datetime(df['transaction_time'], errors='coerce')
        # 删除时间缺失的记录
        df = df.dropna(subset=['transaction_time'])
        print(f"✓ 时间格式转换完成")
    
    # 3.2 转换金额格式
    if 'transaction_amount' in df.columns:
        # 处理金额列，移除人民币符号和逗号
        df['transaction_amount'] = df['transaction_amount'].astype(str).str.replace('¥', '').str.replace(',', '')
        df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce')
        # 删除金额缺失的记录
        df = df.dropna(subset=['transaction_amount'])
        print(f"✓ 金额格式转换完成")
    
    # 3.3 仅保留支出记录
    if 'transaction_direction' in df.columns:
        df = df[df['transaction_direction'] == '支出']
        print(f"✓ 已筛选出支出记录")
    
    # 3.4 仅保留成功交易
    if 'transaction_status_raw' in df.columns:
        df = df[df['transaction_status_raw'] == '支付成功']
        print(f"✓ 已筛选出成功交易")
    
    # 3.5 删除资金划转类交易
    exclude_keywords = ['转账', '红包', '提现', '充值', '退款', '信用卡还款', '理财']
    if 'transaction_type_raw' in df.columns:
        for keyword in exclude_keywords:
            df = df[~df['transaction_type_raw'].str.contains(keyword, na=False)]
    print(f"✓ 已删除资金划转类交易")
    
    # 3.6 根据交易单号去重
    if 'transaction_id_raw' in df.columns:
        df = df.drop_duplicates(subset=['transaction_id_raw'])
        print(f"✓ 已根据交易单号去重")
    
    final_count = len(df)
    print(f"✓ 筛选完成：{initial_count} → {final_count} 条记录")
    
    return df


def anonymize_data(df):
    """
    步骤4：数据脱敏处理
    """
    print("\n" + "=" * 60)
    print("步骤4：数据脱敏处理")
    print("=" * 60)
    
    # 4.1 删除敏感字段
    sensitive_cols = ['counterparty', 'merchant_id']
    for col in sensitive_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"✓ 已删除敏感字段: {col}")
    
    # 4.2 对交易单号进行MD5哈希
    if 'transaction_id_raw' in df.columns:
        df['transaction_id'] = df['transaction_id_raw'].apply(
            lambda x: hashlib.md5(str(x).encode()).hexdigest()[:16].upper()
        )
        df = df.drop(columns=['transaction_id_raw'])
        print(f"✓ 交易单号已进行MD5哈希")
    
    # 4.3 商户名称模糊化为类目
    def map_category(product, trans_type):
        """根据商品描述和交易类型映射消费类目"""
        if pd.isna(product):
            return '其他'
        
        product_str = str(product)
        
        # 根据关键词匹配
        for keyword, category in CATEGORY_MAPPING.items():
            if keyword in product_str:
                return category
        
        # 根据交易类型推断
        if trans_type and '外卖' in str(trans_type):
            return '餐饮美食'
        if trans_type and '滴滴' in str(trans_type):
            return '交通出行'
        
        return '其他'
    
    df['category'] = df.apply(
        lambda row: map_category(row.get('product'), row.get('transaction_type_raw')),
        axis=1
    )
    print(f"✓ 商户名称已映射为消费类目")
    
    # 删除原始商品字段
    if 'product' in df.columns:
        df = df.drop(columns=['product'])
    if 'transaction_type_raw' in df.columns:
        df = df.drop(columns=['transaction_type_raw'])
    
    return df


def handle_missing_values(df):
    """
    步骤5：缺失值处理
    """
    print("\n" + "=" * 60)
    print("步骤5：缺失值处理")
    print("=" * 60)
    
    initial_count = len(df)
    
    # 5.1 删除时间和金额缺失的记录
    df = df.dropna(subset=['transaction_time', 'transaction_amount'])
    
    # 5.2 对缺失的支付方式标记为"其他"
    if 'payment_method_raw' in df.columns:
        df['payment_method_raw'] = df['payment_method_raw'].fillna('其他')
    
    # 5.3 根据时间和金额推断缺失的类目
    def infer_category(row):
        if pd.notna(row.get('category')) and row['category'] != '其他':
            return row['category']
        
        hour = row['transaction_time'].hour
        amount = row['transaction_amount']
        weekday = row['transaction_time'].weekday()
        
        # 工作日午餐时段，金额15-50元 → 餐饮美食
        if weekday < 5 and 11 <= hour <= 14 and 15 <= amount <= 50:
            return '餐饮美食'
        
        # 工作日晚餐时段，金额20-100元 → 餐饮美食
        if weekday < 5 and 17 <= hour <= 20 and 20 <= amount <= 100:
            return '餐饮美食'
        
        # 周末晚上，金额50-300元 → 休闲娱乐
        if weekday >= 5 and 18 <= hour <= 23 and 50 <= amount <= 300:
            return '休闲娱乐'
        
        # 小额消费 → 超市购物
        if amount < 30:
            return '超市购物'
        
        return '其他'
    
    df['category'] = df.apply(infer_category, axis=1)
    print(f"✓ 缺失类目已根据规则推断补全")
    
    final_count = len(df)
    print(f"✓ 缺失值处理完成：{initial_count} → {final_count} 条记录")
    
    return df


def detect_outliers(df):
    """
    步骤6：异常值检测与处理
    使用IQR四分位法和3σ原则
    """
    print("\n" + "=" * 60)
    print("步骤6：异常值检测与处理")
    print("=" * 60)
    
    amounts = df['transaction_amount']
    
    # 方法1：IQR四分位法
    Q1 = amounts.quantile(0.25)
    Q3 = amounts.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound_iqr = Q1 - 1.5 * IQR
    upper_bound_iqr = Q3 + 1.5 * IQR
    
    print(f"\n【IQR四分位法】")
    print(f"  Q1 = {Q1:.2f}元")
    print(f"  Q3 = {Q3:.2f}元")
    print(f"  IQR = {IQR:.2f}元")
    print(f"  上限 = {upper_bound_iqr:.2f}元")
    print(f"  下限 = {lower_bound_iqr:.2f}元")
    
    outliers_iqr = df[(amounts < lower_bound_iqr) | (amounts > upper_bound_iqr)]
    print(f"  检测出异常值: {len(outliers_iqr)} 个")
    
    # 方法2：3σ原则
    mean = amounts.mean()
    std = amounts.std()
    lower_bound_3sigma = mean - 3 * std
    upper_bound_3sigma = mean + 3 * std
    
    print(f"\n【3σ原则】")
    print(f"  均值 = {mean:.2f}元")
    print(f"  标准差 = {std:.2f}元")
    print(f"  上限 = {upper_bound_3sigma:.2f}元")
    print(f"  下限 = {lower_bound_3sigma:.2f}元")
    
    outliers_3sigma = df[(amounts < lower_bound_3sigma) | (amounts > upper_bound_3sigma)]
    print(f"  检测出异常值: {len(outliers_3sigma)} 个")
    
    # 异常值处理策略
    # 1. 删除明显错误的数据（负数或金额大于10000）
    df_clean = df[(amounts > 0) & (amounts <= 10000)].copy()
    removed_extreme = len(df) - len(df_clean)
    print(f"\n✓ 已删除明显错误数据（负数或>10000元）: {removed_extreme} 条")
    
    # 2. 对于IQR检测出的异常值，保留但标记
    df_clean['is_outlier_iqr'] = (df_clean['transaction_amount'] > upper_bound_iqr)
    df_clean['is_outlier_3sigma'] = (df_clean['transaction_amount'] > upper_bound_3sigma)
    
    outlier_count = df_clean['is_outlier_iqr'].sum()
    print(f"✓ 已标记大额交易（IQR异常）: {outlier_count} 条")
    
    return df_clean


def feature_engineering(df):
    """
    步骤7：特征工程
    """
    print("\n" + "=" * 60)
    print("步骤7：特征工程")
    print("=" * 60)
    
    # 7.1 标准化支付方式
    payment_mapping = {
        '零钱': '微信零钱',
        '零钱通': '微信零钱通',
        '招商银行': '招商银行借记卡',
        '工商银行': '工商银行借记卡',
        '建设银行': '建设银行借记卡',
        '农业银行': '农业银行借记卡',
        '中国银行': '中国银行借记卡',
        '交通银行': '交通银行借记卡',
        '邮储银行': '邮储银行借记卡',
        '信用卡': '信用卡'
    }
    
    def standardize_payment_method(method):
        if pd.isna(method):
            return '其他'
        method_str = str(method)
        for key, value in payment_mapping.items():
            if key in method_str:
                return value
        return '其他'
    
    df['payment_method'] = df['payment_method_raw'].apply(standardize_payment_method)
    if 'payment_method_raw' in df.columns:
        df = df.drop(columns=['payment_method_raw'])
    print(f"✓ 支付方式已标准化")
    
    # 7.2 交易状态标准化
    df['transaction_status'] = '成功'
    if 'transaction_status_raw' in df.columns:
        df = df.drop(columns=['transaction_status_raw'])
    
    # 7.3 时间特征提取
    df['week_day'] = df['transaction_time'].dt.day_name()
    df['week_day_cn'] = df['transaction_time'].dt.dayofweek.map({
        0: '星期一', 1: '星期二', 2: '星期三', 3: '星期四',
        4: '星期五', 5: '星期六', 6: '星期日'
    })
    
    # 7.4 消费时段划分
    def get_time_period(hour):
        if 6 <= hour < 9:
            return '早餐时段(06:00-09:00)'
        elif 9 <= hour < 11:
            return '上午时段(09:00-11:00)'
        elif 11 <= hour < 14:
            return '午餐时段(11:00-14:00)'
        elif 14 <= hour < 17:
            return '下午时段(14:00-17:00)'
        elif 17 <= hour < 21:
            return '晚餐时段(17:00-21:00)'
        else:
            return '深夜时段(21:00-06:00)'
    
    df['time_period'] = df['transaction_time'].dt.hour.apply(get_time_period)
    print(f"✓ 时间特征已提取")
    
    # 7.5 节假日特征
    df['date_str'] = df['transaction_time'].dt.strftime('%Y-%m-%d')
    df['is_holiday'] = df['date_str'].isin(HOLIDAYS_2026)
    df = df.drop(columns=['date_str'])
    print(f"✓ 节假日特征已生成")
    
    # 7.6 消费属性特征（是否必要消费）
    df['is_necessary'] = df['category'].isin(NECESSARY_CATEGORIES)
    print(f"✓ 消费属性特征已生成")
    
    # 7.7 交易类型
    df['transaction_type'] = '支出'
    
    return df


def reorder_columns(df):
    """
    重新排列列顺序，使其符合报告要求
    """
    column_order = [
        'transaction_id',
        'transaction_time',
        'transaction_amount',
        'transaction_type',
        'category',
        'payment_method',
        'transaction_status',
        'week_day_cn',
        'time_period',
        'is_holiday',
        'is_necessary',
        'is_outlier_iqr',
        'is_outlier_3sigma'
    ]
    
    # 只保留存在的列
    existing_cols = [col for col in column_order if col in df.columns]
    df = df[existing_cols]
    
    return df


def save_intermediate_data(df, output_file):
    """
    保存中间数据用于人工判断
    包含异常值标记，供人工审核确认
    """
    print("\n" + "=" * 60)
    print("步骤6.5：保存中间数据（用于人工判断）")
    print("=" * 60)
    
    # 选择用于人工判断的关键列
    review_columns = ['transaction_id', 'transaction_time', 'transaction_amount', 
                      'transaction_type_raw', 'product', 'category',
                      'is_outlier_iqr', 'is_outlier_3sigma']
    
    # 只保留存在的列
    existing_cols = [col for col in review_columns if col in df.columns]
    df_review = df[existing_cols].copy()
    
    # 添加人工判断列（空白，供用户填写）
    df_review['人工审核结果'] = ''
    df_review['备注'] = ''
    
    # 突出显示异常值记录
    outlier_mask = df_review['is_outlier_iqr'] | df_review['is_outlier_3sigma']
    df_review.loc[outlier_mask, '人工审核结果'] = '待审核'
    
    df_review.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✓ 中间数据已保存至: {output_file}")
    print(f"✓ 共 {len(df_review)} 条记录，其中 {outlier_mask.sum()} 条标记为异常值")
    print(f"  请在Excel中打开此文件，对'待审核'记录进行人工判断")
    print(f"  审核完成后可修改'人工审核结果'列：保留/删除")


def save_results(df, output_file):
    """
    保存清洗后的最终数据
    """
    print("\n" + "=" * 60)
    print("步骤8：保存结果")
    print("=" * 60)
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✓ 数据已保存至: {output_file}")
    print(f"✓ 最终数据集包含 {len(df)} 条记录")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("数据统计摘要")
    print("=" * 60)
    print(f"总记录数: {len(df)}")
    print(f"总消费金额: {df['transaction_amount'].sum():.2f}元")
    print(f"平均消费金额: {df['transaction_amount'].mean():.2f}元")
    print(f"中位数消费金额: {df['transaction_amount'].median():.2f}元")
    print(f"\n消费类目分布:")
    print(df['category'].value_counts())
    print(f"\n支付方式分布:")
    print(df['payment_method'].value_counts())
    print(f"\n消费时段分布:")
    print(df['time_period'].value_counts())
    print(f"\n必要消费占比: {df['is_necessary'].mean()*100:.1f}%")
    print(f"节假日消费占比: {df['is_holiday'].mean()*100:.1f}%")


def main():
    """
    主函数：执行完整的数据清洗流程
    """
    print("\n" + "=" * 60)
    print("微信支付账单数据清洗工具")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 步骤1：读取数据
        df = load_data(INPUT_FILE)
        
        # 步骤2：标准化列名
        df = standardize_columns(df)
        
        # 步骤3：筛选和清洗
        df = filter_and_clean(df)
        
        # 步骤4：脱敏处理
        df = anonymize_data(df)
        
        # 步骤5：缺失值处理
        df = handle_missing_values(df)
        
        # 步骤6：异常值检测
        df = detect_outliers(df)
        
        # 步骤6.5：保存中间数据用于人工判断（包含原始商户信息供人工审核）
        save_intermediate_data(df, INTERMEDIATE_FILE)
        
        # 步骤7：特征工程
        df = feature_engineering(df)
        
        # 步骤8：重新排列列
        df = reorder_columns(df)
        
        # 步骤9：保存最终结果
        save_results(df, OUTPUT_FILE)
        
        print("\n" + "=" * 60)
        print("数据清洗完成！")
        print("=" * 60)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return df
        
    except Exception as e:
        print(f"\n✗ 数据处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    cleaned_data = main()
