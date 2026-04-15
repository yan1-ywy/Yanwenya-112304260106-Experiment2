import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取原始训练数据
def load_raw_data():
    train_data = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t', quoting=3)
    test_data = pd.read_csv('testData.tsv/testData.tsv', sep='\t', quoting=3)
    return train_data, test_data

# 数据探索
def explore_data():
    print("Loading raw data...")
    train_data, test_data = load_raw_data()
    
    print("\n=== 训练数据基本信息 ===")
    print(f"训练数据形状: {train_data.shape}")
    print("\n训练数据前5行:")
    print(train_data.head())
    
    print("\n=== 测试数据基本信息 ===")
    print(f"测试数据形状: {test_data.shape}")
    print("\n测试数据前5行:")
    print(test_data.head())
    
    print("\n=== 情感分布 ===")
    sentiment_dist = train_data['sentiment'].value_counts()
    print(sentiment_dist)
    
    # 绘制情感分布饼图
    plt.figure(figsize=(6, 6))
    plt.pie(sentiment_dist, labels=['正面情感', '负面情感'], autopct='%1.1f%%', startangle=90)
    plt.title('情感分布')
    plt.savefig('sentiment_distribution.png')
    print("\n情感分布饼图已保存为 sentiment_distribution.png")
    
    # 分析评论长度
    train_data['review_length'] = train_data['review'].apply(len)
    print("\n=== 评论长度统计 ===")
    print(train_data['review_length'].describe())
    
    # 绘制评论长度分布
    plt.figure(figsize=(10, 6))
    sns.histplot(train_data['review_length'], bins=50, kde=True)
    plt.title('评论长度分布')
    plt.xlabel('评论长度')
    plt.ylabel('频数')
    plt.savefig('review_length_distribution.png')
    print("评论长度分布图已保存为 review_length_distribution.png")
    
    return train_data, test_data

if __name__ == "__main__":
    explore_data()
