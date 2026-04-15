import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 读取预处理后的数据
def load_cleaned_data():
    train_data = pd.read_csv('train_cleaned.csv')
    test_data = pd.read_csv('test_cleaned.csv')
    return train_data, test_data

# 提取TF-IDF特征
def extract_features():
    print("Loading cleaned data...")
    train_data, test_data = load_cleaned_data()
    
    print("Extracting TF-IDF features...")
    # 初始化TF-IDF向量化器，增加特征数并添加n-gram
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    
    # 拟合并转换训练数据
    X_train = vectorizer.fit_transform(train_data['cleaned_review'])
    
    # 转换测试数据
    X_test = vectorizer.transform(test_data['cleaned_review'])
    
    # 获取标签
    y_train = train_data['sentiment']
    
    print(f"Train features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    return X_train, y_train, X_test, test_data

if __name__ == "__main__":
    extract_features()
