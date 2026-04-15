import pandas as pd
import re
import string

# NLTK的标准停用词列表（完整）
STOP_WORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

# 否定词列表（需要保留）
NEGATION_WORDS = set(['not', 'no', 'never', 'nor', 'neither'])

# 读取训练数据
def load_train_data():
    train_data = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t', quoting=3)
    return train_data

# 读取测试数据
def load_test_data():
    test_data = pd.read_csv('testData.tsv/testData.tsv', sep='\t', quoting=3)
    return test_data

# 简单的词干提取函数
def simple_stemmer(word):
    # 简单的词干提取规则
    if word.endswith('ing'):
        return word[:-3]
    elif word.endswith('ed'):
        return word[:-2]
    elif word.endswith('es'):
        return word[:-2]
    elif word.endswith('s') and len(word) > 3:
        return word[:-1]
    return word

# 清洗文本
def clean_text(text):
    # 1. 移除HTML标签
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 2. 小写化
    text = text.lower()
    
    # 3. 标点处理（保留情感表达和否定形式）
    # 替换连续的感叹号和问号为单个
    text = re.sub(r'!+', '!', text)
    text = re.sub(r'\?+', '?', text)
    # 保留重要标点，移除其他标点
    text = re.sub(r'[^a-zA-Z0-9\s!\?\']', ' ', text)
    
    # 4. 分词
    words = text.split()
    
    # 5. 处理停用词（保留否定词）
    words = [w for w in words if not (w in STOP_WORDS and w not in NEGATION_WORDS)]
    
    # 6. 词干提取
    words = [simple_stemmer(w) for w in words]
    
    # 7. 处理否定词短语
    # 将否定词与后续单词组合，如 "not good" 变为 "not_good"
    processed_words = []
    i = 0
    while i < len(words):
        if words[i] in NEGATION_WORDS and i + 1 < len(words):
            processed_words.append(f"{words[i]}_{words[i+1]}")
            i += 2
        else:
            processed_words.append(words[i])
            i += 1
    
    # 重新组合为文本
    return ' '.join(processed_words)

# 预处理数据
def preprocess_data():
    print("Loading training data...")
    train_data = load_train_data()
    print(f"Training data shape: {train_data.shape}")
    
    print("Loading test data...")
    test_data = load_test_data()
    print(f"Test data shape: {test_data.shape}")
    
    print("Cleaning training data...")
    train_data['cleaned_review'] = train_data['review'].apply(clean_text)
    
    print("Cleaning test data...")
    test_data['cleaned_review'] = test_data['review'].apply(clean_text)
    
    # 保存处理后的数据
    train_data.to_csv('train_cleaned.csv', index=False)
    test_data.to_csv('test_cleaned.csv', index=False)
    
    print("Preprocessing completed!")
    return train_data, test_data

if __name__ == "__main__":
    preprocess_data()
