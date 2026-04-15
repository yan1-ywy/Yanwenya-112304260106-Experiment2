import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from feature_extraction import extract_features

# 训练模型
def train_model():
    print("Extracting features...")
    X_train, y_train, X_test, test_data = extract_features()
    
    # 分割训练集和验证集
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print("Training model...")
    # 初始化逻辑回归模型
    model = LogisticRegression(max_iter=2000, C=2.0, penalty='l2')
    
    # 训练模型
    model.fit(X_train_split, y_train_split)
    
    # 在验证集上评估模型
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    
    accuracy = accuracy_score(y_val, y_val_pred)
    auc = roc_auc_score(y_val, y_val_prob)
    
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation AUC: {auc:.4f}")
    print("Classification report:")
    print(classification_report(y_val, y_val_pred))
    
    # 在测试集上预测
    print("Predicting on test data...")
    y_test_pred = model.predict(X_test)
    
    # 生成提交文件
    submission = pd.DataFrame({
        'id': test_data['id'],
        'sentiment': y_test_pred
    })
    submission.to_csv('submission.csv', index=False, quoting=3)
    print("Submission file generated: submission.csv")
    
    return model, accuracy, auc

if __name__ == "__main__":
    train_model()
