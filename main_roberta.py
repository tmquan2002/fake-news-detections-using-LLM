import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Tạo thư mục lưu ảnh nếu chưa có
if not os.path.exists('images'):
    os.makedirs('images')

print("⏳ Đang tải dataset và model (lần đầu sẽ hơi lâu)...")

# 1. Load Data
dataset = load_dataset("gonzaloa/fake_news", split="validation[:200]")
df = pd.DataFrame(dataset)
df['label_name'] = df['label'].map({0: "Fake", 1: "Real"})

# 2. Load Model
model_name = "hamzab/roberta-fake-news-classification"
classifier = pipeline("text-classification", model=model_name, tokenizer=model_name, truncation=True, max_length=512)

# 3. Inference
print("🚀 Đang chạy dự đoán...")
predictions = []
true_labels = df['label'].tolist()

for text in df['text'].tolist():
    output = classifier(text[:512])[0]
    label = 0 if 'FAKE' in output['label'].upper() else 1
    predictions.append(label)

# 4. Evaluation
acc = accuracy_score(true_labels, predictions)
print(f"\n🏆 Accuracy: {acc * 100:.2f}%")
print(classification_report(true_labels, predictions, target_names=['Fake', 'Real']))

# 5. Save Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Fake', 'Pred Real'], yticklabels=['Actual Fake', 'Actual Real'])
plt.title('Confusion Matrix')
plt.savefig('images/confusion_matrix.png') # Lưu ảnh thay vì chỉ show
print("✅ Đã lưu biểu đồ vào thư mục images/")