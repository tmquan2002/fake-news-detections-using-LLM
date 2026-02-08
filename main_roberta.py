import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Configure ---
# Create images folder to save results
if not os.path.exists('images'):
    os.makedirs('images')

# --- STEP 1: LOAD DATA & EDA ---
print("Loading datasets...")

# Get 200 samples from hugging face databases
dataset = load_dataset("gonzaloa/fake_news", split="validation[:200]")
df = pd.DataFrame(dataset)

# Map labels 0/1 as Fake/Real
df['label_name'] = df['label'].map({0: "Fake", 1: "Real"})

print(f"{len(df)} samples uploaded.")

# === Bar Chart showing number of fake news and real news ===
print("Drawing EDA chart...")
plt.figure(figsize=(6, 4))
sns.countplot(x='label_name', data=df, palette='viridis', hue='label_name', legend=False)
plt.title('Data Distribution (Experimental Subset)')
plt.xlabel('Label')
plt.ylabel('Count')
plt.savefig('images/data_distribution.png')
print("Chart saved in: images/data_distribution.png")
plt.close()
# ===============================================

# --- STEP 2: LOAD MODEL ROBERTA ---
model_name = "hamzab/roberta-fake-news-classification"
print(f"\n Loading model RoBERTa: {model_name}...")
classifier = pipeline("text-classification", model=model_name, tokenizer=model_name, truncation=True, max_length=512)
print("Model ready!")


# --- STEP 3: INFERENCE ---
print("\n Intererence 200 samples...")
predictions = []
true_labels = df['label'].tolist()

# Loop each data in list
for text in df['text'].tolist():
    # Cut text into 512 tokens and put into model
    output = classifier(text[:512])[0]
    # Turn label from real/fake back to 0/1 for checking
    label = 0 if 'FAKE' in output['label'].upper() else 1
    predictions.append(label)


# --- STEP 4: Researching & Save Confusion Matrix ---
print("\n" + "="*50)
print("RESULT")
print("="*50)

# Accuracy calculation
acc = accuracy_score(true_labels, predictions)
print(f"Accuracy: {acc * 100:.2f}%")
print("\n📄 Classification Report:")
print(classification_report(true_labels, predictions, target_names=['Fake', 'Real']))

# Draw and save Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Fake', 'Pred Real'],
            yticklabels=['Actual Fake', 'Actual Real'])
plt.title('Confusion Matrix (RoBERTa)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Lưu ảnh
plt.savefig('images/confusion_matrix.png')
print("\n Confusion Matrix saved in: images/confusion_matrix.png")
print("="*50)