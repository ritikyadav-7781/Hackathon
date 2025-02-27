import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import tensorflow as tf
import nltk
import string
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from transformers import DistilBertTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model
model = tf.keras.models.load_model('sarcasm_model.h5')  # Replace with your model path

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    emoji = re.compile("["
      u"\U0001F600-\U0001FFFF"
      u"\U0001F300-\U0001F5FF"
      u"\U0001F680-\U0001F6FF"
      u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251" "]+", flags=re.UNICODE)
    return text

# Function to tokenize sentences
def tokenize_sentences(sentences, max_length=25):
    tokens = tokenizer(
        sentences,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )
    return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}

# Function to generate predictions
def generate_predictions(df):
    # Clean text
    df['Tweet'] = df['Tweet'].apply(clean_text)
    
    # Tokenize sentences
    tokenized_data = tokenize_sentences(df['Tweet'].values.tolist())
    
    # Generate predictions
    predictions = model.predict([tokenized_data['input_ids'], tokenized_data['attention_mask']])
    predicted_classes = (predictions > 0.5).astype(int)
    
    return predicted_classes

# Function to evaluate predictions
def evaluate_predictions(true_labels, predicted_labels):
    # Classification report
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Main ETL function
def etl_pipeline(input_file):
    # Step 1: Extract - Load the dataset
    df = pd.read_csv(input_file)
    
    # Step 2: Transform - Generate predictions
    predictions = generate_predictions(df)
    
    # Step 3: Load - Save predictions
    df['Predicted_Label'] = predictions
    df.to_csv('predictions.csv', index=False)
    
    # Step 4: Evaluate - Generate classification report and confusion matrix
    if 'Label' in df.columns:
        evaluate_predictions(df['Label'], predictions)
    else:
        print("No ground truth labels provided. Skipping evaluation.")

# Run the pipeline
if __name__ == "__main__":
    input_file = "test.csv"  # Replace with your input file path
    etl_pipeline(input_file)