import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
import tensorflow as tf
import pickle
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# --- Clase personalizada para mostrar progreso ---
class TrainingProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Época {epoch + 1} - Precisión: {logs.get('accuracy'):.4f} | Pérdida: {logs.get('loss'):.4f} | Val. Precisión: {logs.get('val_accuracy'):.4f} | Val. Pérdida: {logs.get('val_loss'):.4f}")

# --- Funciones de procesamiento ---
def load_emails(directory, label):
    emails = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if not os.path.isfile(filepath) or filename.startswith('.'):
            continue
            
        try:
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    with open(filepath, 'r', encoding=encoding) as file:
                        content = file.read()
                        emails.append({'text': content, 'label': label})
                    break
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            print(f"Error al procesar {filename}: {str(e)}")
    
    return pd.DataFrame(emails)

def clean_email(text):
    text = re.sub(r'^.?(From:|To:|Subject:|Date:).?\n', '', text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r'^\s*--.*?$', '', text, flags=re.DOTALL|re.MULTILINE)
    text = re.sub(r'http\S+|www\S+|https\S+|<[^>]+>', '', text)
    text = re.sub(r'[^\w\sáéíóúñ]', ' ', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

# --- Funciones del modelo ---
def save_model(model, tokenizer, maxlen):
    model.save('spam_model.h5')
    with open('tokenizer.pkl', 'wb') as handle:
        pickle.dump((tokenizer, maxlen), handle)

def load_saved_model():
    if not (os.path.exists('spam_model.h5') and os.path.exists('tokenizer.pkl')):
        return None, None, None
    
    model = load_model('spam_model.h5')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer, maxlen = pickle.load(handle)
    return model, tokenizer, maxlen

def train_model(ham_dir, spam_dir):
    print("\n=== CARGANDO DATOS ===")
    df_spam = load_emails(spam_dir, 1)
    df_ham = load_emails(ham_dir, 0)

    if len(df_spam) == 0 or len(df_ham) == 0:
        raise ValueError("No se cargaron archivos. Verifica las rutas.")

    print(f"\nDatos cargados: {len(df_ham)} HAM | {len(df_spam)} SPAM")
    
    df = pd.concat([df_spam, df_ham]).sample(frac=1, random_state=42)
    df['text'] = df['text'].apply(clean_email)

    tokenizer = Tokenizer(num_words=8000, oov_token='<OOV>', filters='')
    tokenizer.fit_on_texts(df['text'])
    X = tokenizer.texts_to_sequences(df['text'])

    lengths = [len(seq) for seq in X]
    maxlen = int(np.percentile(lengths, 90))
    X = pad_sequences(X, maxlen=maxlen, padding='post', truncating='post')
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Sequential([
        Embedding(input_dim=8000, output_dim=128, input_length=maxlen),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    