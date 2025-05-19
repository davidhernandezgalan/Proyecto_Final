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

