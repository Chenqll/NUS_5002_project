import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import transformers
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

import nltk
import re


from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, 
                             roc_curve, f1_score, confusion_matrix,
                             classification_report)


print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
# 2.10.0
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

data = pd.read_csv('data/comments.csv')
print(data)
data.dropna(inplace=True)

def clean_text(text):
    # remove urls
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    # remove puncutations
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    # remove tab
    text = re.sub('\t', ' ',  text)
    # remove multiple spaces
    text = re.sub(r" +", ' ', text)
    return text
    
print(data)
data['NEW_COMMENT'] = data['Comment'].apply(clean_text)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")