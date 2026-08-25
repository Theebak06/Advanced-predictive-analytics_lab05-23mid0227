# Generated from: 23MID0227_TWEETSENTIMENT.ipynb
# Converted at: 2026-08-25T17:18:46.305Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

#NAME:THEEBAK S
#Product and Brand Sentiment Prediction from Tweet Data

#DATASET-A (TWITTER US AIRLINE)


import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Deep Learning / Transformers (Optional / Advanced)
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder


SEED = 42
np.random.seed(SEED)
OUT = 'lab05_outputs'
os.makedirs(OUT, exist_ok=True)

DATA_PATH = "C:/Users/ADMIN/Downloads/archive (12)/Tweets.csv"
TEXT_COL = 'text'
TARGET_COL = 'airline_sentiment'
ID_COL = 'tweet_id'
ENTITY_COL = 'airline'

# Load dataset and exclude leakage/PII columns
raw_df = pd.read_csv(DATA_PATH)
df = raw_df[[c for c in [ID_COL, TEXT_COL, TARGET_COL, ENTITY_COL] if c in raw_df.columns]].dropna(subset=[TEXT_COL, TARGET_COL]).copy()
df[TEXT_COL] = df[TEXT_COL].astype(str)

print(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df[TARGET_COL].value_counts())


#Minimal Tweet Normalization (Preserving Emojis, Hashtags, Punctuation)

def normalize_tweet(text):
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    text = re.sub(r'@\w+', '<USER>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df[TEXT_COL].map(normalize_tweet)

# 3. Stratified Split (Leakage-Safe Training & Locked Test Partition)
train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=SEED, stratify=df[TARGET_COL]
)

train_df.to_csv(os.path.join(OUT, 'train_manifest.csv'), index=False)
test_df.to_csv(os.path.join(OUT, 'test_manifest.csv'), index=False)

X_train, y_train = train_df['clean_text'], train_df[TARGET_COL]
X_test, y_test = test_df['clean_text'], test_df[TARGET_COL]

# ------------------------------------------------------------------------------
# 4. Baselines: Dummy Classifier & VADER Lexical Baseline
# ------------------------------------------------------------------------------
# Dummy (Most Frequent) Baseline
dummy = DummyClassifier(strategy='most_frequent', random_state=SEED)
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
print(f"Dummy Classifier Macro F1: {f1_score(y_test, dummy_pred, average='macro'):.4f}")

# VADER Lexical Baseline
vader = SentimentIntensityAnalyzer()
def vader_predict(text):
    compound = vader.polarity_scores(text)['compound']
    if compound >= 0.05: return 'positive'
    if compound <= -0.05: return 'negative'
    return 'neutral'

vader_pred = X_test.map(vader_predict)
print(f"VADER Lexical Baseline Macro F1: {f1_score(y_test, vader_pred, average='macro'):.4f}")

# 5. Classical Models Cross-Validation (5-Fold Stratified CV)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

models = {
    'MultinomialNB': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True)),
        ('clf', MultinomialNB(alpha=0.5))
    ]),
    'LogisticRegression': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED))
    ]),
    'LinearSVC': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True)),
        ('clf', LinearSVC(class_weight='balanced', random_state=SEED))
    ])
}

results = []
for name, pipe in models.items():
    scores = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring={'macro_f1': 'f1_macro', 'weighted_f1': 'f1_weighted', 'accuracy': 'accuracy'},
        n_jobs=-1, return_train_score=False
    )
    results.append({
        'model': name,
        'macro_f1_mean': scores['test_macro_f1'].mean(),
        'macro_f1_sd': scores['test_macro_f1'].std(),
        'weighted_f1_mean': scores['test_weighted_f1'].mean(),
        'accuracy_mean': scores['test_accuracy'].mean(),
        'fit_time_mean': scores['fit_time'].mean()
    })

cv_results = pd.DataFrame(results).sort_values('macro_f1_mean', ascending=False)
cv_results.to_csv(os.path.join(OUT, 'cv_results.csv'), index=False)
print("\n--- 5-Fold Cross-Validation Summary (Training Data Only) ---")
print(cv_results)


#  Locked Test Set Evaluation & Artifact Serialization

best_model_name = cv_results.iloc[0]['model']
best_pipeline = models[best_model_name]
best_pipeline.fit(X_train, y_train)

test_pred = best_pipeline.predict(X_test)
print(f"\n--- Final Locked Test Results ({best_model_name}) ---")
print(classification_report(y_test, test_pred, digits=4))

# Save artifacts
pred_df = test_df[[c for c in [ID_COL, TEXT_COL, TARGET_COL, ENTITY_COL] if c in test_df.columns]].copy()
pred_df['prediction'] = test_pred
pred_df.to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False)
joblib.dump(best_pipeline, os.path.join(OUT, 'selected_pipeline.joblib'))

#  Entity / Airline Stratification Analysis

if ENTITY_COL in pred_df.columns:
    entity_dist = pd.crosstab(pred_df[ENTITY_COL], pred_df['prediction'], normalize='index').round(4) * 100
    entity_dist['Total Support (N)'] = pred_df.groupby(ENTITY_COL).size()
    entity_dist.to_csv(os.path.join(OUT, 'entity_sentiment_distribution.csv'))
    print("\n--- Entity Sentiment Distribution (%) ---")
    print(entity_dist)


#  Advanced Extension: BiLSTM Model

MAX_WORDS = 20000
MAX_LEN = 80
EMBED_DIM = 128

tok = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tok.fit_on_texts(X_train)

Xtr_seq = pad_sequences(tok.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post', truncating='post')
Xte_seq = pad_sequences(tok.texts_to_sequences(X_test), maxlen=MAX_LEN, padding='post', truncating='post')

le = LabelEncoder()
ytr_enc = le.fit_transform(y_train)
yte_enc = le.transform(y_test)

bilstm = Sequential([
    Embedding(MAX_WORDS, EMBED_DIM),
    Bidirectional(LSTM(64)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

bilstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
bilstm_callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
]

bilstm.fit(Xtr_seq, ytr_enc, validation_split=0.15, epochs=6, batch_size=64, callbacks=bilstm_callbacks, verbose=1)
bilstm_pred = np.argmax(bilstm.predict(Xte_seq), axis=-1)
print(f"BiLSTM Test Macro F1: {f1_score(yte_enc, bilstm_pred, average='macro'):.4f}")



#DATASET B TWEETEVAL(SENTIMENT_TEST.CSV DATASET)

# ==============================================================================
# TWEET SENTIMENT ANALYSIS PIPELINE (sentiment_test_2.csv / TweetEval Format)
# ==============================================================================

import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn & Evaluation
!pip install -q accelerate>=0.26.0
!pip install --upgrade "accelerate>=1.1.0" "transformers[torch]"
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# Lexical Baseline Setup
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
except ImportError:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()

# Deep Learning (TensorFlow / Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Hugging Face Transformers
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate

# ==============================================================================
# 1. SETUP & DATA LOADING
# ==============================================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
OUT = 'lab05_outputs'
os.makedirs(OUT, exist_ok=True)

# Dataset file selection (checks for sentiment_test_2.csv or fallback)
DATA_PATH = "C:/Users/ADMIN/Downloads/archive (14)/sentiment_test.csv" 
TEXT_COL = 'text'
TARGET_COL = 'label'

print("=" * 75)
print(f"STAGE 1: LOADING & AUDITING DATASET ({DATA_PATH})")
print("=" * 75)

df = pd.read_csv(DATA_PATH).dropna(subset=[TEXT_COL, TARGET_COL]).copy()
df[TEXT_COL] = df[TEXT_COL].astype(str)
df[TARGET_COL] = df[TARGET_COL].astype(int)

# Mapping dictionary for labels (0: Negative, 1: Neutral, 2: Positive)
LABEL_NAMES = {0: 'negative', 1: 'neutral', 2: 'positive'}
print(f"Total Instances: {len(df)}")
print("Class Distribution:")
print(df[TARGET_COL].value_counts().rename(index=LABEL_NAMES))

# ==============================================================================
# 2. MINIMAL TWEET PREPROCESSING & STRATIFIED SPLIT
# ==============================================================================
print("\n" + "=" * 75)
print("STAGE 2: MINIMAL NORMALIZATION & STRATIFIED TRAIN/TEST SPLIT")
print("=" * 75)

def normalize_tweet(text: str) -> str:
    # Standardize URLs and user tags while preserving punctuation, emojis, and hashtags
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    text = re.sub(r'@\w+', '<USER>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df[TEXT_COL].map(normalize_tweet)

# 80/20 Stratified Partition
train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=SEED, stratify=df[TARGET_COL]
)

train_df.to_csv(os.path.join(OUT, 'train_manifest.csv'), index=False)
test_df.to_csv(os.path.join(OUT, 'test_manifest.csv'), index=False)

X_train, y_train = train_df['clean_text'], train_df[TARGET_COL]
X_test, y_test = test_df['clean_text'], test_df[TARGET_COL]

print(f"Training partition size: {len(X_train)} | Locked test partition size: {len(X_test)}")

# ==============================================================================
# 3. BASELINES (DUMMY & VADER LEXICAL BASELINE)
# ==============================================================================
print("\n" + "=" * 75)
print("STAGE 3: DUMMY & VADER BASELINE EVALUATION")
print("=" * 75)

# Dummy Classifier (Majority Class)
dummy = DummyClassifier(strategy='most_frequent', random_state=SEED)
dummy.fit(X_train, y_train)
dummy_preds = dummy.predict(X_test)
print(f"Dummy Classifier Macro F1:      {f1_score(y_test, dummy_preds, average='macro'):.4f}")

# VADER Lexical Baseline (Mapped to 0: Negative, 1: Neutral, 2: Positive)
def vader_predict_numeric(text: str) -> int:
    compound = vader.polarity_scores(text)['compound']
    if compound <= -0.05:
        return 0  # Negative
    elif compound >= 0.05:
        return 2  # Positive
    return 1      # Neutral

vader_preds = X_test.map(vader_predict_numeric)
print(f"VADER Lexical Baseline Macro F1: {f1_score(y_test, vader_preds, average='macro'):.4f}")

# ==============================================================================
# 4. 5-FOLD STRATIFIED CROSS-VALIDATION (CLASSICAL MODELS)
# ==============================================================================
print("\n" + "=" * 75)
print("STAGE 4: 5-FOLD CROSS-VALIDATION ON TRAINING DATA")
print("=" * 75)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

models = {
    'MultinomialNB': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True)),
        ('clf', MultinomialNB(alpha=0.5))
    ]),
    'LogisticRegression': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED))
    ]),
    'LinearSVC': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True)),
        ('clf', LinearSVC(class_weight='balanced', random_state=SEED))
    ])
}

cv_records = []
for name, pipe in models.items():
    scores = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring={'macro_f1': 'f1_macro', 'weighted_f1': 'f1_weighted', 'accuracy': 'accuracy'},
        n_jobs=-1, return_train_score=False
    )
    cv_records.append({
        'model': name,
        'macro_f1_mean': scores['test_macro_f1'].mean(),
        'macro_f1_sd': scores['test_macro_f1'].std(),
        'weighted_f1_mean': scores['test_weighted_f1'].mean(),
        'accuracy_mean': scores['test_accuracy'].mean(),
        'fit_time_mean': scores['fit_time'].mean()
    })

cv_results = pd.DataFrame(cv_records).sort_values('macro_f1_mean', ascending=False)
cv_results.to_csv(os.path.join(OUT, 'cv_results.csv'), index=False)
print(cv_results.to_string(index=False))

# ==============================================================================
# 5. FINAL LOCKED TEST EVALUATION & SERIALIZATION
# ==============================================================================
print("\n" + "=" * 75)
print("STAGE 5: FINAL LOCKED TEST SET EVALUATION")
print("=" * 75)

best_name = cv_results.iloc[0]['model']
best_pipe = models[best_name]
best_pipe.fit(X_train, y_train)

test_pred = best_pipe.predict(X_test)
print(f"Selected Best Classical Model: {best_name}\n")
print(classification_report(y_test, test_pred, target_names=['Negative (0)', 'Neutral (1)', 'Positive (2)'], digits=4))

# Save predictions and serialize model
pred_df = test_df.copy()
pred_df['predicted_label'] = test_pred
pred_df.to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False)
joblib.dump(best_pipe, os.path.join(OUT, 'selected_pipeline.joblib'))

# ==============================================================================
# 6. BiLSTM NEURAL SEQUENCE MODEL
# ==============================================================================
print("\n" + "=" * 75)
print("STAGE 6: BiLSTM DEEP LEARNING MODEL")
print("=" * 75)

MAX_WORDS = 25000
MAX_LEN = 80
EMBED_DIM = 128

tokenizer_lstm = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer_lstm.fit_on_texts(X_train)

Xtr_seq = pad_sequences(tokenizer_lstm.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post')
Xte_seq = pad_sequences(tokenizer_lstm.texts_to_sequences(X_test), maxlen=MAX_LEN, padding='post')

bilstm_model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBED_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(64)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

bilstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
bilstm_model.fit(
    Xtr_seq, y_train,
    validation_split=0.15,
    epochs=6,
    batch_size=64,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],
    verbose=1
)

bilstm_preds = np.argmax(bilstm_model.predict(Xte_seq), axis=-1)
print(f"\nBiLSTM Test Macro F1: {f1_score(y_test, bilstm_preds, average='macro'):.4f}")



# ==============================================================================
# 8. ACCEPTANCE TESTS
# ==============================================================================
print("\n" + "=" * 75)
print("STAGE 8: ACCEPTANCE TESTS & VERIFICATION")
print("=" * 75)

assert os.path.exists(os.path.join(OUT, 'cv_results.csv')), "cv_results.csv missing"
assert os.path.exists(os.path.join(OUT, 'selected_pipeline.joblib')), "selected_pipeline.joblib missing"
assert os.path.exists(os.path.join(OUT, 'test_predictions.csv')), "test_predictions.csv missing"

reloaded_pipe = joblib.load(os.path.join(OUT, 'selected_pipeline.joblib'))
np.testing.assert_array_equal(
    reloaded_pipe.predict(X_test.iloc[:15]),
    best_pipe.predict(X_test.iloc[:15])
)

print("All pipeline acceptance tests and reproducibility checks passed successfully.")


#Dataset C - Brand Sentiment Analysis Dataset


import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Modules
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Lexical Baseline Setup
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
except ImportError:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()

# Deep Learning (TensorFlow / Keras)
from transformers import logging
logging.set_verbosity_error()
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Pretrained Transformers
from transformers import pipeline

# ==============================================================================
# 1. SETUP & DATA AUDITING
# ==============================================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
OUT = 'lab05_outputs'
os.makedirs(OUT, exist_ok=True)

DATA_PATH = "C:/Users/ADMIN/Downloads/archive (15)/Dataset - Train.csv"
TEXT_COL = 'tweet_text'
RAW_TARGET_COL = 'is_there_an_emotion_directed_at_a_brand_or_product'
ENTITY_COL = 'emotion_in_tweet_is_directed_at'

print("=" * 80)
print("STAGE 1: DATA LOADING, LABEL HARMONIZATION & AUDITING")
print("=" * 80)

raw_df = pd.read_csv(DATA_PATH)
df = raw_df.dropna(subset=[TEXT_COL, RAW_TARGET_COL]).copy()

# Label Harmonization: Map to 3-class standard and drop ambiguous labels
LABEL_MAP = {
    'Negative emotion': 'negative',
    'No emotion toward brand or product': 'neutral',
    'Positive emotion': 'positive'
}

df = df[df[RAW_TARGET_COL].isin(LABEL_MAP.keys())].copy()
df['target_sentiment'] = df[RAW_TARGET_COL].map(LABEL_MAP)
df[TEXT_COL] = df[TEXT_COL].astype(str)

print(f"Dataset Loaded: {df.shape[0]} valid rows")
print("\nHarmonized Sentiment Distribution:")
print(df['target_sentiment'].value_counts())

# ==============================================================================
# 2. MINIMAL TWEET NORMALIZATION & STRATIFIED SPLIT
# ==============================================================================
print("\n" + "=" * 80)
print("STAGE 2: MINIMAL PREPROCESSING & LEAKAGE-SAFE TRAIN/TEST SPLIT")
print("=" * 80)

def normalize_tweet(text: str) -> str:
    # Normalize handles & URLs while keeping polarity punctuation, hashtags & emojis
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    text = re.sub(r'@\w+', '<USER>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df[TEXT_COL].map(normalize_tweet)

# 80/20 Stratified Partition
train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=SEED, stratify=df['target_sentiment']
)

train_df.to_csv(os.path.join(OUT, 'train_manifest.csv'), index=False)
test_df.to_csv(os.path.join(OUT, 'test_manifest.csv'), index=False)

X_train, y_train = train_df['clean_text'], train_df['target_sentiment']
X_test, y_test = test_df['clean_text'], test_df['target_sentiment']

print(f"Training partition size: {len(X_train)} | Locked test partition size: {len(X_test)}")

# ==============================================================================
# 3. BASELINES (DUMMY CLASSIFIER & VADER LEXICAL BASELINE)
# ==============================================================================
print("\n" + "=" * 80)
print("STAGE 3: DUMMY & VADER LEXICAL BASELINES")
print("=" * 80)

# Dummy Classifier (Majority Class)
dummy = DummyClassifier(strategy='most_frequent', random_state=SEED)
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
print(f"Dummy Classifier Macro F1:      {f1_score(y_test, dummy_pred, average='macro'):.4f}")

# VADER Lexical Baseline
def vader_predict(text: str) -> str:
    compound = vader.polarity_scores(text)['compound']
    if compound <= -0.05:
        return 'negative'
    elif compound >= 0.05:
        return 'positive'
    return 'neutral'

vader_pred = X_test.map(vader_predict)
print(f"VADER Lexical Baseline Macro F1: {f1_score(y_test, vader_pred, average='macro'):.4f}")

# ==============================================================================
# 4. 5-FOLD CROSS-VALIDATION (CLASSICAL ML PIPELINES)
# ==============================================================================
print("\n" + "=" * 80)
print("STAGE 4: 5-FOLD CROSS-VALIDATION (TRAINING DATA ONLY)")
print("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

models = {
    'MultinomialNB': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True)),
        ('clf', MultinomialNB(alpha=0.5))
    ]),
    'LogisticRegression': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED))
    ]),
    'LinearSVC': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True)),
        ('clf', LinearSVC(class_weight='balanced', random_state=SEED))
    ])
}

cv_records = []
for name, pipe in models.items():
    scores = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring={'macro_f1': 'f1_macro', 'weighted_f1': 'f1_weighted', 'accuracy': 'accuracy'},
        n_jobs=-1, return_train_score=False
    )
    cv_records.append({
        'model': name,
        'macro_f1_mean': scores['test_macro_f1'].mean(),
        'macro_f1_sd': scores['test_macro_f1'].std(),
        'weighted_f1_mean': scores['test_weighted_f1'].mean(),
        'accuracy_mean': scores['test_accuracy'].mean(),
        'fit_time_mean': scores['fit_time'].mean()
    })

cv_results = pd.DataFrame(cv_records).sort_values('macro_f1_mean', ascending=False)
cv_results.to_csv(os.path.join(OUT, 'cv_results.csv'), index=False)
print(cv_results.to_string(index=False))

# ==============================================================================
# 5. FINAL LOCKED TEST EVALUATION & ARTIFACT SERIALIZATION
# ==============================================================================
print("\n" + "=" * 80)
print("STAGE 5: LOCKED TEST EVALUATION & ENTITY STRATIFICATION")
print("=" * 80)

best_name = cv_results.iloc[0]['model']
best_pipe = models[best_name]
best_pipe.fit(X_train, y_train)

test_pred = best_pipe.predict(X_test)
print(f"Selected Model: {best_name}\n")
print(classification_report(y_test, test_pred, digits=4))

# Save predictions and serialize pipeline
pred_df = test_df.copy()
pred_df['predicted_sentiment'] = test_pred
pred_df.to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False)
joblib.dump(best_pipe, os.path.join(OUT, 'selected_pipeline.joblib'))

# Stratified Entity/Brand Analysis (with minimum support N >= 20)
if ENTITY_COL in pred_df.columns:
    entity_data = pred_df.dropna(subset=[ENTITY_COL])
    entity_counts = entity_data[ENTITY_COL].value_counts()
    valid_entities = entity_counts[entity_counts >= 20].index

    entity_sub = entity_data[entity_data[ENTITY_COL].isin(valid_entities)]
    entity_dist = pd.crosstab(
        entity_sub[ENTITY_COL], 
        entity_sub['predicted_sentiment'], 
        normalize='index'
    ).round(4) * 100
    entity_dist['Support (N)'] = entity_counts[valid_entities]
    entity_dist.to_csv(os.path.join(OUT, 'entity_sentiment_distribution.csv'))
    print("\nPredicted Sentiment by Product/Brand (%):\n", entity_dist)

# ==============================================================================
# 6. BiLSTM NEURAL SEQUENCE MODEL
# ==============================================================================
print("\n" + "=" * 80)
print("STAGE 6: BiLSTM NEURAL SEQUENCE MODEL")
print("=" * 80)

MAX_WORDS = 20000
MAX_LEN = 80
EMBED_DIM = 128

tokenizer_lstm = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer_lstm.fit_on_texts(X_train)

Xtr_seq = pad_sequences(tokenizer_lstm.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post')
Xte_seq = pad_sequences(tokenizer_lstm.texts_to_sequences(X_test), maxlen=MAX_LEN, padding='post')

le = LabelEncoder()
ytr_enc = le.fit_transform(y_train)
yte_enc = le.transform(y_test)

bilstm = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBED_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(64)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

bilstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
bilstm.fit(
    Xtr_seq, ytr_enc,
    validation_split=0.15,
    epochs=6,
    batch_size=64,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],
    verbose=1
)

bilstm_preds = np.argmax(bilstm.predict(Xte_seq), axis=-1)
print(f"\nBiLSTM Test Macro F1: {f1_score(yte_enc, bilstm_preds, average='macro'):.4f}")

# ==============================================================================
# 7. TRANSFORMER (Pretrained Twitter-RoBERTa / BERTweet)
# ==============================================================================
print("\n" + "=" * 80)
print("STAGE 7: TRANSFORMER MODEL EVALUATION (Twitter-RoBERTa)")
print("=" * 80)

# Pretrained Twitter-RoBERTa model evaluation pipeline
hf_classifier = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=-1, # CPU mode; set to 0 if CUDA GPU is available
    truncation=True,
    max_length=96
)

# Align transformer output strings with target classes
test_texts = X_test.tolist()
hf_raw = hf_classifier(test_texts, batch_size=32)
hf_preds = [p['label'].lower() for p in hf_raw]

print("\n=== Pretrained Twitter-RoBERTa Locked Test Results ===")
print(classification_report(y_test, hf_preds, digits=4))
print(f"Transformer Macro F1: {f1_score(y_test, hf_preds, average='macro'):.4f}")

# ==============================================================================
# 8. ACCEPTANCE TESTS
# ==============================================================================
print("\n" + "=" * 80)
print("STAGE 8: ACCEPTANCE TESTS & VERIFICATION")
print("=" * 80)

assert os.path.exists(os.path.join(OUT, 'cv_results.csv')), "cv_results.csv missing"
assert os.path.exists(os.path.join(OUT, 'selected_pipeline.joblib')), "selected_pipeline.joblib missing"
assert os.path.exists(os.path.join(OUT, 'test_predictions.csv')), "test_predictions.csv missing"

# Reload pipeline and check prediction reproducibility
reloaded_pipe = joblib.load(os.path.join(OUT, 'selected_pipeline.joblib'))
np.testing.assert_array_equal(
    reloaded_pipe.predict(X_test.iloc[:20]),
    best_pipe.predict(X_test.iloc[:20])
)

print("All acceptance criteria, data artifacts, and pipeline checks passed successfully.")