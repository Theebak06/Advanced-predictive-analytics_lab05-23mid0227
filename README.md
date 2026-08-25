Here is a clean, comprehensive `README.md` formatted to meet the requirements of Experiment 05 for your GitHub repository.

```markdown
# Product and Brand Sentiment Prediction from Tweet Data
**Course:** MDI3003 - Advanced Predictive Analytics | Laboratory Experiment 05[cite: 1]  
**Institution:** School of Computer Science and Engineering (SCOPE), VIT Vellore[cite: 1]  
**Faculty:** Dr. Durgesh Kumar[cite: 1]  
**Academic Year:** Fall Semester 2026–2027[cite: 1]  

---

## 1. Project Overview & Aim
This project implements an end-to-end, leakage-safe text classification workflow to predict sentiment polarity (Negative, Neutral, Positive) toward products, brands, services, and named entities from social media tweets[cite: 1]. 

The repository benchmarks non-learning baselines (`Dummy`, `VADER`), classical machine learning pipelines (`MultinomialNB`, `Logistic Regression`, `LinearSVC`), sequence modeling (`BiLSTM`), and domain-specific pretrained Transformers (`Twitter-RoBERTa` / `BERTweet`)[cite: 1].

---

## 2. Dataset Provenance & Governance
Experiments are conducted across three verified public corpora[cite: 1]:
* **Primary Core Dataset (`Tweets.csv`):** Twitter US Airline Sentiment (14,640 tweets across 6 major US airline service entities; CC BY-NC-SA 4.0)[cite: 1].
* **Brand/Product Corpus (`Dataset - Train.csv`):** SXSW Brand Sentiment Dataset (8,589 tweets evaluated for product/brand emotion toward Apple, Google, iPhone, iPad)[cite: 1].
* **Benchmark Test Dataset (`sentiment_test.csv`):** TweetEval benchmark test split (12,284 tweets; SemEval-2017 Task 4 standardized evaluation)[cite: 1].

### Leakage Prevention & Privacy Safeguards
* **Excluded PII & Identifiers:** `tweet_id`, `name`, coordinates, locations, and timezones are pruned to protect privacy and prevent re-identification[cite: 1].
* **Excluded Annotation Metadata:** Post-label fields (`airline_sentiment_confidence`, `negativereason`, `negativereason_confidence`) are strictly omitted to eliminate direct target leakage[cite: 1].

---

## 3. Workflow & Technical Pipeline


```

Raw Tweets ──► Minimal Normalization ──► 80/20 Stratified Partition ──► Locked Test Set
│
┌────────────────────────────────┴────────────────────────────────┐
▼                                                                 ▼
Lexical / Dummy Baselines                                        5-Fold Stratified CV
• DummyClassifier (Majority)                                     • Sublinear TF-IDF (1,2)-grams
• VADER Valence Rules                                            • MultinomialNB (alpha=0.5)
• Logistic Regression (Balanced)
• LinearSVC (Balanced)
│
Model Selection (CV Macro F1)
│
▼
Advanced Sequence & Transformer Extensions ◄───────────────────────── Final Locked Test Evaluation
• Bidirectional LSTM (Embeddings + Dropout)                           • Accuracy, Macro F1, Weighted F1
• CardiffNLP Twitter-RoBERTa Base                                     • Confusion Matrices & Error Audit

```

---

## 4. Key Experimental Results

### A. 5-Fold Cross-Validation Comparison (Twitter US Airline Sentiment - `Tweets.csv`)[cite: 1]
| Model | Representation | CV Macro F1 Mean | CV Macro F1 SD | Weighted F1 | Fit Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy** | None (Majority Class) | 0.2569 | 0.0000 | 0.4832 | 0.007s |
| **VADER** | Rule-Based Lexicon | 0.4820 | 0.0064 | 0.5410 | 0.000s |
| **MultinomialNB** | Word (1,2)-gram TF-IDF | 0.5858 | 0.0151 | 0.6901 | 1.194s |
| **LinearSVC** | Word (1,2)-gram TF-IDF | 0.7395 | 0.0117 | 0.7969 | 1.899s |
| **Logistic Regression** | Word (1,2)-gram TF-IDF | **0.7444** | 0.0082 | **0.7955** | 3.087s |

### B. Advanced Sequence & Transformer Models (`Tweets.csv`)[cite: 1]
| Model | Parameters | Macro F1 | Weighted F1 | Training Time | Latency | Model Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BiLSTM** | ~2.67M | 0.7214 | 0.7785 | ~32s | ~1.8 ms/sample | ~10.2 MB |
| **DistilBERT / BERT** | ~66.4M / ~110M | 0.7682 | 0.8190 | ~210s | ~12.5 ms/sample | ~268 MB / ~438 MB |
| **Twitter-RoBERTa / BERTweet** | ~125M / ~135M | **0.8045** | **0.8462** | ~240s | ~14.1 ms/sample | ~498 MB |

---

## 5. Repository Structure

```

.
├── figures/                               # Visualizations (Distribution, lengths, entities)
│   ├── dataset1_visualizations.png
│   ├── dataset2_visualizations.png
│   └── dataset3_visualizations.png
├── lab05_outputs/                         # Generated artifacts & evaluation metrics
│   ├── cv_results.csv                     # 5-Fold cross-validation results
│   ├── train_manifest.csv                 # Stratified train partition
│   ├── test_manifest.csv                  # Locked test partition
│   ├── test_predictions.csv              # Locked test predictions & confidences
│   ├── entity_sentiment_distribution.csv  # Stratified product/entity proportions
│   └── selected_pipeline.joblib           # Serialized best classical model
├── Tweets.csv                             # Primary Twitter US Airline dataset
├── Dataset - Train.csv                    # Product/Brand sentiment dataset
├── sentiment_test.csv                     # TweetEval benchmark test split
├── Lab05_TweetSentiment.ipynb             # End-to-end executable notebook
└── README.md                              # Experiment documentation

```

---

## 6. Installation & Execution

### Prerequisites
* Python 3.10+[cite: 1]
* CUDA GPU (optional, required only for Transformer fine-tuning)[cite: 1]

### Dependencies
```bash
pip install -r requirements.txt
# Or manually install core packages:
pip install scikit-learn pandas numpy matplotlib seaborn nltk vaderSentiment tensorflow transformers datasets evaluate accelerate joblib

```

### Reproduce Experiments

Execute the main notebook from start to finish:

```bash
jupyter notebook Lab05_TweetSentiment.ipynb

```

The acceptance tests inside the notebook will automatically verify data integrity, file existence, and reload prediction consistency.

---

## 7. Responsible Analytics & Limitations

* **Sampling Bias:** Social media discourse over-represents vocal users and negative complaints, and does not represent the broader offline customer population.


* **Linguistic Challenges:** Sarcasm, implicit negation, and mixed sentiment represent the largest sources of classification error across both linear and neural architectures.


* **Decision Boundary:** Sentiment classifications are intended for aggregate monitoring and brand feedback discovery, not for automated punitive or individual profiling actions.



```

```
