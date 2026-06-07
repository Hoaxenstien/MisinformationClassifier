
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)

# ── 1. Load Data ─────────────────────────────────────────────────────────────

print("=" * 60)
print("MISINFORMATION CLASSIFIER — Multinomial Naive Bayes")
print("=" * 60)

df = pd.read_csv('fake_or_real_news.csv')
print(f"\nDataset loaded: {len(df)} articles")
print(f"Label distribution:\n{df['label'].value_counts()}\n")

# Combine title + text for richer features
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
df = df.dropna(subset=['content'])
df['binary_label'] = (df['label'] == 'REAL').astype(int)  # 1=REAL, 0=FAKE

X = df['content']
y = df['binary_label']

# Train/test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

# ── 2. Preprocessing Variants ─────────────────────────────────────────────────

STOP_WORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

def tokenize_basic(text):
    """Lowercase + split only."""
    return text.lower().split()

def tokenize_stopwords(text):
    """Lowercase + stopword removal."""
    tokens = text.lower().split()
    return [t for t in tokens if t not in STOP_WORDS]

def tokenize_stem(text):
    """Lowercase + stopword removal + Porter stemming."""
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return [stemmer.stem(t) for t in tokens]

def join(tokenizer):
    """Return a string-output function for sklearn's analyzer."""
    def fn(text):
        return ' '.join(tokenizer(text))
    return fn

# ── 3. Pipeline Definitions ───────────────────────────────────────────────────

pipelines = {
    'CountVec (no preprocessing)': Pipeline([
        ('vec', CountVectorizer(max_features=50000)),
        ('clf', MultinomialNB(alpha=1.0))
    ]),
    'TF-IDF (no preprocessing)': Pipeline([
        ('vec', TfidfVectorizer(max_features=50000)),
        ('clf', MultinomialNB(alpha=1.0))
    ]),
    'TF-IDF + Stopword Removal': Pipeline([
        ('vec', TfidfVectorizer(
            preprocessor=join(tokenize_stopwords),
            tokenizer=str.split,
            max_features=50000
        )),
        ('clf', MultinomialNB(alpha=1.0))
    ]),
    'TF-IDF + Stopwords + Stemming': Pipeline([
        ('vec', TfidfVectorizer(
            preprocessor=join(tokenize_stem),
            tokenizer=str.split,
            max_features=50000
        )),
        ('clf', MultinomialNB(alpha=1.0))
    ]),
    'TF-IDF + Bigrams + Stopwords': Pipeline([
        ('vec', TfidfVectorizer(
            preprocessor=join(tokenize_stopwords),
            tokenizer=str.split,
            ngram_range=(1, 2),
            max_features=100000
        )),
        ('clf', MultinomialNB(alpha=1.0))
    ]),
    'ComplementNB + TF-IDF + Stopwords': Pipeline([
        ('vec', TfidfVectorizer(
            preprocessor=join(tokenize_stopwords),
            tokenizer=str.split,
            max_features=50000
        )),
        ('clf', ComplementNB(alpha=1.0))
    ]),
}

# ── 4. Evaluate All Pipelines ─────────────────────────────────────────────────

results = []
trained_pipelines = {}

print("\n{:<40} {:>8} {:>8} {:>8} {:>8}".format(
    'Pipeline', 'Accuracy', 'Precision', 'Recall', 'F1'))
print("-" * 72)

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    # 5-fold CV F1 on training set
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1')

    results.append({
        'Pipeline': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'CV F1 Mean': cv_scores.mean(),
        'CV F1 Std': cv_scores.std()
    })
    trained_pipelines[name] = (pipe, y_pred)

    print("{:<40} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}".format(
        name, acc, prec, rec, f1))

results_df = pd.DataFrame(results)
best_name = results_df.loc[results_df['F1'].idxmax(), 'Pipeline']
best_pipe, best_pred = trained_pipelines[best_name]
print(f"\n★ Best pipeline by F1: {best_name}")

# ── 5. Detailed Report for Best Model ─────────────────────────────────────────

print(f"\n--- Classification Report ({best_name}) ---")
print(classification_report(y_test, best_pred, target_names=['FAKE', 'REAL']))

# ── 6. Figures ────────────────────────────────────────────────────────────────

sns.set_theme(style='whitegrid', palette='muted')
COLORS = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c']

# -- Fig 1: F1 comparison bar chart --
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(results_df['Pipeline'], results_df['F1'], color=COLORS, edgecolor='white')
ax.set_xlabel('F1 Score (Test Set)', fontsize=12)
ax.set_title('F1 Score by Preprocessing Pipeline', fontsize=14, fontweight='bold')
ax.set_xlim(0.85, 1.01)
for bar, val in zip(bars, results_df['F1']):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('figures/fig1_f1_comparison.png', dpi=150)
plt.close()
print("Saved: figures/fig1_f1_comparison.png")

# -- Fig 2: Confusion matrix for best model --
cm = confusion_matrix(y_test, best_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred FAKE', 'Pred REAL'],
            yticklabels=['True FAKE', 'True REAL'], ax=ax)
ax.set_title(f'Confusion Matrix\n{best_name}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig2_confusion_matrix.png', dpi=150)
plt.close()
print("Saved: figures/fig2_confusion_matrix.png")

# -- Fig 3: ROC curves for all pipelines --
fig, ax = plt.subplots(figsize=(7, 6))
for (name, (pipe, _)), color in zip(trained_pipelines.items(), COLORS):
    try:
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=color)
    except Exception:
        pass
ax.plot([0,1],[0,1],'k--', lw=1, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — All Pipelines', fontsize=14, fontweight='bold')
ax.legend(fontsize=7, loc='lower right')
plt.tight_layout()
plt.savefig('figures/fig3_roc_curves.png', dpi=150)
plt.close()
print("Saved: figures/fig3_roc_curves.png")

# -- Fig 4: Top predictive features for FAKE and REAL --
vec = best_pipe.named_steps['vec']
clf = best_pipe.named_steps['clf']
feature_names = np.array(vec.get_feature_names_out())

# log-prob differences: higher = more predictive of that class
# class 0 = FAKE, class 1 = REAL
log_probs = clf.feature_log_prob_  # shape (2, n_features)
fake_top_idx = np.argsort(log_probs[0])[-20:][::-1]
real_top_idx = np.argsort(log_probs[1])[-20:][::-1]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, idx, label, color in zip(
        axes,
        [fake_top_idx, real_top_idx],
        ['FAKE', 'REAL'],
        ['#e74c3c', '#2ecc71']):
    words = feature_names[idx]
    scores = log_probs[0 if label == 'FAKE' else 1][idx]
    ax.barh(words[::-1], scores[::-1], color=color, edgecolor='white')
    ax.set_title(f'Top 20 Features → {label}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Log Probability', fontsize=10)
plt.suptitle(f'Most Predictive Words\n({best_name})', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig4_top_features.png', dpi=150)
plt.close()
print("Saved: figures/fig4_top_features.png")

# -- Fig 5: Preprocessing metric comparison (grouped bar) --
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
x = np.arange(len(results_df))
width = 0.2
fig, ax = plt.subplots(figsize=(12, 5))
for i, (metric, color) in enumerate(zip(metrics, COLORS)):
    ax.bar(x + i*width, results_df[metric], width, label=metric, color=color)
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(results_df['Pipeline'], rotation=20, ha='right', fontsize=8)
ax.set_ylim(0.85, 1.01)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Classification Metrics by Pipeline', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('figures/fig5_metrics_comparison.png', dpi=150)
plt.close()
print("Saved: figures/fig5_metrics_comparison.png")

# -- Fig 6: CV F1 with error bars --
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(results_df['Pipeline'], results_df['CV F1 Mean'],
       yerr=results_df['CV F1 Std'], capsize=5, color=COLORS, edgecolor='white')
ax.set_ylabel('5-Fold CV F1 (mean ± std)', fontsize=11)
ax.set_title('Cross-Validation F1 by Pipeline', fontsize=13, fontweight='bold')
ax.set_xticklabels(results_df['Pipeline'], rotation=20, ha='right', fontsize=8)
ax.set_ylim(0.85, 1.01)
plt.tight_layout()
plt.savefig('figures/fig6_cv_f1.png', dpi=150)
plt.close()
print("Saved: figures/fig6_cv_f1.png")

# ── 7. Save results CSV ───────────────────────────────────────────────────────
results_df.to_csv('pipeline_results.csv', index=False)
print("\nSaved: pipeline_results.csv")
print("\nAll done.")
