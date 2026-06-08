import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve
)

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)

# download stopwords if not already present
nltk.download('stopwords', quiet=True)

# load data
print("Loading dataset...")
df = pd.read_csv('fake_or_real_news.csv')
print(f"Dataset loaded: {len(df)} articles")
print(f"Label distribution:\n{df['label'].value_counts()}\n")

# combine title and body text for richer features
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
df = df.dropna(subset=['content'])
df['binaryLabel'] = (df['label'] == 'REAL').astype(int)  # 1=REAL, 0=FAKE

X = df['content']
y = df['binaryLabel']

# 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

# preprocessing helpers
stopWords = set(stopwords.words('english'))
stemmer = PorterStemmer()

def tokenizeBasic(text):
    # lowercase and split only
    return text.lower().split()

def tokenizeStopwords(text):
    # lowercase and remove stopwords
    tokens = text.lower().split()
    return [t for t in tokens if t not in stopWords]

def tokenizeStem(text):
    # lowercase, remove stopwords, then stem
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in stopWords]
    return [stemmer.stem(t) for t in tokens]

def joinTokens(tokenizer):
    # wraps a tokenizer to return a single string (required by sklearn preprocessor)
    def fn(text):
        return ' '.join(tokenizer(text))
    return fn

# pipeline definitions
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
            preprocessor=joinTokens(tokenizeStopwords),
            tokenizer=str.split,
            max_features=50000
        )),
        ('clf', MultinomialNB(alpha=1.0))
    ]),
    'TF-IDF + Stopwords + Stemming': Pipeline([
        ('vec', TfidfVectorizer(
            preprocessor=joinTokens(tokenizeStem),
            tokenizer=str.split,
            max_features=50000
        )),
        ('clf', MultinomialNB(alpha=1.0))
    ]),
    'TF-IDF + Bigrams + Stopwords': Pipeline([
        ('vec', TfidfVectorizer(
            preprocessor=joinTokens(tokenizeStopwords),
            tokenizer=str.split,
            ngram_range=(1, 2),
            max_features=100000
        )),
        ('clf', MultinomialNB(alpha=1.0))
    ]),
    'ComplementNB + TF-IDF + Stopwords': Pipeline([
        ('vec', TfidfVectorizer(
            preprocessor=joinTokens(tokenizeStopwords),
            tokenizer=str.split,
            max_features=50000
        )),
        ('clf', ComplementNB(alpha=1.0))
    ]),
}

# train and evaluate each pipeline
results = []
trainedPipelines = {}

print("\n{:<40} {:>8} {:>8} {:>8} {:>8}".format(
    'Pipeline', 'Accuracy', 'Precision', 'Recall', 'F1'))


for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    yPred = pipe.predict(X_test)

    acc  = accuracy_score(y_test, yPred)
    prec = precision_score(y_test, yPred)
    rec  = recall_score(y_test, yPred)
    f1   = f1_score(y_test, yPred)

    # 5 fold cross validation on training set
    cvScores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1')

    results.append({
        'Pipeline': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'CV F1 Mean': cvScores.mean(),
        'CV F1 Std': cvScores.std()
    })
    trainedPipelines[name] = (pipe, yPred)

    print("{:<40} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}".format(
        name, acc, prec, rec, f1))

resultsDF = pd.DataFrame(results)
bestName = resultsDF.loc[resultsDF['F1'].idxmax(), 'Pipeline']
bestPipe, bestPred = trainedPipelines[bestName]
print(f"\nBest pipeline by F1: {bestName}")

print(f"\nClassification Report ({bestName})")
print(classification_report(y_test, bestPred, target_names=['FAKE', 'REAL']))

# generate figures
sns.set_theme(style='whitegrid', palette='muted')
colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c']

# fig 1: F1 comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(resultsDF['Pipeline'], resultsDF['F1'], color=colors, edgecolor='white')
ax.set_xlabel('F1 Score (Test Set)', fontsize=12)
ax.set_title('F1 Score by Preprocessing Pipeline', fontsize=14, fontweight='bold')
ax.set_xlim(0.85, 1.01)
for bar, val in zip(bars, resultsDF['F1']):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('figures/fig1_f1_comparison.png', dpi=150)
plt.close()
print("Saved figures/fig1_f1_comparison.png")

# fig 2: confusion matrix for best model
cm = confusion_matrix(y_test, bestPred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred FAKE', 'Pred REAL'],
            yticklabels=['True FAKE', 'True REAL'], ax=ax)
ax.set_title(f'Confusion Matrix\n{bestName}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig2_confusion_matrix.png', dpi=150)
plt.close()
print("Saved figures/fig2_confusion_matrix.png")

# fig 3: ROC curves for all pipelines
fig, ax = plt.subplots(figsize=(7, 6))
for (name, (pipe, _)), color in zip(trainedPipelines.items(), colors):
    try:
        yProb = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, yProb)
        auc = roc_auc_score(y_test, yProb)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=color)
    except Exception:
        pass
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - All Pipelines', fontsize=14, fontweight='bold')
ax.legend(fontsize=7, loc='lower right')
plt.tight_layout()
plt.savefig('figures/fig3_roc_curves.png', dpi=150)
plt.close()
print("Saved figures/fig3_roc_curves.png")

# fig 4: top predictive features for fake and real
vec = bestPipe.named_steps['vec']
clf = bestPipe.named_steps['clf']
featureNames = np.array(vec.get_feature_names_out())

# higher log prob = more predictive of that class (0=fake, 1=real)
logProbs = clf.feature_log_prob_
fakeTopIdx = np.argsort(logProbs[0])[-20:][::-1]
realTopIdx = np.argsort(logProbs[1])[-20:][::-1]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, idx, label, color in zip(
        axes,
        [fakeTopIdx, realTopIdx],
        ['FAKE', 'REAL'],
        ['#e74c3c', '#2ecc71']):
    words = featureNames[idx]
    scores = logProbs[0 if label == 'FAKE' else 1][idx]
    ax.barh(words[::-1], scores[::-1], color=color, edgecolor='white')
    ax.set_title(f'Top 20 Features for {label}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Log Probability', fontsize=10)
plt.suptitle(f'Most Predictive Words\n({bestName})', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig4_top_features.png', dpi=150)
plt.close()
print("Saved figures/fig4_top_features.png")

# fig 5: grouped bar chart of all metrics by pipeline
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
x = np.arange(len(resultsDF))
width = 0.2
fig, ax = plt.subplots(figsize=(12, 5))
for i, (metric, color) in enumerate(zip(metrics, colors)):
    ax.bar(x + i*width, resultsDF[metric], width, label=metric, color=color)
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(resultsDF['Pipeline'], rotation=20, ha='right', fontsize=8)
ax.set_ylim(0.85, 1.01)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Classification Metrics by Pipeline', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('figures/fig5_metrics_comparison.png', dpi=150)
plt.close()
print("Saved figures/fig5_metrics_comparison.png")

# fig 6: cross validation F1 with error bars
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(resultsDF['Pipeline'], resultsDF['CV F1 Mean'],
       yerr=resultsDF['CV F1 Std'], capsize=5, color=colors, edgecolor='white')
ax.set_ylabel('5-Fold CV F1 (mean +/- std)', fontsize=11)
ax.set_title('Cross-Validation F1 by Pipeline', fontsize=13, fontweight='bold')
ax.set_xticklabels(resultsDF['Pipeline'], rotation=20, ha='right', fontsize=8)
ax.set_ylim(0.85, 1.01)
plt.tight_layout()
plt.savefig('figures/fig6_cv_f1.png', dpi=150)
plt.close()
print("Saved figures/fig6_cv_f1.png")

# save results to csv
resultsDF.to_csv('pipeline_results.csv', index=False)
print("\nSaved pipeline_results.csv")
