# Movie Sentiment Analysis

A machine learning project for classifying movie reviews as positive or negative using Natural Language Processing (NLP) techniques and ensemble learning methods.

## Project Overview

This project analyzes Amazon Movies and TV reviews to build a sentiment classification system. The pipeline includes data sampling, exploratory data analysis, text preprocessing, feature extraction using TF-IDF, and model training with hyperparameter tuning.

## Dataset

- **Source**: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io) - Movies and TV category
- **Original Size**: ~8GB (Movies_and_TV.jsonl)
- **Sampled Data**: 500K reviews initially sampled, then balanced to 60K reviews (30K positive, 30K negative)
- **Features**: Rating, title, review text, timestamp, verified purchase status

## Project Structure

```
Movie-Sentiment-Analysis/
├── DataSampling.ipynb          # Data sampling and preprocessing notebook
├── MainWork.ipynb              # Main analysis and model training notebook
├── data sampling graphs/       # Visualizations from data sampling phase
├── MainWork Graphs/            # Model performance visualizations
│   ├── confusion_matrices.png
│   ├── model_comparison.png
│   ├── roc_curve_comparison.png
│   ├── precision_recall_comparison.png
│   ├── sentiment_distribution_main.png
│   └── ...
├── trained models/             # Saved model files
│   ├── stacking_ensemble_model.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── logistic_regression_tuned.joblib
│   ├── naive_bayes_tuned.joblib
│   └── linear_svm_tuned.joblib
├── .gitignore
└── README.md
```

## Methodology

### 1. Data Preprocessing
- Balanced sampling to address class imbalance (30K positive, 30K negative)
- Text cleaning (HTML removal, whitespace normalization, lowercasing)
- Combined title and review text for richer features

### 2. Feature Extraction
- **TF-IDF Vectorization** with unigrams and bigrams
- 81,400 features extracted from the text corpus
- Sublinear TF scaling for better feature representation

### 3. Models Trained

| Model | Default Accuracy | Tuned Accuracy |
|-------|-----------------|----------------|
| Logistic Regression | 93.52% | 94.09% |
| Multinomial Naive Bayes | 93.33% | 93.55% |
| Linear SVM | 94.17% | 94.17% |
| Hard Voting Ensemble | - | 94.30% |
| Soft Voting Ensemble | - | 94.33% |
| **Stacking Ensemble** | - | **94.45%** |

### 4. Hyperparameter Tuning
- GridSearchCV with 5-fold cross-validation
- Optimized parameters for each base model

### 5. Ensemble Methods
- **Voting Classifier**: Hard and soft voting combinations
- **Stacking Classifier**: Meta-learner (Logistic Regression) on base model predictions

## Results

- **Best Model**: Stacking Ensemble with **94.45% accuracy**
- **AUC Score**: 0.988 (Stacking Ensemble)
- **Average Precision**: 0.988 (Stacking Ensemble)

## Key Findings

### Top Positive Phrases (Log-Ratio Analysis)
- "wait for season", "really enjoyed this", "highly recommend this"

### Top Negative Phrases
- "don waste your", "waste your time", "worst movies"

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
joblib
```

## Usage

### Loading the Trained Model

```python
import joblib

# Load the model and vectorizer
model = joblib.load("trained models/stacking_ensemble_model.joblib")
tfidf = joblib.load("trained models/tfidf_vectorizer.joblib")

# Predict sentiment for new reviews
new_review = ["This movie was absolutely amazing!"]
features = tfidf.transform(new_review)
prediction = model.predict(features)
# 1 = Positive, 0 = Negative
```

## Data Files (Not Included)

The following large data files are not included in this repository due to size constraints:
- `Movies_and_TV.jsonl` (~8GB) - Original dataset
- `Sampled_500K_Reviews.csv` (~188MB) - Intermediate sampled data
- `Final_Reviews.csv` (~26MB) - Final processed dataset

## Author

Girish Raj Thapa

## License

This project is for educational purposes as part of an Artificial Intelligence module.