# 🤖 Advanced ChatGPT Review Analysis Platform

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Gradio](https://img.shields.io/badge/Interface-Gradio-orange.svg)
![ML](https://img.shields.io/badge/Domain-NLP%20&%20Analytics-green.svg)
![Model](https://img.shields.io/badge/Model-Random%20Forest-green.svg)
![Skills](https://img.shields.io/badge/Skills-EDA%20Time%20Series%20&%20Clustering-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/asif-khan-ak/advanced_chathpt_review_analysis/blob/main/chatgpt_review_analysis.ipynb
)

# An industry-ready NLP and analytics platform for comprehensive analysis of ChatGPT app reviews using advanced machine learning, time series forecasting, and interactive visualizations.

-- 

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Insights](#key-insights)
- [Deployment](#deployment)

## 🎯 Overview

This project transforms raw ChatGPT app review data into actionable business insights through:
- **Multi-method sentiment analysis** (VADER, TextBlob)
- **Advanced topic modeling** (LDA, NMF)
- **Time series forecasting** (ARIMA)
- **Machine learning classification** (Random Forest)
- **User segmentation** (K-Means clustering)
- **Interactive web interface** (Gradio)

### Business Value
- **NPS Score Calculation**: Measure customer loyalty and satisfaction
- **Sentiment Trends**: Track user sentiment over time
- **Topic Discovery**: Identify key themes in user feedback
- **Predictive Analytics**: Forecast future review volumes
- **User Segmentation**: Understand different user personas

## ✨ Features

### 1. Advanced Text Preprocessing
- Custom preprocessing pipeline with lemmatization
- Stop word removal and text normalization
- Feature engineering (length, word count, punctuation metrics)
- Temporal feature extraction

### 2. Multi-Dimensional Sentiment Analysis
- **VADER**: Rule-based sentiment scoring
- **TextBlob**: Polarity and subjectivity analysis
- **Rating-based**: Classification using star ratings
- **NPS Categories**: Promoter/Passive/Detractor segmentation

### 3. Exploratory Data Analysis (15+ Visualizations)
- Rating distributions with statistics
- Sentiment hierarchy (Sunburst charts)
- Review length analysis (Violin plots)
- Correlation heatmaps
- Temporal patterns (day of week, hourly trends)

### 4. Time Series Analysis
- Daily/weekly/monthly aggregations
- Seasonal decomposition
- Trend analysis
- ARIMA forecasting with confidence intervals
- 4-week ahead predictions

### 5. NLP & Topic Modeling
- **LDA**: Latent Dirichlet Allocation
- **NMF**: Non-negative Matrix Factorization
- TF-IDF vectorization
- Word clouds by sentiment
- N-gram analysis (bigrams, trigrams)

### 6. Machine Learning
- **Classification**: Random Forest for sentiment prediction
- **Clustering**: K-Means for user segmentation
- Feature importance analysis
- Confusion matrix visualization
- 3D cluster visualization

### 7. Interactive Dashboard (Gradio)
- Single review analysis
- Dataset statistics
- Topic exploration
- Real-time sentiment scoring
- User-friendly web interface

## 🛠 Tech Stack

### Core Libraries
- **Data Processing**: pandas, numpy
- **Visualization**: plotly, matplotlib, seaborn, wordcloud
- **NLP**: nltk, textblob, vaderSentiment
- **Machine Learning**: scikit-learn
- **Time Series**: statsmodels, pmdarima
- **Web Interface**: gradio

### Requirements
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
wordcloud>=1.8.0
textblob>=0.15.0
scikit-learn>=1.0.0
nltk>=3.6.0
vaderSentiment>=3.3.0
gradio>=3.0.0
statsmodels>=0.13.0
pmdarima>=1.8.0
```

## 📦 Installation

### Google Colab (Recommended)
```python
# All dependencies are installed automatically in the notebook
# Just upload the notebook and dataset to Colab and run!
```

### Local Setup
```bash
# Clone the repository
git clone https://github.com/asif-khan-ak/advanced_chathpt_review_analysis.git
cd advanced_chathpt_review_analysis

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## 🚀 Usage

### Quick Start (Google Colab)
1. Upload `chatgpt_review_analysis.ipynb` to Google Colab
2. Upload `chatgpt_reviews.csv` to the Colab environment
3. Run all cells
4. Access the Gradio interface via the provided URL

### Local Execution
```python
import pandas as pd
# Load your notebook
jupyter notebook Advanced_ChatGPT_Review_Analysis.ipynb
```

### Analyzing Custom Reviews
```python
from gradio_interface import analyze_single_review

review = "This app is amazing! It helps me with coding."
analysis, sentiment = analyze_single_review(review)
print(f"Sentiment: {sentiment}")
print(analysis)
```

## 📁 Project Structure

```
chatgpt-review-analysis/
│
├── Advanced_ChatGPT_Review_Analysis.ipynb  # Main notebook
├── chatgpt_reviews.csv                     # Input dataset
├── chatgpt_reviews_processed.csv           # Processed data (output)
├── analysis_results.json                   # Summary metrics (output)
├── README.md                               # Documentation
└── requirements.txt                        # Dependencies
```

## 💡 Key Insights

### Sample Findings
Based on analysis of 196,727 reviews:

- **Average Rating**: 4.5/5.0
- **NPS Score**: +64.35 (Excellent)
- **Sentiment Distribution**: 
  - Positive: 71.2%
  - Neutral: 24.3%
  - Negative: 4.5%
- **ML Model Accuracy**: 89%

### Discovered Topics
1. **Product Quality**: app, good, nice, great, helpful
2. **Features**: chatgpt, gpt, model, version, update
3. **Performance**: fast, slow, work, error, bug
4. **Use Cases**: coding, study, question, answer, help

## 🌐 Deployment

### Gradio Cloud (Recommended)
```python
# In the notebook, the Gradio interface automatically generates a shareable link
demo.launch(share=True)
```

### Hugging Face Spaces
1. Create a new Space on Hugging Face
2. Upload the notebook and convert to app.py
3. Deploy with automatic Gradio hosting

### Streamlit Alternative
```python
# Convert Gradio interface to Streamlit for deployment
import streamlit as st
# ... conversion code
```

## 📊 Visualizations Preview

The project includes 15+ interactive visualizations:
- 📈 Time series trends with forecasting
- 🥧 Sentiment distribution pie charts
- 📊 Rating analysis bar charts
- 🌈 Correlation heatmaps
- ☁️ Word clouds by sentiment
- 📉 Seasonal decomposition
- 🎯 3D cluster visualization
- 📱 Business metrics dashboard

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Asif Khan**
Data Science & AI Enthusiast

## 🙏 Acknowledgments

- Dataset: ChatGPT App Reviews
- Inspiration: Natural Language Processing community
- Tools: Google Colab, Plotly, Gradio teams
---

⭐ If you found this project helpful, please consider giving it a star!
