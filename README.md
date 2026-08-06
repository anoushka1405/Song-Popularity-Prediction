# Song Popularity Predictor

> An end-to-end machine learning project that predicts song popularity using Spotify audio features
---

## Overview

This project analyzes 114,000 Spotify songs to predict song popularity using audio features like energy, danceability, and genre. An end-to-end ML pipeline compares 4 models, with  XGBoost achieving 77.5% accuracy, 0.859 ROC-AUC, and 84% recall for popular songs. Includes an interactive Streamlit web app for real-time predictions and feature exploration.

- Live Demo : https://song-popularity-prediction-f4p3qzfezzdjac44wesjzd.streamlit.app/

- Project Summary : https://anoushka1405.github.io/Song-Popularity-Prediction/Project_Summary.html


## Features

###  Machine Learning Pipeline
- Data preprocessing and cleaning
- Feature engineering 
- Class imbalance handling
- Model training with cross-validation
- Threshold tuning for optimal recall
- Comprehensive evaluation metrics

###  Analysis & Visualization
- Popularity distribution analysis
- Feature correlation heatmaps
- Genre performance comparison
- ROC curves and confusion matrices
- Feature importance rankings

---


## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/your-username/song-popularity-predictor.git
cd song-popularity-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Update dataset path** (in `app.py` line 234)
```python
data = load_data("path/to/your/dataset.xlsx")
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open in browser**
Navigate to `http://localhost:8501`

---

## 🤖 Models & Performance

### Models Implemented

| Model | Accuracy | ROC-AUC | F1-Score | Training Time |
|-------|----------|---------|----------|---------------|
| **XGBoost** ⭐ | 0.68 | **0.75** | **0.73** | ~60s |
| Random Forest | 0.68 | 0.74 | 0.69 | ~45s |
| Gradient Boosting | 0.67 | 0.73 | 0.68 | ~50s |
| Logistic Regression | 0.65 | 0.71 | 0.66 | ~5s |

⭐ **Best Model**: XGBoost with threshold tuning (0.404)

### Key Metrics
- **ROC-AUC Score**: 0.75 (Good discriminative ability)
- **Recall (Popular Songs)**: 90% (Excellent detection rate)
- **Precision**: 61% (Acceptable trade-off)
- **F1-Score**: 0.73 (Balanced performance)

---

## Technologies

### Core Technologies
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

### Libraries Used
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web Framework**: Streamlit
- **Model Interpretation**: SHAP

---

## Results

### Top 5 Most Important Features
1. **Genre** - Most influential predictor
2. **Energy** - Higher energy → more popular
3. **Danceability** - Danceable songs perform better
4. **Loudness** - Louder tracks correlate with popularity
5. **Energy × Danceability** - Engineered feature (interaction effect)

### Key Insights
- **Genre Matters**: Pop, hip-hop, and electronic dominate
- **Energy Wins**: High-energy songs are 2x more likely to be popular
- **Make It Danceable**: Danceability has strong positive correlation
- **Turn It Up**: Louder songs (>-5dB) perform better
- **Sweet Spot**: 3-4 minute songs are optimal
- **Less Acoustic**: Electronic production outperforms acoustic

### Business Impact
-  Helps streaming platforms improve recommendations
-  Guides artists on song characteristics for success
-  Assists record labels in identifying potential hits
-  Optimizes playlist curation for engagement

---

## Acknowledgments

- Spotify for providing the audio features API
- Scikit-learn community for excellent ML tools
- Streamlit team for the amazing web framework
- All contributors and supporters

---

<div align="center">

[⬆ Back to Top](#-song-popularity-predictor)

</div>
