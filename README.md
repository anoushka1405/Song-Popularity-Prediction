# 🎵 Song Popularity Predictor

> An end-to-end machine learning project that predicts song popularity using Spotify audio features

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models & Performance](#models--performance)
- [Technologies](#technologies)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project analyzes 114,000 songs from Spotify to predict whether a song will be popular based on its audio features. Using machine learning algorithms, the system achieves **75% ROC-AUC score** and provides an interactive web application for real-time predictions.

### Live Demo : https://song-popularity-prediction-f4p3qzfezzdjac44wesjzd.streamlit.app/

### Project Summary : https://anoushka1405.github.io/Song-Popularity-Prediction/Project_Summary.html

### Key Highlights

- 📊 **Comprehensive EDA** with 15+ visualizations
- 🤖 **3 ML Models** trained and compared
- 🎨 **Interactive Web App** built with Streamlit
- 📈 **Feature Engineering** with 6 custom features
- 🎯 **90% Recall** for popular songs
- 🌐 **Deployment Ready** for Streamlit Cloud

---

## ✨ Features

### 🏠 Interactive Web Application
- **Data Explorer**: Visualize distributions, correlations, and genre analysis
- **Model Training**: Train and compare multiple ML models
- **Live Predictions**: Real-time popularity predictions with confidence scores
- **Model Insights**: Feature importance and correlation analysis
- **Professional UI**: Clean, intuitive interface with Plotly visualizations

### 🔬 Machine Learning Pipeline
- Data preprocessing and cleaning
- Feature engineering (6 custom features)
- Class imbalance handling
- Model training with cross-validation
- Threshold tuning for optimal recall
- Comprehensive evaluation metrics

### 📊 Analysis & Visualization
- Popularity distribution analysis
- Feature correlation heatmaps
- Genre performance comparison
- ROC curves and confusion matrices
- Feature importance rankings

---

## 🖼️ Demo

### Home Page
*Beautiful overview with dataset statistics and popularity distribution*

### Data Explorer
*Interactive visualizations showing feature distributions and correlations*

### Model Training
*Train multiple models and compare performance in real-time*

### Make Predictions
*Input song features and get instant predictions with confidence scores*

---

## 🚀 Installation

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

## 💻 Usage

### Running the Jupyter Notebook
```bash
jupyter notebook Song_Popularity_Analysis_Complete.ipynb
```

### Running the Streamlit App
```bash
streamlit run app.py
```

### Running XGBoost Analysis
```bash
python xgboost_shap_analysis.py
```

### Example Prediction
```python
# Input features
song_features = {
    'danceability': 0.75,
    'energy': 0.85,
    'valence': 0.65,
    'tempo': 128,
    'loudness': -5.0,
    # ... other features
}

# Get prediction
prediction = model.predict(song_features)
# Output: "Popular" with 87% confidence
```

---

## 📁 Project Structure

```
song-popularity-predictor/
│
├── app.py                                    # Streamlit web application
├── Song_Popularity_Analysis_Complete.ipynb  # Main analysis notebook
├── xgboost_shap_analysis.py                 # XGBoost & SHAP analysis
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
├── STREAMLIT_GUIDE.md                       # Detailed app guide
│
├── outputs/
│   ├── Song_Popularity_Analysis_Report.docx # Professional report
│   └── Project_Summary.html                 # Visual summary
│
└── data/
    └── dataset.xlsx                         # Dataset (not included)
```

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

## 🛠️ Technologies

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

## 📊 Results

### Top 5 Most Important Features
1. **Genre** - Most influential predictor
2. **Energy** - Higher energy → more popular
3. **Danceability** - Danceable songs perform better
4. **Loudness** - Louder tracks correlate with popularity
5. **Energy × Danceability** - Engineered feature (interaction effect)

### Key Insights
- 🎸 **Genre Matters**: Pop, hip-hop, and electronic dominate
- ⚡ **Energy Wins**: High-energy songs are 2x more likely to be popular
- 💃 **Make It Danceable**: Danceability has strong positive correlation
- 🔊 **Turn It Up**: Louder songs (>-5dB) perform better
- ⏱️ **Sweet Spot**: 3-4 minute songs are optimal
- 🎹 **Less Acoustic**: Electronic production outperforms acoustic

### Business Impact
- ✅ Helps streaming platforms improve recommendations
- ✅ Guides artists on song characteristics for success
- ✅ Assists record labels in identifying potential hits
- ✅ Optimizes playlist curation for engagement

---

## 🔮 Future Improvements

### Short Term
- [ ] Add XGBoost model to Streamlit app
- [ ] Implement SHAP explanations in UI
- [ ] Add batch prediction upload feature
- [ ] Create downloadable prediction reports

### Medium Term
- [ ] Incorporate temporal features (release date, season)
- [ ] Add artist popularity aggregation
- [ ] Implement hyperparameter tuning in UI
- [ ] Create model retraining pipeline

### Long Term
- [ ] Deploy as REST API with FastAPI
- [ ] Add deep learning models (Neural Networks)
- [ ] Incorporate social media engagement data
- [ ] Build recommendation system
- [ ] Add A/B testing framework

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- 🐛 Bug fixes and improvements
- 📊 Additional visualizations
- 🤖 New machine learning models
- 📝 Documentation enhancements
- ✨ UI/UX improvements

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Spotify for providing the audio features API
- Scikit-learn community for excellent ML tools
- Streamlit team for the amazing web framework
- All contributors and supporters

---

<div align="center">

**Made with Passion, by Anoushka**

*A portfolio project demonstrating end-to-end ML capabilities*

[⬆ Back to Top](#-song-popularity-predictor)

</div>
