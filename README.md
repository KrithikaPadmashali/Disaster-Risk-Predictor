# Disaster-Risk-Predictor
The Disaster Impact Predictor is a machine learning-based system designed to identify high-risk regions for disasters using environmental and demographic data. The project predicts risk levels and visualizes them on an interactive map.

##  Objectives
- Predict disaster risk levels using historical and environmental data  
- Classify regions into **Low, Medium, and High Risk**  
- Visualize results using an interactive **heatmap**  
- Build a scalable and modular ML pipeline  

---

## Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Folium (for map visualization)  
- Matplotlib / Seaborn (EDA)  
- Git & GitHub  

---

## 📊Input Features
The model uses the following features:
- Rainfall  
- Temperature  
- Population Density  
- Past Disaster Occurrences  
- Latitude & Longitude (for mapping)

---

##  Output
- Risk Score (0 = Low, 1 = Medium, 2 = High)  
- Interactive Risk Map (HTML Heatmap)

---
## Project Structure
```
Disaster-Risk-Predictor/
│
├── data/
│ ├── raw/
│ ├── processed/
│
├── notebooks/
│
├── src/
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── model.py
│ ├── visualization.py
│
├── outputs/
│ ├── maps/
│ ├── models/
│
├── README.md
└── requirements.txt

```
---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/KrithikaPadmashali/Disaster-Risk-Predictor.git
cd Disaster-Risk-Predictor


## 🏗️ Project Structure
