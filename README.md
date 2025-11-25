Predictive Maintenance using Machine Learning and Deep Learning

This repository contains two Google Colab notebooks implementing an integrated **Predictive Maintenance framework** using both **traditional Machine Learning (ML)** and **Deep Learning (DL)** models.  
The goal is to predict equipment failures and estimate Remaining Useful Life (RUL) across multiple industrial datasets.

## Files Overview
- **`ML_AI4I_Predictive_Maintenance.ipynb`**  
  - Implements Random Forest, XGBoost, Logistic Regression, and a basic Neural Network.  
  - Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI).  
  - Includes preprocessing, SMOTE balancing, model training, evaluation, and ensemble integration.

- **`DL_Predictive_Maintenance_LSTM_GRU.ipynb`**  
  - Implements LSTM and GRU architectures for time-series modeling.  
  - Datasets: AI4I 2020, NASA C-MAPSS, and IEEE PHM 2012.  
  - Includes preprocessing for time-series data, transfer learning (AI4I → PHM), and cross-domain evaluation.

## Datasets Used
- **AI4I 2020 Predictive Maintenance Dataset** – [UCI Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)  
- **NASA C-MAPSS Turbofan Engine Dataset** – [NASA Data Portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)  
- **IEEE PHM 2012 Bearing Dataset** – [Kaggle](https://www.kaggle.com/datasets/alanhabrony/ieee-phm-2012-data-challenge/data)


## Running the Notebooks
1. Open each notebook in [Google Colab](https://colab.research.google.com/).  
2. Upload or link datasets from the sources above.  
3. Run all cells sequentially — results (metrics, plots, and comparisons) appear at the end.

No external configuration is required; all dependencies are installed within the notebooks.


## Key Evaluation Metrics
- **Classification (AI4I, PHM):** Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC  
- **Regression (NASA C-MAPSS):** RMSE, MAE, R²  
- **Transfer Learning:** Fine-tuned evaluation  
- **Fairness:** False Positive/Negative Rates, Disparate Impact Ratio  
