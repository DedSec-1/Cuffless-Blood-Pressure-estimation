Comparision Table for Diabetes prediction using biomedical signals

|S.No| Paper Name | Method Used | Database Details | Accuracy (performance)|
|----|------------|-------------|------------------|-------------|
|1| Classification of Diabetes Using Photoplethysmogram (PPG) Waveform Analysis: Logistic Regression Modeling | Logistic Regression | 587 Subjects = 459+128 -> PPG was used | 92.3% |
|2| Diabetic diagnose test based on PPG signal and identification system | ARMA (Auto Regression Moving Avg) | 24750 datapoints containing raw PPG and physical char. | 94% |
|3| Non-Invasive Classification of Blood Glucose Level for  Early Detection Diabetes Based on Photoplethysmography Signal | ARMA, SVM, DT, Ensemble based tech. | PPG + Sys & Dys BP with a few Physical traits | Ensembled bagged trees - 98% | 
|4| Design of intelligent diabetes mellitus detection system using hybrid feature selection based XGBoost classifier | KNN, SVM, RF and XGBoost | 217 patients ->   59.9 % normal, 32.7 % diabetic and 7.37 % prediabetic patients | 99.3%




Metrics for BP

| Model Name | Accuracy | MSE | RMSE | Var |
|-|-|-|-|-|
| Linear Regression | 81.99 | 486.355 | 22.053 | 0.03| 
| Random Forest | 92.977 | 102.4973 | 10.124 | 0.80 |
| Adaboost Regressor | 83.65 | 381.6853 | 19.5367 | 0.25|
| Ridge Regressor| 81.827 | 495.338 | 22.256 | 0.02 |
| Support Vector Regression | 83.053 | 477.639 | 21.854 | 0.05 |