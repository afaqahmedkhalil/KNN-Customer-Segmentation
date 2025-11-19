# KNN Customer Segmentation Project


A complete end-to-end Machine Learning project following an industry checklist
This project applies K-Nearest-Neighbors (KNN) to segment customers based on:

* Gender
* Age
* Annual Income
* Spending Score

**The goal is to classify customers into segments:**
- Low
- Medium
- High

## Project Checklist

This project follows a full machine-learning pipeline:

## 1. Problem Understanding

Identify customer segments to support marketing, targeting, and business decisions.

## 2. Data Exploration (EDA)

* Dataset shape
* Missing values
* Distribution of features
* Class imbalance check 
* Outlier detection

## 3. Data Cleaning & Encoding

* Label encoding for Gender
* No missing values
* No duplicates
* Outliers reviewed but kept (KNN is sensitive — later handled)

## 4. Scaling

StandardScaler was applied to all numerical features.

## 5. Dimensionality Reduction

Not needed. 
Only 4 features.

## 6. Train-Test Split

80/20 split

Stratified to preserve class ratios

## 7. Class Balancing

### Original class counts:

| Segment | Count |
| ------- | ----- |
| Medium  | 94    |
| High    | 57    |
| Low     | 49    |

### After balancing (undersampling):

| Segment | Count |
| ------- | ----- |
| Low     | 75    |
| Medium  | 75    |
| High    | 75    |

## 8. Distance Metric & K Selection
```
Distance metric = Euclidean

Best K found = 1
(using validation search)

```

## 9. Model Training

### Trained KNN with:
```
n_neighbors = 1

metric = 'minkowski'

weights = 'distance'
```
## 10. Evaluation

####  Accuracy
```
Accuracy: 0.925
```

#### Confusion Matrix
```
[[11, 0, 0],
 [ 0, 9, 1],
 [ 0, 2, 17]]
```

#### Classification Report

| Class                | Precision | Recall | F1       | Support |
| -------------------- | --------- | ------ | -------- | ------- |
| High                 | 1.00      | 1.00   | 1.00     | 11      |
| Low                  | 0.82      | 0.90   | 0.86     | 10      |
| Medium               | 0.94      | 0.89   | 0.92     | 19      |
| **Overall Accuracy** | —         | —      | **0.93** | 40      |

``` ROC-AUC: 0.9467 ```

#### Feature Importance

(Applied using permutation importance)

| Feature        | Importance |
| -------------- | ---------- |
| Spending Score | **0.4675** |
| Annual Income  | 0.0900     |
| Age            | 0.0825     |
| Gender         | 0.0100     |

## Interpretation:

The Spending Score contributes almost 47% toward customer segmentation.

## Project Files
```
├── data/
│     └── Mall_Customers.csv
├── notebook/
│     └── knn_segmentation.ipynb
├── src/
│     ├── preprocessing.py
│     ├── model_training.py
│     ├── evaluation.py
│     └── utils.py
└── README.md
```

## Results Summary

```
KNN achieved 92.5% accuracy

Best K = 1

ROC-AUC = 0.947

Segmentation most influenced by Spending Score

Balanced dataset improved classification stability

Clear class separation achieved after scaling
```

## Conclusion

This project demonstrates a complete ML pipeline from raw data to model evaluation.

## KNN performed extremely well after:

* Balancing data
* Scaling
* Choosing optimal K
* Using distance-based weights
* It can be expanded by adding:
* DBSCAN clustering
* Visualization dashboards
* Deployment as a web API

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

## How to Run

Follow the steps below to run this project on your system:

1. Clone the repository:
```bash
git clone https://github.com/afaqahmedkhalil/KNN-Customer-Segmentation.git
```
2. Navigate into the project folder:

``` 
cd KNN-Customer-Segmentation
```

3. Install required dependencies:

``` 
pip install -r requirements.txt
```

4. Run preprocessing (optional step):

``` 
python src/preprocessing.py
```

5. Train the model:

``` 
python src/model_training.py
```

6. Evaluate the model:

  ``` 
  python src/evaluation.py
```
  
7. To view the analysis notebook:

``` 
jupyter notebook notebook/knn_segmentation.ipynb
```
