# 🌸 Flower Classification

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-93.3%25-brightgreen?style=for-the-badge)

> Classifying iris flower species using **Logistic Regression** and **Decision Tree** based on sepal and petal measurements.  
> Built as part of the **SyntecxHub Machine Learning Track — Week 1 Project**.

---

## 📌 Project Overview

This project builds and compares two classification models to predict the species of an iris flower (Setosa, Versicolor, or Virginica) based on four physical measurements: sepal length, sepal width, petal length, and petal width.

| Metric | Value |
|--------|-------|
| **Best Model** | Logistic Regression |
| **Accuracy** | 0.9333 |
| **Dataset** | Scikit-Learn — Iris Dataset |
| **Total Samples** | 150 |
| **Test Samples** | 15 |
| **Features** | 4 |
| **Classes** | 3 (Setosa, Versicolor, Virginica) |

---

## 📂 Project Structure

```
Flower-Classification/
│
├── main.py # Main pipeline: EDA, training, evaluation
├── predict.py # Load model & make predictions on new input
├── logistic_model.pkl # Saved trained Logistic Regression model
└── README.md # You are here
```

---

## 🔄 Pipeline

```
Load Data → EDA (Pairplot) → Train/Test Split → Train Models
    → Evaluate & Compare → Confusion Matrix → Save Best Model → Predict
```

### Step-by-step:

**1. Load & Explore**
```python
x, y = datasets.load_iris(return_X_y=True)
print(x.shape) # (150, 4)
print(y.shape) # (150,)
```

**2. Exploratory Data Analysis (EDA)**
```python
df = pd.DataFrame(x, columns=datasets.load_iris().feature_names)
df['species'] = y

sns.pairplot(df, hue='species')
plt.show()
```
Visualizing feature pairs shows that **Setosa** is linearly separable from the other two species across all features, while **Versicolor** and **Virginica** show some overlap, especially in petal measurements.

**3. Train/Test Split**
```python
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.1, random_state=42, stratify=y
)
# Train: 135 | Test: 15
```

**4. Train & Compare 2 Models**
```python
# Model 1: Logistic Regression
model = linear_model.LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

# Model 2: Decision Tree
dt_model = tree.DecisionTreeClassifier()
dt_model.fit(x_train, y_train)
```

**5. Confusion Matrix**
```python
cm = metrics.confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")
```

---

## 📊 Results

```
Logistic Regression:
  Accuracy : 0.9333

Decision Tree:
  Accuracy : 0.8667
```

**Confusion Matrix (Logistic Regression):**
```
[[5 0 0]
 [0 4 1]
 [0 0 5]]
```

The **Logistic Regression model** outperformed the Decision Tree, correctly classifying 14 out of 15 test samples. All Setosa samples were classified perfectly (a fully separable class), while the single misclassification occurred between Versicolor and Virginica — the two species with overlapping feature distributions.

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/Flower-Classification.git
cd Flower-Classification
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Train the model
```bash
python main.py
```

### 4. Make predictions
```bash
python predict.py
```

---

## 🧰 Tech Stack

- **Python 3.11**
- **Pandas** — Data loading & manipulation
- **NumPy** — Numerical operations
- **Scikit-Learn** — ML models & evaluation
- **Matplotlib & Seaborn** — Data visualization
- **Pickle** — Model serialization

---

## 📈 Key Insights

- **Setosa** is perfectly separable from the other two species using petal length/width alone
- **Logistic Regression (93.3%)** outperformed **Decision Tree (86.7%)** on this small test set
- The only misclassification was between **Versicolor** and **Virginica**, the two most visually similar species
- A `LogisticRegression` model with `max_iter=200` was sufficient for convergence

---

## 🔮 Example Prediction

```python
# New flower with measurements: 5.1, 3.5, 1.4, 0.2
Predicted Species: setosa
```

---

## 👤 Author

**Mazen Mohamed**  
Machine Learning Trainee — SyntecxHub  
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat&logo=github)](https://github.com/MazenMohamed20)

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
