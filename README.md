# Neural Network Classification with PySpark

This project applies a neural network model using PySpark to perform classification on structured data.
It was developed in the context of a **MAS Data Science programme** and focuses on scalable data processing and machine learning.

---

## 📌 Overview

The project demonstrates how to:

* preprocess large datasets using PySpark
* handle missing values and feature transformations
* build feature vectors for machine learning
* train a neural network model (Multilayer Perceptron)
* evaluate model performance

The workflow follows a typical data science pipeline from raw data to model evaluation.

---

## 📊 Data

The dataset is stored in **Parquet format**:

* `ne3.parquet`

It contains multiple numerical variables (`UW10`–`UW24`) and a timestamp column (`ts`), from which the **year** is derived and used as the target variable.

---

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

* Extracting **year** from timestamp
* Handling missing values using **mean imputation**
* Rounding and cleaning numerical values
* Converting all features to `double` type
* Removing unnecessary columns (e.g. timestamp)

---

## 🔧 Feature Engineering

* All variables (`UW10`–`UW24`) are combined into a feature vector using:

```python
VectorAssembler
```

* The target variable (`year`) is converted into labels using:

```python
StringIndexer
```

---

## 🤖 Model

The model used is a **Multilayer Perceptron Classifier (MLP)** from PySpark:

* Input layer: 15 features
* Hidden layer: 1000 neurons
* Output layer: 8 classes

```python
layers = [15, 1000, 8]
```

---

## 🔀 Train/Test Split

The dataset is split into:

* 70% training data
* 30% test data

---

## 📈 Evaluation

Model performance is evaluated using:

* **Accuracy metric** via `MulticlassClassificationEvaluator`

```python
metricName = "accuracy"
```

---

## 🧠 Key Learning Points

* Scalable data processing with PySpark
* Handling missing data in distributed systems
* Feature engineering for machine learning
* Neural networks in Spark MLlib
* End-to-end ML pipeline design

---

## 📂 Project Structure

```id="pjx9tx"
.
├── 1_preprocess_data.ipynb
├── 2_neural_network.py
├── ne3.parquet
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install pyspark pandas matplotlib
```

---

### 2. Run preprocessing (optional)

```bash
jupyter notebook
```

Open:

```
1_preprocess_data.ipynb
```

---

### 3. Run model

```bash
python 2_neural_network.py
```

---

## ⚠️ Notes

* The dataset path must be adjusted:

```python
spark.read.parquet('/ne3.parquet')
```

* Large datasets may require more memory or a distributed setup

---

## 📬 Contact

For questions or collaboration, feel free to reach out.
