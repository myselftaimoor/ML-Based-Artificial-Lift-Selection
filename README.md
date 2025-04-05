# Artificial Lift Optimization in Eagle Ford Shale Formation

This repository contains the implementation of a **predictive model** developed to optimize **artificial lift methods** in the **Eagle Ford Shale Formation**, with a focus on improving efficiency in unconventional oil wells. The model was trained to predict the most efficient lift method based on input parameters and well characteristics.

## ⭐ Overview

- **Goal**: Improve efficiency in unconventional oil wells by predicting the optimal artificial lift method using machine learning.
- **Accuracy**: Achieved **96% accuracy** with various machine learning models.
- **Best Model**: Random Forest provided the highest accuracy among models such as SVM, KNN, and Logistic Regression.

## 🧠 Methodology

1. **Data Preprocessing**: The dataset includes various well attributes, operational conditions, and performance metrics.
2. **Feature Engineering**: Key features include well depth, pressure, flow rate, and other geological factors.
3. **Model Selection**: Multiple models were evaluated, including:
    - **SVM (Support Vector Machine)**
    - **Random Forest**
    - **KNN (K-Nearest Neighbors)**
    - **Logistic Regression**
4. **Model Training**: The models were trained and validated using cross-validation.
5. **Evaluation**: The Random Forest model outperformed the others, achieving the highest accuracy of **96%**.

## 🏗️ Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/ai-lift-optimization.git
   cd ai-lift-optimization

2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

4. **Run the Model Training**
   ```bash
   python train_model.py

5. **Test the Model**
   ```bash
   python test_model.py

## 🔍 Results
- **Random Forest Model:** Achieved 96% accuracy in predicting optimal artificial lift methods.

- **Model Performance:** The final model significantly improved efficiency in selecting the correct artificial lift method for various well conditions.

## 📂 Directory Structure

train_model.py: Code for training the model

test_model.py: Code for evaluating the trained model

data/: Contains raw and processed datasets

models/: Stores trained machine learning models

requirements.txt: List of dependencies

Canvas/: Link to access complete project code and documentation

## 🛠️ Tech Stack

Programming Language: Python

Machine Learning Libraries: Scikit-learn, NumPy, Pandas

Data Preprocessing: Pandas, NumPy

Model Evaluation: Cross-validation, GridSearchCV

Development Environment: Jupyter Notebook (for exploration)

## 🤝 Contributing

Feel free to fork this repository and submit pull requests if you'd like to contribute improvements, optimizations, or suggestions.

## 📄 License

This project is licensed under the MIT License.
