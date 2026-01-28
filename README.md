# Neural Approaches for Classification: From Perceptron Theory to Sonar Detection

This repository explores the fundamentals of neural computation through the implementation of supervised learning algorithms. It covers the theoretical study of linear classifiers and their practical application on real-world acoustic data.

## 📂 Project Structure

The project is divided into two main notebooks:

### 1. Perceptron Fundamentals (`1_Perceptron_Theory.ipynb`)
This notebook focuses on the algorithmic behavior of the Perceptron using synthetic data generated via a **Teacher-Student** framework.
* **Algorithms:** Implementation of **Online Perceptron** (Stochastic) vs. **Batch Perceptron**.
* **Convergence Analysis:** Evaluation of the "Overlap" ($R$) between the student and teacher weight vectors.
* **Visualization:** Graphical plotting of decision boundaries and error rates relative to the number of dimensions ($N$) and training examples ($P$).

### 2. Sonar Signal Classification (`2_Sonar_Classification.ipynb`)
A transition to real-world data using the **Sonar Dataset**, which consists of 208 patterns (111 metal cylinders and 97 rocks) obtained by bouncing sonar signals off surfaces at various angles.
* **Data Engineering:** Custom parsing of sonar signal files and binary label encoding.
* **Model Optimization:** Implementation of a **Batch Perceptron** with robustness features.
* **Generalization Techniques:** * **Early Stopping:** Monitoring validation error to prevent overfitting.
    * **Geometric Margin ($\gamma$):** Analysis of the stability of the solution.
    * **Performance:** Achieved an average test error ($E_g$) demonstrating the model's ability to classify unseen sonar signals effectively.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** `NumPy` (linear algebra), `Matplotlib` (visualizations), `Pandas` & `re` (data processing).

## 🚀 Key Learning Outcomes
* Mastery of gradient-based optimization and the Perceptron learning rule.
* Ability to differentiate between Online and Batch learning dynamics.
* Understanding the **Bias-Variance tradeoff** through Early Stopping and Validation sets.
* Experience in processing and normalizing real-world signal data for neural networks.

## 📊 Results Summary
The study demonstrates that while the Perceptron is a simple linear model, when combined with proper **stability analysis** and **early stopping**, it provides a robust baseline for binary classification tasks even on complex datasets like the Sonar signals.
