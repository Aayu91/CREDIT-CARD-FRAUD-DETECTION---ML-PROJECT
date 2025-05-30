Credit Card Fraud Detection :

This project implements a machine learning-based system to detect fraudulent transactions in credit card data using Logistic Regression. It addresses class imbalance with undersampling and evaluates the model's performance on both training and test datasets.

📌 Project Overview :

Credit card fraud is a growing issue in the digital age. The goal of this project is to build a binary classification model that can predict whether a given credit card transaction is fraudulent or legitimate.

📂 Dataset: 

   * The dataset used is Credit Card Fraud Detection from Kaggle.

   * It contains transactions made by European cardholders in September 2013.

   * Features are numerical (due to PCA transformation), and the Class column indicates fraud (1) or legitimate (0).

⚙️ Technologies & Libraries:

  (i) Python

  (ii) Pandas & NumPy

  (iii) Scikit-learn

  (iv) Matplotlib (optional)

🧠 Model & Workflow:

1. Data Preprocessing
      Load dataset and check for null values

      Analyze class distribution

      Balance the dataset using undersampling

2. Feature Selection
  Separate input features (X) and output label (Y)

3. Model Training
  Train a LogisticRegression model from Scikit-learn

  Use train_test_split to divide data into 80% training and 20% testing

4. Evaluation
   Compute training and test accuracy using accuracy_score

    Evaluate model performance using:

    Accuracy

    Precision (optional if added in later code)

📊 Results:
      Accuracy on training and test sets is printed

      Performance may vary depending on undersampling size

🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
Install required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Download the dataset and place it in the appropriate path (e.g., /content/creditcard.csv or modify the notebook accordingly).

Run the notebook:

bash
Copy
Edit
jupyter notebook credit_card_fraud_detection.ipynb
📝 Notes
    Dataset is highly imbalanced, so care is taken to balance it before training.

    Logistic Regression is a baseline model; other models like Random Forest or XGBoost can be explored for improvement.

📈 Future Improvements:

   Implement other classifiers (e.g., Random Forest, SVM)

   Use oversampling techniques like SMOTE

  Use ROC-AUC and confusion matrix for detailed evaluation

📄 License:

This project is licensed under the MIT License.
