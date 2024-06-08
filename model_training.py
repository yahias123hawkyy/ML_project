from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import pandas as pd

def trainTheModel():
    dataset = pd.read_csv('featured_data.csv')

    # Drop the 'Unnamed: 0' and 'CustomerID' columns if they exist
    if 'Unnamed: 0' in dataset.columns:
        dataset.drop(columns=['Unnamed: 0'], inplace=True)
    if 'CustomerID' in dataset.columns:
        dataset.drop(columns=['CustomerID'], inplace=True)

    # Convert 'Churn' column to binary (1 for 'Yes', 0 for 'No')
    dataset['Churn'] = dataset['Churn'].map({'Yes': 1, 'No': 0})

    # Define the features (X) and the target (y)
    X = dataset.drop(columns=['Churn'])
    y = dataset['Churn']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize the models
    log_reg = LogisticRegression(random_state=42)
    decision_tree = DecisionTreeClassifier(random_state=42)
    random_forest = RandomForestClassifier(random_state=42)
    gradient_boosting = GradientBoostingClassifier(random_state=42)

    # Train the models
    log_reg.fit(X_train, y_train)
    decision_tree.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)
    gradient_boosting.fit(X_train, y_train)

    # Define a function to evaluate the models
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        return accuracy, precision, recall, f1, roc_auc

    # Evaluate each model
    models = [log_reg, decision_tree, random_forest, gradient_boosting]
    model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']

    for model, name in zip(models, model_names):
        accuracy, precision, recall, f1, roc_auc = evaluate_model(model, X_test, y_test)
        print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    # Define a function for cross-validation
    def cross_validate_model(model, X, y):
        cv_accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_precision = cross_val_score(model, X, y, cv=5, scoring='precision')
        cv_recall = cross_val_score(model, X, y, cv=5, scoring='recall')
        cv_f1 = cross_val_score(model, X, y, cv=5, scoring='f1')
        cv_roc_auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        return cv_accuracy, cv_precision, cv_recall, cv_f1, cv_roc_auc

    # Perform cross-validation for each model
    for model, name in zip(models, model_names):
        cv_accuracy, cv_precision, cv_recall, cv_f1, cv_roc_auc = cross_validate_model(model, X, y)
        print(f"{name} - CV Accuracy: {cv_accuracy.mean():.4f}, CV Precision: {cv_precision.mean():.4f}, CV Recall: {cv_recall.mean():.4f}, CV F1-Score: {cv_f1.mean():.4f}, CV ROC-AUC: {cv_roc_auc.mean():.4f}")


