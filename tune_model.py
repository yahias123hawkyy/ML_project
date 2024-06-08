import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def tuneTheModel():
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

    # Evaluate each default model
    default_models = [log_reg, decision_tree, random_forest, gradient_boosting]
    default_model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']

    print("Default Model Performance:")
    for model, name in zip(default_models, default_model_names):
        accuracy, precision, recall, f1, roc_auc = evaluate_model(model, X_test, y_test)
        print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    # Hyperparameter Tuning

    # Logistic Regression
    param_grid_log_reg = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

    grid_search_log_reg = GridSearchCV(estimator=LogisticRegression(random_state=42), param_grid=param_grid_log_reg, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search_log_reg.fit(X_train, y_train)
    best_log_reg = grid_search_log_reg.best_estimator_
    print(f"\nBest hyperparameters for Logistic Regression: {grid_search_log_reg.best_params_}")

    # Decision Tree
    param_grid_decision_tree = {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search_decision_tree = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid_decision_tree, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search_decision_tree.fit(X_train, y_train)
    best_decision_tree = grid_search_decision_tree.best_estimator_
    print(f"\nBest hyperparameters for Decision Tree: {grid_search_decision_tree.best_params_}")

    # Random Forest
    param_grid_random_forest = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search_random_forest = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid_random_forest, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search_random_forest.fit(X_train, y_train)
    best_random_forest = grid_search_random_forest.best_estimator_
    print(f"\nBest hyperparameters for Random Forest: {grid_search_random_forest.best_params_}")

    # Gradient Boosting
    param_grid_gradient_boosting = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.001, 0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search_gradient_boosting = GridSearchCV(estimator=GradientBoostingClassifier(random_state=42), param_grid=param_grid_gradient_boosting, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search_gradient_boosting.fit(X_train, y_train)
    best_gradient_boosting = grid_search_gradient_boosting.best_estimator_
    print(f"\nBest hyperparameters for Gradient Boosting: {grid_search_gradient_boosting.best_params_}")

    # Evaluate each tuned model
    tuned_models = [best_log_reg, best_decision_tree, best_random_forest, best_gradient_boosting]
    tuned_model_names = ['Tuned Logistic Regression', 'Tuned Decision Tree', 'Tuned Random Forest', 'Tuned Gradient Boosting']

    print("\nTuned Model Performance:")
    for model, name in zip(tuned_models, tuned_model_names):
        accuracy, precision, recall, f1, roc_auc = evaluate_model(model, X_test, y_test)
        print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    # Perform cross-validation for each tuned model
    def cross_validate_model(model, X, y):
        cv_accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_precision = cross_val_score(model, X, y, cv=5, scoring='precision')
        cv_recall = cross_val_score(model, X, y, cv=5, scoring='recall')
        cv_f1 = cross_val_score(model, X, y, cv=5, scoring='f1')
        cv_roc_auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        return cv_accuracy, cv_precision, cv_recall, cv_f1, cv_roc_auc

    print("\nCross-Validation Performance:")
    for model, name in zip(tuned_models, tuned_model_names):
        cv_accuracy, cv_precision, cv_recall, cv_f1, cv_roc_auc = cross_validate_model(model, X, y)
        print(f"{name} - CV Accuracy: {cv_accuracy.mean():.4f}, CV Precision: {cv_precision.mean():.4f}, CV Recall: {cv_recall.mean():.4f}, CV F1-Score: {cv_f1.mean():.4f}, CV ROC-AUC: {cv_roc_auc.mean():.4f}")

    return best_random_forest, X_train, X_test

# Run the model tuning function
# best_random_forest, X_train, X_test = tuneTheModel()
