import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer

def interpretTheModel(best_random_forest, X_train, X_test):
    # Get feature importance from the Random Forest model
    feature_importances = best_random_forest.feature_importances_
    features = X_train.columns

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    }) 

    # Sort the DataFrame by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance from Random Forest')
    plt.show()

    # Create a LIME explainer
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=['Not Churn', 'Churn'],
        mode='classification'
    )

    # Select an instance from the test set to explain
    i = 0  # You can change this to any index you want to explain
    exp = explainer.explain_instance(
        data_row=X_test.iloc[i].values,
        predict_fn=best_random_forest.predict_proba
    )

    # Print the explanation
    exp.show_in_notebook(show_table=True, show_all=False)
    exp.as_pyplot_figure()
    plt.show()

