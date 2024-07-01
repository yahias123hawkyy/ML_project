import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer

def interpretTheModel(best_random_forest, X_train, X_test):
    feature_importances = best_random_forest.feature_importances_
    features = X_train.columns

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    }) 

    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance from Random Forest')
    plt.show()

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=['Not Churn', 'Churn'],
        mode='classification'
    )

    
    i = 0  
    exp = explainer.explain_instance(
        data_row=X_test.iloc[i].values,
        predict_fn=best_random_forest.predict_proba
    )

    # Print the explanation
    exp.show_in_notebook(show_table=True, show_all=False)
    exp.as_pyplot_figure()
    plt.show()



# for i in range(3):  # Example: explain the first three instances
#     exp = explainer.explain_instance(
#         data_row=X_test.iloc[i].values,
#         predict_fn=best_random_forest.predict_proba
#     )
#     print(f"Explanation for instance {i}:")
#     exp.show_in_notebook(show_table=True, show_all=False)
#     exp.as_pyplot_figure()
#     plt.show()