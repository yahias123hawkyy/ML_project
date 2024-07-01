import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency

def exploreInDepthDataSet(dataset):
    # Remove the first column if it's unnamed
    if dataset.columns[0].startswith('Unnamed'):
        dataset = dataset.drop(dataset.columns[0], axis=1)

    # Display the first few rows of the dataset
    print(dataset.head())

    # Display the summary of the dataset
    print(dataset.info())

    # Print the count of missing values in each column
    print(dataset.isnull().sum())

    # Describe the dataset to get summary statistics
    print(dataset.describe())

    # Get the column names
    columns = dataset.columns

    # Describe each feature
    for column in columns:
        print(f"Feature: {column}")
        print(dataset[column].value_counts())
        print("\n")

    # Visualize distributions with histograms
    numerical_features = dataset.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = dataset.select_dtypes(include=['object', 'bool']).columns

    # Visualize numerical feature distributions with histograms and box plots
    for feature in numerical_features:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        
        sns.histplot(dataset[feature], kde=True, ax=axes[0])
        axes[0].set_title(f'Distribution of {feature}')
        
        sns.boxplot(x=dataset[feature], ax=axes[1])
        axes[1].set_title(f'Box plot of {feature}')
        
        plt.show()

    # Visualize categorical feature distributions with bar charts
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=dataset[feature], order=dataset[feature].value_counts().index)
        plt.title(f'Distribution of {feature}')
        plt.show()

    # Pair plot for numerical features
    sns.pairplot(dataset[numerical_features])
    plt.show()

    # Correlation matrix heatmap
    correlation_matrix = dataset[numerical_features].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

    # Hypothesis testing
    target_variable = 'Churn'  # Example target variable for testing

    if target_variable in dataset.columns:
        for feature in numerical_features:
            if feature != target_variable and dataset[target_variable].nunique() == 2:
                groups = dataset[target_variable].unique()
                group1 = dataset[dataset[target_variable] == groups[0]][feature]
                group2 = dataset[dataset[target_variable] == groups[1]][feature]
                t_stat, p_val = ttest_ind(group1, group2, nan_policy='omit')
                print(f'T-test for {feature} by {target_variable}: t-stat={t_stat:.4f}, p-value={p_val:.4f}')

        for feature in categorical_features:
            if feature != target_variable:
                contingency_table = pd.crosstab(dataset[feature], dataset[target_variable])
                chi2, p_val= chi2_contingency(contingency_table)
                print(f'Chi-square test for {feature} by {target_variable}: chi2={chi2:.4f}, p-value={p_val:.4f}')