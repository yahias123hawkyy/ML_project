import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler

def exploreTheDataSet(dataset):
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

    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(dataset[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.show()

    # Visualize distributions with box plots
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=dataset[feature])
        plt.title(f'Box plot of {feature}')
        plt.show()

    # Scatter plot for numerical features
    sns.pairplot(dataset[numerical_features])
    plt.show()

    # Correlation matrix
    correlation_matrix = dataset[numerical_features].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()
    
    
    
    
    
    
    
