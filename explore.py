import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler

def exploreTheDataSet(dataset):
    if dataset.columns[0].startswith('Unnamed'):
        dataset = dataset.drop(dataset.columns[0], axis=1)

    print(dataset.head())

    print(dataset.info())

    print(dataset.isnull().sum())

    print(dataset.describe())

    columns = dataset.columns

    for column in columns:
        print(f"Feature: {column}")
        print(dataset[column].value_counts())
        print("\n")

    numerical_features = dataset.select_dtypes(include=['float64', 'int64']).columns

    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(dataset[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.show()

    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=dataset[feature])
        plt.title(f'Box plot of {feature}')
        plt.show()

    sns.pairplot(dataset[numerical_features])
    plt.show()

    correlation_matrix = dataset[numerical_features].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()
    
    
    
    
    
    
    
