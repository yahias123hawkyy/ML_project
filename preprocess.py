import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocessTheDataSet(dataset):

    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in dataset.columns:
        dataset.drop(columns=['Unnamed: 0'], inplace=True)

    numerical_features = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [
        'Gender', 'Service_Internet', 'Service_Phone', 'Service_TV',
        'Contract', 'PaymentMethod', 'StreamingMovies', 'StreamingMusic',
        'OnlineSecurity', 'TechSupport'
    ]

    for feature in numerical_features:
        dataset[feature].fillna(dataset[feature].median(), inplace=True)

    for feature in categorical_features:
        dataset[feature].fillna(dataset[feature].mode()[0], inplace=True)

    # Treat outliers
    def cap_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series.clip(lower_bound, upper_bound)

    for feature in numerical_features:
        dataset[feature] = cap_outliers(dataset[feature])

    # Encode categorical variables
    dataset_encoded = pd.get_dummies(dataset, columns=categorical_features, drop_first=True)

    # Normalize numerical features
    scaler = StandardScaler()
    dataset_encoded[numerical_features] = scaler.fit_transform(dataset_encoded[numerical_features])

    # Save cleaned dataset
    cleaned_dataset_path = 'cleaned_data.csv'
    dataset_encoded.to_csv(cleaned_dataset_path, index=False)

    print(f"Cleaned dataset saved to '{cleaned_dataset_path}'")

# Example usage
# dataset = pd.read_csv('your_dataset.csv')
# preprocessTheDataSet(dataset)
