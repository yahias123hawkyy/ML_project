import pandas as pd
from sklearn.preprocessing import StandardScaler

def featureEngineeringDataSet():
    dataset = pd.read_csv('cleaned_data.csv')

    print("Dataset columns:", dataset.columns)
    # 1. Total Service Usage
    dataset['TotalServices'] = (
        dataset['Service_Internet_Fiber optic'].astype(int) +
        dataset['Service_Phone_Yes'].astype(int) +
        dataset['Service_TV_Yes'].astype(int)
    )

    # 2. Monthly Charges per Service
    dataset['MonthlyChargesPerService'] = dataset['MonthlyCharges'] / dataset['TotalServices']
    dataset['MonthlyChargesPerService'].replace([float('inf'), -float('inf')], 0, inplace=True)
    dataset['MonthlyChargesPerService'].fillna(0, inplace=True)

    # 3. Tenure per Service
    dataset['TenurePerService'] = dataset['Tenure'] / dataset['TotalServices']
    dataset['TenurePerService'].replace([float('inf'), -float('inf')], 0, inplace=True)
    dataset['TenurePerService'].fillna(0, inplace=True)

    numerical_features = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'MonthlyChargesPerService', 'TenurePerService']
    scaler = StandardScaler()
    dataset[numerical_features] = scaler.fit_transform(dataset[numerical_features])

    enhanced_dataset_path = 'featured_data.csv'
    dataset.to_csv(enhanced_dataset_path, index=False)

    print(f"featured_data dataset saved to {enhanced_dataset_path}")