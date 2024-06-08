import pandas as pd
from sklearn.preprocessing import StandardScaler

def featureEngineeringDataSet():
    # Load the cleaned dataset
    dataset = pd.read_csv('cleaned_data.csv')

    # Verify the columns in the dataset
    print("Dataset columns:", dataset.columns)

    # Feature Engineering: Create new features
    
    
    
    
    # 1. Total Service Usage
    # Combines service-related columns to create a feature representing the total number of services a customer subscribes to.
    dataset['TotalServices'] = (
        dataset['Service_Internet_Fiber optic'].astype(int) +
        dataset['Service_Phone_Yes'].astype(int) +
        dataset['Service_TV_Yes'].astype(int)
    )

    # 2. Monthly Charges per Service
    # Calculates the average monthly charges per service. This helps in identifying customers who might be paying disproportionately higher amounts.
    dataset['MonthlyChargesPerService'] = dataset['MonthlyCharges'] / dataset['TotalServices']
    dataset['MonthlyChargesPerService'].replace([float('inf'), -float('inf')], 0, inplace=True)
    dataset['MonthlyChargesPerService'].fillna(0, inplace=True)

    # 3. Tenure per Service
    # Calculates the average tenure per service. This provides insights into customer loyalty.
    dataset['TenurePerService'] = dataset['Tenure'] / dataset['TotalServices']
    dataset['TenurePerService'].replace([float('inf'), -float('inf')], 0, inplace=True)
    dataset['TenurePerService'].fillna(0, inplace=True)

    # Normalize the numerical features (including the new features)
    numerical_features = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'MonthlyChargesPerService', 'TenurePerService']
    scaler = StandardScaler()
    dataset[numerical_features] = scaler.fit_transform(dataset[numerical_features])

    # Save the enhanced dataset to a new CSV file
    enhanced_dataset_path = 'featured_data.csv'
    dataset.to_csv(enhanced_dataset_path, index=False)

    print(f"featured_data dataset saved to {enhanced_dataset_path}")