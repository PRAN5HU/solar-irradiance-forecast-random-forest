import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
import joblib

# Function to load and preprocess data
def load_and_preprocess_data(train_file, test_file):
    try:
        # Load training data
        train_df = pd.read_excel(train_file, engine='openpyxl')

        # Load testing data
        test_df = pd.read_excel(test_file, engine='openpyxl')

        # Convert 'Date' column to datetime format
        train_df['Date'] = pd.to_datetime(train_df['Date'], format='%d-%m-%Y %H:%M:%S')
        test_df['Date'] = pd.to_datetime(test_df['Date'], format='%d-%m-%Y %H:%M:%S')

        # Feature engineering (example: extracting hour of the day)
        train_df['Hour'] = train_df['Date'].dt.hour
        test_df['Hour'] = test_df['Date'].dt.hour

        # Extract features and target variable for training and testing
        X_train = train_df[['AT', 'RH', 'WS', 'Hour']]
        y_train = train_df['GHI']

        X_test = test_df[['AT', 'RH', 'WS', 'Hour']]
        y_test = test_df['GHI']

        return X_train, X_test, y_train, y_test, test_df

    except Exception as e:
        print(f"Error loading or preprocessing data: {str(e)}")
        return None, None, None, None, None

# Function to normalize data
def normalize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    return X_train_normalized, X_test_normalized, scaler

# Function to train Random Forest model using GridSearchCV
def train_random_forest(X_train, y_train):
    try:
        # Define the model
        model = RandomForestRegressor(random_state=42)

        # Define parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        # Perform GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                   cv=5, scoring='r2', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Get best model from GridSearchCV
        best_model = grid_search.best_estimator_

        # Print best parameters found by GridSearchCV
        print(f'Best parameters found: {grid_search.best_params_}')

        return best_model

    except Exception as e:
        print(f"Error training Random Forest model: {str(e)}")
        return None

# Function to adjust predictions based on time of day and other factors
def adjust_predictions(y_pred, test_df):
    try:
        # Adjust predictions based on time of day (diurnal variation)
        hour = test_df['Date'].dt.hour
        morning = (hour >= 5) & (hour < 6)  # 5:00 AM to 6:00 AM
        early_morning = (hour >= 6) & (hour < 7)  # 6:00 AM to 7:00 AM
        morning_peak = (hour >= 7) & (hour < 7.5)  # 7:00 AM to 7:30 AM
        noon_peak = (hour >= 10) & (hour <= 13)  # 10:00 AM to 1:00 PM
        afternoon_decrease = (hour >= 14) & (hour < 15)  # 2:00 PM to 3:00 PM

        # Adjust predictions accordingly
        y_pred[morning] *= 1.1  # Adjusted prediction for early morning
        y_pred[early_morning] *= 1.2  # Adjusted prediction for morning
        y_pred[morning_peak] *= 1.5  # Adjusted prediction for morning peak
        y_pred[noon_peak] *= 0.8  # Adjusted prediction for noon peak
        y_pred[afternoon_decrease] *= 0.9  # Adjusted prediction for afternoon decrease

        return y_pred

    except Exception as e:
        print(f"Error adjusting predictions: {str(e)}")
        return None

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test, test_df):
    try:
        # Make predictions
        y_pred = model.predict(X_test)

        # Adjust predictions based on time of day and other factors
        y_pred = adjust_predictions(y_pred, test_df)

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Perform t-test (example: if needed)
        t_stat, p_value = ttest_ind(y_test, y_pred)

        # Print evaluation metrics
        print(f'R-squared: {r2}')
        print(f'Mean Squared Error: {mse}')
        print(f'Mean Absolute Error: {mae}')
        print(f'T-test results: T-statistic = {t_stat}, p-value = {p_value}')

        return y_pred, r2, mse, mae

    except Exception as e:
        print(f"Error evaluating model: {str(e)}")
        return None, None, None, None

# Function to save the model
def save_model(model, filepath):
    try:
        joblib.dump(model, filepath)
        print(f"Model saved successfully as {filepath}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

# Function to export actual and predicted data to Excel
def export_predictions_to_excel(test_df, y_test, y_pred, filepath):
    try:
        # Create a DataFrame with actual and predicted values
        results_df = pd.DataFrame({
            'Date': test_df['Date'],
            'Actual_GHI': y_test,
            'Predicted_GHI': y_pred
        })

        # Export DataFrame to Excel
        results_df.to_excel(filepath, index=False)
        print(f"Predictions exported successfully to {filepath}")

    except Exception as e:
        print(f"Error exporting predictions to Excel: {str(e)}")

# Main function to execute the entire process
def main(train_file, test_file):
    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test, test_df = load_and_preprocess_data(train_file, test_file)

        if X_train is not None and X_test is not None and y_train is not None and y_test is not None and test_df is not None:
            # Normalize data
            X_train_normalized, X_test_normalized, scaler = normalize_data(X_train, X_test)

            # Train Random Forest model
            best_model = train_random_forest(X_train_normalized, y_train)

            if best_model is not None:
                # Evaluate model performance
                y_pred, r2, mse, mae = evaluate_model(best_model, X_test_normalized, y_test, test_df)

                # Save the best model
                save_model(best_model, 'best_random_forest_model.pkl')

                # Export predictions to Excel
                export_predictions_to_excel(test_df, y_test, y_pred, 'actual_vs_predicted_GHI.xlsx')

                # Optionally, plot actual vs predicted values
                plt.figure(figsize=(10, 6))
                plt.plot(test_df['Date'], y_test, marker='', color='blue', linewidth=2, label='Actual GHI')
                plt.plot(test_df['Date'], y_pred, marker='', color='red', linewidth=2, linestyle='dashed',
                         label='Predicted GHI')
                plt.xlabel('Date')
                plt.ylabel('GHI')
                plt.title('Actual vs Predicted GHI')
                plt.legend()
                plt.grid(True)
                plt.show()

        else:
            print("Error: Failed to load or preprocess data.")

    except Exception as e:
        print(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    # Paths to training and testing data
    train_file = r"your_train_data.xlsx"  # Provide your actual file path here
    test_file = r"your_test_data.xlsx"    # Provide your actual file path here
    main(train_file, test_file)
