import pandas as pd
from src.data.data_loader import load_and_prepare_data, split_data
from src.model.model_training import train_logistic_regression_model, train_svm_model
from src.data.data_visualization import visualize_numeric_data, visualize_categorical_data, visualize_means_grouped_by_attrition
from src.model.model_evaluation import assess_model
from src.utils.utilities import configure_logging, save_figure

def main():
    logger = configure_logging()
    
    try:
        # Load and preprocess the data       
        dataframe, raw_dataframe, numerical_columns, categorical_columns = load_and_prepare_data('data/HR_Employee_Attrition.xlsx')
        
        visualize_numeric_data(raw_dataframe, numerical_columns)
        visualize_categorical_data(raw_dataframe, categorical_columns)
        visualize_means_grouped_by_attrition(raw_dataframe, numerical_columns)
        
        # Split the data
        x_train, x_test, y_train, y_test = split_data(dataframe)
        
        # Train Logistic Regression model
        logistic_model = train_logistic_regression_model(x_train, y_train)
        assess_model(logistic_model, x_train, x_test, y_train, y_test, 'logistic')
        
        # Train SVM model with different kernels
        for kernel_type in ['linear', 'rbf', 'poly']:
            svm_model = train_svm_model(x_train, y_train, kernel_type=kernel_type)
            print(f"Evaluating SVM with {kernel_type} kernel:")
            assess_model(svm_model, x_train, x_test, y_train, y_test, kernel_type)
    
    except Exception as error:
        print(f"An error occurred: {error}")

if __name__ == "__main__":
    main()
