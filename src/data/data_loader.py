import pandas as pd
from src.utils.utilities import configure_logging

logger = configure_logging()

def load_and_prepare_data(filepath):
    try:
        # Read the dataset
        dataframe = pd.read_excel(filepath)
        logger.info('Data loaded successfully.')
        
        print(dataframe.sample(5))
        
        # Dropping unnecessary columns
        dataframe = dataframe.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1)
        logger.info('Unnecessary columns dropped.')
        
        print(dataframe.info())
        
        # Creating numerical and categorical columns
        numerical_columns = [
            'DailyRate', 'Age', 'DistanceFromHome', 'MonthlyIncome', 'MonthlyRate',
            'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'NumCompaniesWorked',
            'HourlyRate', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
            'TrainingTimesLastYear'
        ]
        categorical_columns = [
            'Attrition', 'OverTime', 'BusinessTravel', 'Department', 'Education', 'EducationField',
            'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'StockOptionLevel', 
            'Gender', 'PerformanceRating', 'JobInvolvement', 'JobLevel', 'JobRole', 'MaritalStatus', 
            'RelationshipSatisfaction'
        ]
        
        original_dataframe = dataframe
        
        # Creating list of columns to create dummy variables for
        dummy_columns = ['BusinessTravel', 'Department','Education', 
                         'EducationField','EnvironmentSatisfaction', 
                         'Gender', 'JobInvolvement','JobLevel', 
                         'JobRole', 'MaritalStatus']

        # Creating dummy variables
        dataframe = pd.get_dummies(data=dataframe, columns=dummy_columns, drop_first=True)
        logger.info('Dummy variables created.')
        
        # Mapping overtime and attrition
        overtime_map = {'Yes': 1, 'No': 0}
        attrition_map = {'Yes': 1, 'No': 0}
        
        dataframe['OverTime'] = dataframe.OverTime.map(overtime_map)
        dataframe['Attrition'] = dataframe.Attrition.map(attrition_map)
        logger.info('OverTime and Attrition mapped.')
        
        return dataframe, original_dataframe, numerical_columns, categorical_columns
    except Exception as error:
        logger.error(f"Error in load_and_prepare_data: {error}")
        
        raise error

def split_data(dataframe):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    try:
        target = dataframe.Attrition
        features = dataframe.drop(columns=['Attrition'])
        
        # Scaling the data
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        scaled_features = pd.DataFrame(scaled_features, columns=features.columns)
        logger.info('Data scaled successfully.')
        
        # Splitting the data
        x_train, x_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=1, stratify=target)
        logger.info('Data split into train and test sets.')
        
        return x_train, x_test, y_train, y_test
    except Exception as error:
        logger.error(f"Error in divide_data: {error}")
        raise error
