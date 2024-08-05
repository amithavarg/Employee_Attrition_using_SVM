import logging
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from src.utils.utilities import configure_logging

logger = configure_logging()

def train_logistic_regression_model(x_train, y_train):
    try:
        logistic_model = LogisticRegression()
        logistic_model.fit(x_train, y_train)
        logger.info('Logistic Regression model trained successfully.')
        return logistic_model
    except Exception as error:
        logger.error(f"Error in train_logistic_regression_model: {error}")
        raise error

def train_svm_model(x_train, y_train, kernel_type='linear', degree=3):
    try:
        svm_model = SVC(kernel=kernel_type, degree=degree)
        trained_model = svm_model.fit(x_train, y_train)
        logger.info(f'SVM model with {kernel_type} kernel trained successfully.')
        return trained_model
    except Exception as error:
        logger.error(f"Error in train_svm_model: {error}")
        raise error
