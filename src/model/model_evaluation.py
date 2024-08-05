import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from src.utils.utilities import configure_logging, save_figure

logger = configure_logging()

def display_metrics(actual, predicted, model_name, dataset_type):
    try:
        # Generate the confusion matrix
        conf_matrix = confusion_matrix(actual, predicted)
        plt.figure(figsize=(8, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='.2f', xticklabels=['Not Attrite', 'Attrite'], yticklabels=['Not Attrite', 'Attrite'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        
        print(classification_report(actual, predicted))
                
        # Save the confusion matrix as an image
        figure = plt.gcf()  # Get the current figure
        save_figure(figure, f'{model_name}_{dataset_type}_confusion_matrix.png')

        # Show the plot
        plt.show()
        
    except Exception as error:
        logger.error(f"Error in display_metrics: {error}")
        raise error

def assess_model(model, x_train, x_test, y_train, y_test, model_name):
    try:
        # Evaluate on the training set
        y_train_pred = model.predict(x_train)
        print("Training Performance:")
        display_metrics(y_train, y_train_pred, model_name, 'Training')
        
        # Evaluate on the test set
        y_test_pred = model.predict(x_test)
        print("Test Performance:")
        display_metrics(y_test, y_test_pred, model_name, 'Test')
        
        logger.info('Model evaluation completed.')
    except Exception as error:
        logger.error(f"Error in assess_model: {error}")
        raise error
