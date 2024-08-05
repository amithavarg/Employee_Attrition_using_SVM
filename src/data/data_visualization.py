import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.utilities import configure_logging, save_figure

logger = configure_logging()

def visualize_numeric_data(dataframe, numerical_columns):
    """
    Visualizes numerical data by creating summary statistics, histograms,
    and a correlation heatmap.
    """
    # Checking summary statistics
    summary_statistics = dataframe[numerical_columns].describe().T

    # Creating histograms
    dataframe[numerical_columns].hist(figsize=(14, 14))
    plt.suptitle('Histograms of Numerical Columns')
    histogram_figure = plt.gcf()
    save_figure(histogram_figure, 'numerical_histograms.png')
    plt.show()

    # Plotting the correlation between numerical variables
    plt.figure(figsize=(15, 8))
    sns.heatmap(dataframe[numerical_columns].corr(), annot=True, fmt='0.2f', cmap='YlGnBu')
    plt.title('Correlation Heatmap of Numerical Variables')
    correlation_figure = plt.gcf()
    save_figure(correlation_figure, 'correlation_heatmap.png')
    plt.show()

def visualize_categorical_data(dataframe, categorical_columns):
    """
    Visualizes categorical data by printing percentages of sub-categories
    and creating bar plots showing the percentage of attrition.
    """
    for column in categorical_columns:
        print(dataframe[column].value_counts(normalize=True))
        print('*' * 40)

    for column in categorical_columns:
        if column != 'Attrition':
            plot = (pd.crosstab(dataframe[column], dataframe['Attrition'], normalize='index') * 100).plot(kind='bar', figsize=(8, 4), stacked=True)
            plt.ylabel('Percentage Attrition %')
            plt.title(f'Attrition by {column}')
            bar_figure = plt.gcf()
            save_figure(bar_figure, f'{column}_attrition_bar.png')
            plt.show()

def visualize_means_grouped_by_attrition(dataframe, numerical_columns):
    """
    Visualizes the means of numerical variables grouped by attrition.
    """
    grouped_means = dataframe.groupby(['Attrition'])[numerical_columns].mean()

    # You can save the grouped means table if needed
    return grouped_means
