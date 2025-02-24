import matplotlib.pyplot as plt
import logging
import os

def configure_logging(log_file='logs/application.log'):
    """Setup logging configuration."""
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s',
    )
    logger = logging.getLogger()
    return logger

def save_figure(figure, filename, folder='output'):
    """Save a matplotlib figure as an image in the specified folder."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, filename)
    figure.savefig(filepath)
    plt.close(figure)
