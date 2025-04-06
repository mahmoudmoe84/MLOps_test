import json
import logging
import pickle
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def test_model():
    # Load configuration
    config = load_config()
    logger.info(f"Testing model with config: {config}")

    # Load test data
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split data the same way as in training
    _, X_test, _, y_test = train_test_split(
        X, y, 
        test_size=config['test_size'],
        random_state=config['random_state']
    )

    # Load the trained model
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.error("No trained model found! Please run train.py first")
        return

    # Make predictions on test set
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    logger.info(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    test_model() 