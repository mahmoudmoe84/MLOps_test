import json
import logging
import pickle
import os
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
    """Load configuration from config.json or return defaults."""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("config.json not found, using default configuration")
        return {
            "test_size": 0.2,
            "random_state": 42,
            "model_type": "DecisionTree"
        }
    except json.JSONDecodeError:
        logger.error("Invalid JSON in config.json")
        raise

def test_model():
    try:
        # Load configuration
        config = load_config()
        logger.info(f"Testing model with config: {config}")

        # Load test data
        iris = load_iris()
        X = iris.data
        y = iris.target
        logger.info(f"Loaded test data: {X.shape}")
        
        # Split data the same way as in training
        _, X_test, _, y_test = train_test_split(
            X, y, 
            test_size=config['test_size'],
            random_state=config['random_state']
        )
        logger.info("Data split completed")

        # Load the trained model
        model_path = 'model.pkl'
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found! Please run train.py first")
            return 1

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return 1

        # Make predictions on test set
        try:
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            logger.info(f"Test accuracy: {accuracy:.4f}")
            return 0 if accuracy > 0 else 1
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return 1

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(test_model()) 