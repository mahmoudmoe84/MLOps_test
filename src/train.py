from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import logging
import json 
import pickle
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('train.log'),
        logging.StreamHandler()
    ]
)

log = logging.getLogger(__name__)

def main():
    log.info('Starting training pipeline')

    # Load configuration
    try:
        with open('config.json') as f:
            config = json.load(f)
        log.info(f"Loaded config: {config}")
    except FileNotFoundError:
        log.warning("config.json not found, using default configuration")
        config = {
            "test_size": 0.2,
            "random_state": 42,
            "model_type": "DecisionTree"
        }
    except json.JSONDecodeError:
        log.error("Invalid JSON in config.json")
        raise

    # Load and prepare data
    iris = load_iris()
    X = iris.data
    y = iris.target
    log.info(f'Data shape: {X.shape}, target: {set(y)}')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    log.info('Data split into train and test sets')

    # Initialize and train model
    if config['model_type'] == 'DecisionTree':
        model = DecisionTreeClassifier(random_state=config['random_state'])
    else:
        raise ValueError(f"Model type {config['model_type']} is not supported")

    log.info('Model initialized')
    model.fit(X_train, y_train)
    log.info('Model has been trained')

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    log.info(f'Model accuracy: {accuracy:.4f}')

    # Save model
    model_path = 'model.pkl'
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        log.info(f'Model saved to {model_path}')
    except Exception as e:
        log.error(f'Failed to save model: {str(e)}')
        raise

if __name__ == "__main__":
    main()
