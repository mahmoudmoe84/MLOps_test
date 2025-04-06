from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import logging
import json 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

log = logging.getLogger(__name__)
log.info('start training')

with open('config.json') as f:
    config = json.load(f)
    
log.info(f"config : {config}")


iris = load_iris()
X = iris.data
y = iris.target
log.info(f'Data shape: {X.shape}, target: {set(y)}')


X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=config['test_size'],random_state=config['random_state']
                                                 )
log.info('data split into train and test')
#train the model 

if config['model_type'] =='DecisionTree':
    model = DecisionTreeClassifier()
    
else:
    raise ValueError('model is not supported')

log.info('model has been built')
model.fit(X,y)
log.info('Model has been trained')

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
log.info(f'model accuracy is : {accuracy:0.4f}')
