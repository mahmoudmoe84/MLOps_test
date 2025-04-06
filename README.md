# MLOps_test

this is a simple Machine Learning Project Setup with **DevContainers** , **mise** , and Python Virtual Enviroment

** Project Structure
```
.
|-- .devcontainer
|   |__ devcontainr.json
|-- mise.toml
|-- rquirements.txt
|-- config.json
|-- src
|   |__train.py
|__ README.md
```

## How to Run
1- Open Project in VSCode or Cursor
2- Click "Reopen in Container" when prompted
3- Wait for the DevContainer to finish building and setting up
4- Once the setup is complete,  activate the python VE

source .venv/bin/activte
python src/train.py

## Features
* Automated DevContainer environment using VSCode

* Python environment isolated via virtualenv

* Tooling setup with mise

* Modular configuration via config.json

* Structured logging for debugging and tracking


## Sample Output

INFO     start training
INFO     config : {'test_size': 0.2, 'random_state': 42, 'model_type': 'DecisionTree'}
INFO     Data shape: (150, 4), target: {0, 1, 2}
INFO     data split into train and test
INFO     model has been built
INFO     Model has been trained
INFO     model accuracy is : 1.0000
