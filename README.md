# MLOps Test Project

This is a simple Machine Learning Project setup with **DevContainers**, **mise**, and Python Virtual Environment.

## Project Structure
```
.
├── .devcontainer/
│   ├── Dockerfile
│   ├── devcontainer.json
│   └── postCreateCommand.sh
├── mise.toml
├── requirements.txt
├── config.json
├── src/
│   └── train.py
└── README.md
```

## How to Run

1. Open project in VSCode or Cursor
2. Click "Reopen in Container" when prompted
3. Wait for the DevContainer to finish building and setting up
4. Once the setup is complete, activate the Python virtual environment:

```bash
source .venv/bin/activate
python src/train.py
```

## Features

* **Containerized Development**: Automated DevContainer environment using VSCode
* **Isolated Environment**: Python environment isolated via virtualenv
* **Tool Management**: Development tooling setup with mise
* **Configuration**: Modular configuration via config.json
* **Logging**: Structured logging for debugging and tracking

## Sample Output

```
INFO     start training
INFO     config : {'test_size': 0.2, 'random_state': 42, 'model_type': 'DecisionTree'}
INFO     Data shape: (150, 4), target: {0, 1, 2}
INFO     data split into train and test
INFO     model has been built
INFO     Model has been trained
INFO     model accuracy is : 1.0000
```

## Technologies Used

* Python
* Docker (DevContainers)
* mise (Tool Version Manager)
* scikit-learn
* VSCode/Cursor IDE
