import torch.nn as nn
import torch
       
class MLPModel(nn.Module):
    def __init__(self, input_dim, task_type="regression"):
        super(MLPModel, self).__init__()
        self.task_type = task_type
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.3)
        
        # Output layer depends on the task
        if self.task_type == "regression":
            self.fc4 = nn.Linear(16, 1)  # Regression: 1 output neuron
        elif self.task_type == "binary_classification":
            self.fc4 = nn.Linear(16, 1)  # Binary classification: 1 output neuron
            self.sigmoid = nn.Sigmoid()  # Sigmoid for binary classification
        elif self.task_type == "multi_class_classification":
            self.fc4 = nn.Linear(16, 2)  # multi-class classification: 2 output neurons
            self.softmax = nn.Softmax(dim=1)  # Softmax for multi-class classification
            
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        if self.task_type == "binary_classification":
            x = self.sigmoid(x)  # Apply sigmoid for binary classification
        elif self.task_type == "multi_class_classification":
            x = self.softmax(x)  # Apply softmax for multi-class classification

        return x
        
class MLPModelTest(nn.Module):
    def __init__(self, input_dim, task_type="regression"):
        super(MLPModelTest, self).__init__()
        self.task_type = task_type
        # Add more layers
        self.fc1 = nn.Linear(input_dim, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 32)         # Second hidden layer
        self.fc3 = nn.Linear(32, 16)         # Third hidden layer
        self.fc4 = nn.Linear(16, 1)          # Output layer
        if self.task_type == "regression":
            self.fc4 = nn.Linear(16, 1)  # Regression: 1 output neuron  
        self.relu = nn.LeakyReLU()                # Activation function              
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

 
        
class NNModel(nn.Module):
    def __init__(self, input_dim, task_type="regression"):
        super(NNModel, self).__init__()
        self.task_type = task_type
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1 if task_type == "regression" else (1 if task_type == "binary_classification" else 2)) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid() if task_type == "binary_classification" else None
        self.softmax = nn.Softmax(dim=1) if task_type == "multi_class_classification" else None

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        if self.task_type == "binary_classification":
            x = self.sigmoid(x)  # Apply sigmoid for binary classification
        elif self.task_type == "multi_class_classification":
            x = self.softmax(x)  # Apply softmax for multi-class classification
        return x
