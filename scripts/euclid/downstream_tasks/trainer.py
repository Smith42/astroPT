import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ModelTrainer:
    def __init__(self, model, lr=0.001, epochs=50, task_type="regression",  pos_weight=None):
        self.model = model
        self.epochs = epochs
        self.task_type = task_type
        # criterion depends on task
        if self.task_type == "regression":
            #self.criterion = nn.MSELoss()    
            #self.criterion = nn.SmoothL1Loss()  
            self.criterion = nn.HuberLoss(delta=1.0)
        elif self.task_type == "binary_classification":
            if pos_weight is not None:
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        elif self.task_type == "multi_class_classification":
            self.criterion = nn.CrossEntropyLoss() #Multi-class classification
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
        #self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.1)  # Reduce LR by 10x every 20 epochs
        #self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)        

    def train(self, X_train, y_train, X_val, y_val):
        train_losses, val_losses = [], []
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            predictions = self.model(X_train).squeeze()
            loss = self.criterion(predictions,  y_train.float())  # Ensure labels are float for BCELoss
            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()  # Update learning rate  
                      
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val).squeeze()
                val_loss = self.criterion(val_predictions, y_val)
            #self.scheduler.step(val_loss)
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
        return train_losses, val_losses

