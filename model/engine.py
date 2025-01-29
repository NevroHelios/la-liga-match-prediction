import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class FootballPredictorEngine():
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
                 epochs: int = 10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs

    def train(self):
        best_loss = float('inf')
        best_model_state = None
        losses = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y.long())
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1

                losses.append(loss.item())
            
            avg_epoch_loss = epoch_loss / num_batches
            
            # Step scheduler with average epoch loss
            self.scheduler.step(avg_epoch_loss)
            
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_state = self.model.state_dict().copy()
                
            if epoch % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch: {epoch}, Loss: {avg_epoch_loss:.4f}, Best Loss: {best_loss:.4f}, LR: {current_lr:.6f}")
            
        # Restore best model
        self.model.load_state_dict(best_model_state)
        print(f"Training completed. Best loss: {best_loss:.4f}")
        return losses, self.model

    def predict(self):
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for x in self.test_loader:
                logits = self.model(x)
                predictions.extend(logits.argmax(dim=1).tolist())
        
        return predictions