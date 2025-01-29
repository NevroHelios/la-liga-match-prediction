import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

from engine import FootballPredictorEngine
from model_builder import FootballPredictor
from data_building import get_data


x_train, y_train, x_test, y_test, i2t, t2i = get_data()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FootballDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x.values).float()
        self.y = torch.tensor(y.values).long()
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = FootballDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = FootballDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
 
model = FootballPredictor(x_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.9,
        patience=10,
        verbose=True
    )

engine = FootballPredictorEngine(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=3
)

losses, best_model = engine.train()
torch.save(best_model.state_dict(), '../saved_models/football_predictor.pth')