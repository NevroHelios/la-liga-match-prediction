import torch
from torch import nn
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

class FootballPredictor(nn.Module):
    def __init__(self, input_dim):
        super(FootballPredictor, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(input_dim, 700)
        self.bn1 = nn.BatchNorm1d(700)
        
        self.fc2 = nn.Linear(700, 500)
        self.bn2 = nn.BatchNorm1d(500)
        
        self.fc3 = nn.Linear(500, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        
        self.fc5 = nn.Linear(32, 3)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout(torch.relu(self.bn3(self.fc3(x))))
        x = self.dropout(torch.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        return x
    

class FootballPredictorWrapper:
    def __init__(self, model_path='saved_models/model_nn.pth', device='cpu'):
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        self.model = FootballPredictor(40)
        
        # Load the model
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        self.model.eval()
        self.classes_ = np.array([0, 1, 2])  # Draw, Home Win, Away Win
    
    def predict(self, X):
        """Convert DataFrame to tensor and return predictions"""
        X = torch.FloatTensor(X.values).to(self.device)
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        """Return probability distribution for each class"""
        X = torch.FloatTensor(X.values).to(self.device)
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()
    
    @property
    def feature_names_in_(self):
        features_used = [
            'Home Team', 'Away Team', 'Match Excitement', 'Home Team Rating',
            'Away Team Rating', 'Home Team Possession %', 'Away Team Possession %',
            'Home Team Off Target Shots', 'Home Team On Target Shots',
            'Home Team Total Shots', 'Home Team Blocked Shots', 'Home Team Corners',
            'Home Team Throw Ins', 'Home Team Pass Success %',
            'Home Team Aerials Won', 'Home Team Clearances', 'Home Team Fouls',
            'Home Team Yellow Cards', 'Home Team Second Yellow Cards',
            'Home Team Red Cards', 'Away Team Off Target Shots',
            'Away Team On Target Shots', 'Away Team Total Shots',
            'Away Team Blocked Shots', 'Away Team Corners', 'Away Team Throw Ins',
            'Away Team Pass Success %', 'Away Team Aerials Won',
            'Away Team Clearances', 'Away Team Fouls', 'Away Team Yellow Cards',
            'Away Team Second Yellow Cards', 'Away Team Red Cards', 'year',
            'Home Team Half Time Goals', 'Away Team Half Time Goals',
            'away_team_away_winrate', 'away_team_overall_winrate',
            'home_team_home_winrate', 'home_team_overall_winrate'
        ]
        return features_used