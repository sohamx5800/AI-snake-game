import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class DQN(nn.Module):
    def __init__(self, input_size=19, hidden_size=256, output_size=3):  # Updated to 19
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self, filename="snake_ai_model.pth"):
        torch.save(self.state_dict(), filename)
        print(f"✅ Model saved to {filename}")

    def load(self, filename="snake_ai_model.pth"):
        if os.path.exists(filename):
            self.load_state_dict(torch.load(filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
            self.eval()
            print(f"✅ Loaded model from {filename}")
        else:
            print(f"⚠ No model found at {filename}")

def get_optimizer(model, lr=0.001):
    return optim.Adam(model.parameters(), lr=lr)