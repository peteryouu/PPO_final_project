import torch

class Actor(torch.nn.Module):
    def __init__(self, input_shape, action_size):
        super(Actor, self).__init__()
        self.input_shape = input_shape
        self.action_size = action_size
        
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        
        self.full_connected = torch.nn.Sequential(
            torch.nn.Linear(7*7*64, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.action_size),
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.full_connected(x)
        return torch.distributions.Categorical(x)
    
class Critic(torch.nn.Module):
    def __init__(self, input_shape):
        super(Critic, self).__init__()
        self.input_shape = input_shape
        
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=4,out_channels=32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        
        self.full_connected = torch.nn.Sequential(
            torch.nn.Linear(in_features=7*7*64, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=1))
        
    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.full_connected(x)
        return x
    
