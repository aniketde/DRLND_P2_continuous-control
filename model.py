import torch.nn.functional as F
import torch.nn as nn
import torch


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, seed,  fc1_units=512, fc2_units=512,fc3_units=512, fc4_units=256,  fc5_units=256):
        super(Actor, self).__init__()

        torch.manual_seed(seed)
        self.bn1 = nn.BatchNorm1d(num_features=state_dim)
        self.fc1 = nn.Linear(state_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, fc4_units)
        self.fc5 = nn.Linear(fc4_units, fc5_units)        
        self.fc6 = nn.Linear(in_features=fc5_units, out_features=action_dim)


        init_weights(self.fc1)
        init_weights(self.fc2)
        init_weights(self.fc3)
        init_weights(self.fc4)
        init_weights(self.fc5)
        init_weights(self.fc6)

    def forward(self, state):
        x = self.bn1(state)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return F.tanh(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, seed, fcs1_units=512, fc2_units=512,fc3_units=512, fc4_units=256,  fc5_units=256):
        super(Critic, self).__init__()

        torch.manual_seed(seed)
        self.bn1 = nn.BatchNorm1d(num_features=state_dim)
        self.fcs1 = nn.Linear(in_features=state_dim, out_features=fcs1_units)
        self.fc2 = nn.Linear(in_features=fcs1_units+action_dim, out_features=fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, fc4_units)
        self.fc5 = nn.Linear(fc4_units, fc5_units)
        self.fc6 = nn.Linear(in_features=fc5_units, out_features=1)
        

        init_weights(self.fcs1)
        init_weights(self.fc2)
        init_weights(self.fc3)
        init_weights(self.fc4)
        init_weights(self.fc5)        
        init_weights(self.fc6)

    def forward(self, state, action):
        x = self.bn1(state)
        x = F.relu(self.fcs1(x))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.fc6(x)
