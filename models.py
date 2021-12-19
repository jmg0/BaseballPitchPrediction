import numpy as np
import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset, DataLoader

import data_preparation

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='PitchFrequencyModel',
                    help='Choose one of: PitchFrequencyModel, NeuralNetwork, LSTM')
args = parser.parse_args()

class PitchFrequencyModel:
    def __init__(self, labels):
        self.labels = labels
        self.pitches, self.counts = np.unique(self.labels, return_counts=True)

    def get_frequencies(self):
        return self.counts

    def predict_next_pitch(self):
        frequencies = self.counts / sum(self.counts)
        return np.random.choice(self.pitches, p=frequencies)

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=3):
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.sequential(x)

class NNDataset(Dataset):
    def __init__(self, features, labels):
        super().__init__()
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_nn(nn_model, train_f, train_l, valid_f, valid_l, epochs=5, lr=0.01):
    train_data = DataLoader(NNDataset(train_f, train_l), batch_size=10, shuffle=True, drop_last=False)
    valid_data = DataLoader(NNDataset(valid_f, valid_l), batch_size=10, shuffle=True, drop_last=False)
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=lr)
    running_loss = 0.0
    for epoch in range(epochs):
        for i, data in enumerate(train_data):
            inputs, labels = data
            optimizer.zero_grad()
            output = nn_model(inputs.float())
            loss = F.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(running_loss)
    nn_model.eval()

class LSTM(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=3, num_layers=1, batch_first=True):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.output = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, l):
        output_feats, (hidden_state, cell_state) = self.lstm(x)
        return self.output(hidden_state[-1])

# DataLoader wasn't working with sequential data so modified data directly in function
def train_lstm(lstm_model, train_f, train_l, valid_f, valid_l, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, lstm_model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        lstm_model.train()
        for x, y in zip(train_f, train_l):
            x, y = torch.reshape(torch.tensor(x.astype(float)), (1,1,11)), torch.tensor(y.astype(int))
            x, y = x.type(torch.FloatTensor), y.type(torch.LongTensor)
            y_pred = lstm_model(x, x[0])
            y_prob = torch.zeros(y_pred.shape)
            y_prob[0, y] = -1.0
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, torch.reshape(y, [1]))
            loss.backward()
            optimizer.step()
    lstm_model.eval()

if __name__ == '__main__':
    # Run through on example data, models won't really learn anything, though
    model = args.model
    features, labels = data_preparation.prep_data_single()
    train_features, train_labels = features[:80], labels[:80]
    valid_features, valid_labels = features[80:], labels[80:]
    num_pitches = len(np.unique(labels))

    if model == 'PitchFrequencyModel':
        pf = PitchFrequencyModel(train_labels)
    elif model == 'NeuralNetwork':
        nn = NeuralNetwork(11, num_pitches)
        train_nn(nn, train_features, train_labels, valid_features, valid_labels)
    elif model == 'LSTM':
        lstm = LSTM(input_size=11, output_size=num_pitches, hidden_size=5, num_layers=1, batch_first=True)
        train_lstm(lstm, train_features, train_labels, valid_features, valid_labels)