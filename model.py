import torch
import torch.nn as nn
from utils import CONFIG, get_config, decode_prediction
import matplotlib.pyplot as plt
from dataset import OCRDataset

class OCRModel(torch.nn.Module):
    def __init__(self, num_chars, input_size, hidden_size, num_layers):
        """OCR model with CNN and RNN layers.

        Args:
            num_chars (int): Number of characters in the dataset.
            input_size (tuple of int): Height and width of the input image.
            hidden_size (int): Number of features in the hidden state of the RNN.
            num_layers (int): Number of recurrent layers.
        """
        super(OCRModel, self).__init__()
        
        self.num_chars = num_chars
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.height, self.width = input_size
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.linear1 = nn.Linear(self.height // 4 * 128, hidden_size * 2)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.rnn = nn.LSTM(
            input_size=hidden_size * 2, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True)

        self.output = nn.Linear(hidden_size, num_chars + 1)
        

    def forward(self, x, target=None, lengths=None):
        batch_size, channels, height, width = x.size()
        # print(f'batch_size: {batch_size}, channels: {channels}, height: {height}, width: {width}')
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # print(f'after conv block 1: {x.size()}')
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # print(f'after conv block 2: {x.size()}')
        # x.shape = batch_size, n_channels, height, width
        
        # permute to batch_size, height, width, n_channels and then flatten
        # because we need to go from left to right (along width axis)
        x = x.permute(0, 3, 1, 2)
        x = x.view(batch_size, x.size(1), -1)
        # print(f'after flattening: {x.size()}')

        x = self.linear1(x)
        x = self.relu3(x)
        x = self.dropout1(x)
        # print(f'after linear layer: {x.size()}')
        
        x, _ = self.rnn(x)
        # print(f'after rnn: {x.size()}')
        
        x = self.output(x)
        # print(f'after output: {x.size()}')
        x = x.permute(1, 0, 2) # (seq_len, batch, num_chars + 1)
        
        if target is not None:
            log_softmax_values = torch.nn.functional.log_softmax(x, dim=2)
            input_lengths = torch.full(
                size=(batch_size,), fill_value=log_softmax_values.size(0), dtype=torch.int32
            )
            # print(f'input_lengths: {input_lengths}')
            target_lengths = torch.full(
                size=(batch_size,), fill_value=target.size(1), dtype=torch.int32
            )
            if lengths is not None:
                for i in range(batch_size):
                    target_lengths[i] = lengths[i]
            # print(f'target_lengths: {target_lengths}')
            loss = nn.CTCLoss(blank=0)(
                log_softmax_values, target, input_lengths, target_lengths
            )
            return x, loss

        return x, None

if __name__ == '__main__':
    # model = OCRModel(26, (get_config()['image']['height'], get_config()['image']['width']), 256, 2)
    # x = torch.rand(1, 3, get_config()['image']['height'], get_config()['image']['width'])
    # y, loss = model(x)
    # print(f'y.size() = {y.size()}')
    # print(loss)
    
    dataset = OCRDataset(
        CONFIG['image_dir'],
        CONFIG['label_dir'], 
        (CONFIG['image']['height'], CONFIG['image']['width']),
        CONFIG['letters'],
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    
    sample = next(iter(dataloader))
    
    n_letters = len(CONFIG['letters'])
    model = OCRModel(n_letters, (CONFIG['image']['height'], CONFIG['image']['width']), 256, 2)
    
    y, loss = model(sample['image'], sample['target'], sample['length'])
    print(f'y.size() = {y.size()}')
    print(f'loss = {loss}')
    
    decoded_preds = decode_prediction(y, CONFIG['letters'])
    
    for i in range(len(decoded_preds)):
        print(f'actual: {sample["label"][i]}, predicted: {decoded_preds[i]}')
