from dataset import OCRDataset
from model import OCRModel

from utils import CONFIG, decode_prediction
import torch

epochs = 10
learning_rate = 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = OCRDataset(
    CONFIG['image_dir'],
    CONFIG['label_dir'], 
    (CONFIG['image']['height'], CONFIG['image']['width']),
    CONFIG['letters'],
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

model = OCRModel(
    num_chars=len(CONFIG['letters']), 
    input_size=(CONFIG['image']['height'], CONFIG['image']['width']), 
    hidden_size=CONFIG['model']['hidden_size'],
    num_layers=CONFIG['model']['num_layers'],
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    print(f'########## Epoch: {epoch} ##########')
    epoch_loss = 0
    for i, sample in enumerate(dataloader):
        img = sample['image'].to(device)
        target = sample['target'].to(device)
        lengths = sample['length'].to(device)
        
        optimizer.zero_grad()
        y, loss = model(img, target, lengths)
        print(f'batches: {i}/{len(dataloader)}, loss: {loss:.4f}')
        epoch_loss += loss.cpu().item()

        loss.backward()
        optimizer.step()

        decoded_preds = decode_prediction(y, CONFIG['letters'])

        if i == (len(dataloader) // 2) or i == len(dataloader) - 1:
            print(f'epoch: {epoch}, batch: {i}')
            for i in range(len(decoded_preds)):
                print(f'actual: {sample["label"][i]}' + '\n' + f'predicted: {decoded_preds[i]}')
    print(f'End of epoch {epoch}, loss: {epoch_loss:.4f}')
    print('-' * 50)