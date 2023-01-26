import yaml
import torch


def get_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

CONFIG = get_config()

def decode_prediction(preds, letters):
    preds = preds.permute(1, 0, 2) # [seq_len, batch_size, num_classes] -> [batch_size, seq_len, num_classes]
    preds = torch.softmax(preds, dim=2)
    preds = preds.argmax(dim=2)
    preds = preds.detach().cpu().numpy()
    texts = []
    for i in range(preds.shape[0]):
        tmp = []
        for k in preds[i, :]:
            k = k - 1
            if k == -1:
                tmp.append('$')
            else:
                tmp.append(letters[k])
        tp = ''.join(tmp)
        texts.append(tp)
    return texts