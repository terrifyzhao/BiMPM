import torch
import torch.nn as nn
from my import load_data
import numpy as np

p, h, label = load_data.load_data('input/test.csv')
data = {}
data['p_char'] = torch.from_numpy(np.array(p))
data['h_char'] = torch.from_numpy(np.array(h))
label = torch.from_numpy(np.array(label))

model = torch.load('BiMPM.pkl')
output = model(**data)
prediction = torch.max(output, 1)[1]
pred_y = prediction.data.numpy()
target_y = label.data.numpy()
acc = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
print('acc:' + str(acc))
