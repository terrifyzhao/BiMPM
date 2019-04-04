import os
import sys
path = os.getcwd()
sys.path.append(path)

from BiMPM_torch import BiMPM
import load_data as load
import torch.nn as nn
import torch
import numpy as np
import torch.utils.data as Data
import args

# p, h, label = load.load_data('input/dev.csv')
p, h, label = load.load_fake_data()
data = {}
data1 = torch.from_numpy(np.array(p))
data2 = torch.from_numpy(np.array(h))
label = torch.from_numpy(np.array(label))

torch_dataset = Data.TensorDataset(data1, data2, label)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
)

if torch.cuda.is_available():
    model = BiMPM().cuda()
else:
    model = BiMPM()

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

for i in range(10):
    print()
    for step, (data1, data2, label) in enumerate(loader):
        data = {}
        data['p_char'] = data1
        data['h_char'] = data2
        output = model(**data)
        loss = loss_func(output, label.cuda())

        prediction = torch.max(output, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = label.data.numpy()
        acc = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        print(f'epoch:{i} step:{step} loss:{loss.data} acc:{acc}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model, 'BiMPM.pkl')
