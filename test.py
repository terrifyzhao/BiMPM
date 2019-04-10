import torch
import load_data
import numpy as np
import torch.utils.data as Data
import args

p, h, label = load_data.load_data('input/test.csv')
data = {}
data1 = torch.from_numpy(np.array(p))
data2 = torch.from_numpy(np.array(h))
label = torch.from_numpy(np.array(label))

# if torch.cuda.is_available():
#     data1 = data1.cuda()
#     data2 = data2.cuda()
#     label = label.cuda()

torch_dataset = Data.TensorDataset(data1, data2, label)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4
)

model = torch.load('BiMPM.pkl')
for step, (data1, data2, label) in enumerate(loader):
    data = {}
    data['p_char'] = data1.cuda()
    data['h_char'] = data2.cuda()
    output = model(**data)
    prediction = torch.max(output, 1)[1]
    if torch.cuda.is_available():
        pred_y = prediction.data.cpu().numpy()
    else:
        pred_y = prediction.data.numpy()
    target_y = label.data.numpy()
    acc = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
    print('acc:' + str(acc))
