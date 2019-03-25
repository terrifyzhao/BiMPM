import torch
import numpy as np

# rand = torch.rand(2, 3, 12)
# split1, split2 = torch.split(rand, 6, dim=-1)
# print(rand)
# print(split1)
# print(split2)


# a = torch.rand(2, 3, 4)
# b = torch.transpose(a, 1, 0)
# print(a)
# print(b)


# a = torch.rand(2, 3)
# b = torch.ones(2, 3)
#
# # a = torch.from_numpy(np.array((1, 2, 3)))
# a = torch.FloatTensor([1, 2, 3])
# print(a)
# c = torch.stack([a]*3, dim=1)
# print(c)

a = torch.rand(2, 3, 3)
print(a)
print(a[:, 0, :])
print(a[:, -1, :])
