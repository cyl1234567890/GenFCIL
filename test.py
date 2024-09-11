import torch
x=[8,2,3]
a=torch.tensor([1,2,3])
b=torch.argmin(a)
print(b, type(b), x[b])