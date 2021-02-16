import torch

x = torch.rand(4, 2)
print(x)
y = torch.rand(3, 2)
print(y)
yy = torch.ones((x.shape[0], y.shape[0], x.shape[1]), dtype=torch.float32) * y
xx = torch.ones((y.shape[0], x.shape[0], x.shape[1]), dtype=torch.float32) * x
xx = torch.transpose(xx, 1, 0)

# print(xx)
# print(yy)

sigma = 1.0

res = torch.exp(- torch.sum((xx - yy)**2, 2) / (2 * sigma**2))

print(res)
print(torch.sum(xx - yy, 2).shape)
