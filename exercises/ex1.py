# encoding: utf-8

import torch

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size ,inz,hidden,outz = 64,1000,200,10
    x = torch.randn(batch_size,inz,device=device)
    y = torch.rand(batch_size,outz)
    w1 = torch.randn(inz,hidden,requires_grad=True)
    w2 = torch.randn(hidden,outz,requires_grad=True)
    lr = 10**-6
    epochs = 500
    for epoch in range(epochs):
        pred = x.mm(w1).clamp(min=0).mm(w2)
        loss = (pred - y).pow(2).sum()
        loss.backward()
        with torch.no_grad():
            w1 -= w1 * lr
            w2 -= w2 * lr
            w1.grad.zero_()
            w2.grad.zero_()


if __name__ == '__main__':
    test()




