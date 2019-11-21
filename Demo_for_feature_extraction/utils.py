import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np

def plot_demo(x1,x2,fx1,gx2,method_name):
    # plot
    plt.subplot(2,2,1)
    plt.title("{} $x_1$ v.s. $f(x_1)$".format(method_name))
    plt.scatter(x1, fx1, s=20)
    plt.xlabel("$x_1$")
    plt.ylabel("$f(x_1)$")
    plt.subplot(2,2,2)
    plt.title("{} $x_2$ v.s. $g(x_2)$".format(method_name))
    plt.scatter(x2, gx2, s=20)
    plt.xlabel("$x_2$")
    plt.ylabel("$g(x_2)$")
    plt.subplot(2,2,3)
    plt.title("{} $f(x_1)$ v.s. $g(x_2)$".format(method_name))
    plt.scatter(fx1, gx2, s=20)
    plt.xlabel("$f(x_1)$")
    plt.ylabel("$g(x_2)$")
    plt.tight_layout()
    plt.savefig("demo_1_{}".format(method_name))
    plt.show()
    plt.close()
    return

def plot_demo_v1(x1,x2,fx1,gx2,reds,blues,method_name):
    plt.subplot(2,2,1)
    plt.title("{} $x_1$ v.s. $x_2$".format(method_name))
    plt.scatter(x1[blues],x2[blues], s=20, c="blue")
    plt.scatter(x1[reds],x2[reds], s=20, c="red")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.subplot(2,2,2)
    plt.title("{} $x_1$ v.s. $f(x_1)$".format(method_name))
    plt.scatter(x1[reds],fx1[reds], s=20,c="red")
    plt.scatter(x1[blues],fx1[blues], s=20,c="blue")

    plt.xlabel("$x_1$")
    plt.ylabel("$f(x_1)$")
    plt.subplot(2,2,3)
    plt.title("{} $x_2$ v.s. $g(x_2)$".format(method_name))
    plt.scatter(x2[reds],gx2[reds], s=20,c="red")
    plt.scatter(x2[blues],gx2[blues], s=20,c="blue")
    plt.xlabel("$x_2$")
    plt.ylabel("$f(x_2)$")

    plt.subplot(2,2,4)
    plt.title("{} $f(x_1)$ v.s. $g(x_2)$".format(method_name))
    plt.scatter(fx1[reds], gx2[reds],s=20,c="red")
    plt.scatter(fx1[blues], gx2[blues],s=20,c="blue")

    plt.xlabel("$f(x_1)$")
    plt.ylabel("$g(x_2)$")
    plt.tight_layout()
    plt.savefig("demo_classification_{}".format(method_name))
    plt.show()
    plt.close()

"""Related to HGR
"""
def _neg_hscore(f,g):
    f0 = f - torch.mean(f,0)
    g0 = g - torch.mean(g,0)
    corr = torch.mean(torch.sum(f0*g0,1))
    cov_f = torch.mm(torch.t(f0),f0) / (f0.size()[0]-1.)
    cov_g = torch.mm(torch.t(g0),g0) / (g0.size()[0]-1.)
    return - corr + torch.trace(torch.mm(cov_f, cov_g)) / 2.

class SimpleNN(nn.Module):
    def __init__(self,input_dim):
        super(SimpleNN, self).__init__()
        self.linear1 = nn.Linear(input_dim, 50)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(50, 100)
        self.out = nn.Linear(100, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.out(x)
        return x

class HGR():
    def __init__(self, x_dim=1, y_dim=1):
        # start learn the HGR
        self.f_model = SimpleNN(x_dim)
        self.g_model = SimpleNN(y_dim)

        return

    def fit(self,x,y,max_iter=10000,lr=0.0001):
        optimizer = torch.optim.Adam([{"params":self.f_model.parameters()},{"params":self.g_model.parameters()}], lr=lr)
        x_tensor = torch.Tensor(x)
        y_tensor = torch.Tensor(y)
        for i in range(max_iter):
            fx = self.f_model(x_tensor)
            gy = self.g_model(y_tensor)
            loss = _neg_hscore(fx, gy)
            loss.backward()
            optimizer.step()
            if i % 5000 == 0 and i > 0:
                print("[{}] neg_hscore: {}".format(i, loss))
        return

    def transform(self,x,y):
        x_tensor = torch.Tensor(x)
        y_tensor = torch.Tensor(y)
        fx = self.f_model(x_tensor)
        gy = self.g_model(y_tensor)
        fx_ = fx.detach().numpy()
        gy_ = gy.detach().numpy()
        return fx_, gy_
