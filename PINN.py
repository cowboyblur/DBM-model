import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from matplotlib import cm
import time

torch.backends.cudnn.benchmark = True

# bulid net
class Net(nn.Module):
    def __init__(self, NN): 
        super(Net, self).__init__()

        self.input_layer = nn.Linear(2, NN)
        self.hidden_layer1 = nn.Linear(NN,NN) 
        self.hidden_layer2 = nn.Linear(NN, NN) 
        self.output_layer = nn.Linear(NN, 1)

    def forward(self, x): 
        out = torch.tanh(self.input_layer(x))
        out = torch.tanh(self.hidden_layer1(out))
        out = torch.tanh(self.hidden_layer2(out))
        out_final = self.output_layer(out)
        return out_final

def laplace(x, net):
    #laplace equation
    u = net(x)  
    u_tx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(net(x)),
                               create_graph=True, allow_unused=True)[0] 
    d_y = u_tx[:, 0].unsqueeze(-1)
    d_x = u_tx[:, 1].unsqueeze(-1)
    u_xx = torch.autograd.grad(d_x, x, grad_outputs=torch.ones_like(d_x),
                               create_graph=True, allow_unused=True)[0][:,1].unsqueeze(-1)  
    u_yy = torch.autograd.grad(d_y, x, grad_outputs=torch.ones_like(d_y),
                               create_graph=True, allow_unused=True)[0][:,0].unsqueeze(-1) 
    return torch.from_numpy(np.zeros((150,1))).float() 

init_time=time.time()
net = Net(64).to('cuda')
#loss function and optimizer
mse_cost_function = torch.nn.MSELoss().to('cuda')
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


x_right=torch.from_numpy(np.zeros((150,1))).float().to('cuda')
x_left=torch.from_numpy(np.full((150,1),97)).float().to('cuda')
y_up=torch.from_numpy(np.zeros((150,1))).float().to('cuda')
y_down=torch.from_numpy(np.full((150,1),97)).float().to('cuda')
bc_zero=torch.from_numpy(np.zeros((150,1))).float().to('cuda')
bc_max=torch.from_numpy(np.full((150,1),97)).float().to('cuda')
iterations = 5000
for epoch in range(iterations):
    optimizer.zero_grad()  # set grad to 0

    x_var = torch.from_numpy(np.random.uniform(low=0, high=97, size=(150, 1))).float().to('cuda')
    y_var = torch.from_numpy(np.random.uniform(low=0, high=97, size=(150, 1))).float().to('cuda')
    u_bc=y_var*100/97

    # boundary
    net_right = net(torch.cat([x_right,y_var], 1))  
    net_left = net(torch.cat([x_var, y_var], 1))  
    net_up = net(torch.cat([x_var, y_up], 1))  
    net_down = net(torch.cat([x_var,y_down],1))
    mse_right = mse_cost_function(net_right, u_bc).to('cuda')  
    mse_left = mse_cost_function(net_left, u_bc).to('cuda') 
    mse_up = mse_cost_function(net_up, bc_zero).to('cuda')  
    mse_down = mse_cost_function(net_down, bc_max).to('cuda')

    # interior point
    x_in = torch.from_numpy(np.random.uniform(low=0, high=97, size=(150, 1))).float().to('cuda')
    y_in = torch.from_numpy(np.random.uniform(low=0, high=97, size=(150, 1))).float().to('cuda')
    y_in.requires_grad=True
    x_in.requires_grad=True
    f_out = laplace(torch.cat([x_in, y_in], 1), net).to('cuda') 
    mse_f = mse_cost_function(f_out, bc_zero).to('cuda')

    # counting loss
    loss =mse_f+mse_up+mse_down+mse_left+mse_right
    loss=loss.to('cuda')
    loss.backward()  
    optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

    with torch.autograd.no_grad():
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{5000}, Loss: {loss.item():.4f}")

final_time=time.time()
print(final_time-init_time)
#Picture
x = np.linspace(0, 97,150)
y = np.linspace(0, 97,150)
ms_x, ms_y = np.meshgrid(x, y)
x = np.ravel(ms_x).reshape(-1, 1)
y = np.ravel(ms_y).reshape(-1, 1)
pt_x = torch.from_numpy(x).float()
pt_y = torch.from_numpy(y).float()
pt_x.requires_grad=True
pt_y.requires_grad=True
pt_x = pt_x.to('cuda')
pt_y = pt_y.to('cuda')
pt_u0 = net(torch.cat([pt_x, pt_y], 1))
print(pt_u0.reshape(150,150))
u = pt_u0.detach().cpu().numpy()
pt_u0 = u.reshape(150,150)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_zlim([0, 150])
ax.plot_surface(ms_y, ms_x, pt_u0, cmap=cm.CMRmap_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
ax.set_xlabel('y')
ax.set_ylabel('x')
ax.set_zlabel('u')
plt.show()
