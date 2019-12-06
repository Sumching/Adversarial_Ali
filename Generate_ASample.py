import torch
import torch.nn as nn
from model import tmodels
from dataset import get_loader
from torch.autograd import Variable
import torchvision
import torch.nn.functional as f
import numpy as np
model1 = tmodels.inception_v3(pretrained=True).cuda().eval()
model = tmodels.vgg19(pretrained=True).cuda().eval()
#print(model)

#model = torch.nn.DataParallel(model).cuda().eval()
def where(cond, x, y):
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)

dataloader = get_loader()

eps = 32/255
res = 0
count = 0

total_num = 0 
for idx, (xbatch, ybatch, targety_batch, FN) in enumerate(dataloader):
    name = FN[0]
    batch_x = Variable(xbatch)
    print(torch.max(batch_x), torch.min(batch_x))
    total_num += batch_x.shape[0]
    batch_y = ybatch
    batch_x = batch_x.cuda()
    print_label = batch_y
    print_label1 = targety_batch
    #print(batch_y)
    batch_y = f.one_hot(batch_y, 1000).float().cuda()
    targety = f.one_hot(targety_batch, 1000).float().cuda()
    #print(batch_x.shape)
    x_adv = Variable(batch_x.data, requires_grad=True)
    
    for i in range(20):
        h_adv = model(x_adv)
        h_adv = f.sigmoid(h_adv)
        cost1 = torch.mean(torch.sum(batch_y*torch.log(h_adv), dim=1))
        cost2 = -torch.mean(torch.sum(targety*torch.log(h_adv), dim=1))
        cost = cost1 + cost2
        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)

        cost.backward()
        x_adv.grad.sign_()
        x_adv = x_adv - (2/255) * x_adv.grad
        x_adv = where(x_adv>batch_x+eps, batch_x+eps, x_adv)
        x_adv = where(x_adv<batch_x-eps, batch_x-eps, x_adv)
        x_adv = torch.clamp(x_adv, -1, 1)
        x_adv = Variable(x_adv.data, requires_grad=True)
    for i in range(20):
        h_adv = model1(x_adv)
        h_adv = f.sigmoid(h_adv)
        cost1 = torch.mean(torch.sum(batch_y*torch.log(h_adv), dim=1))
        cost2 = -torch.mean(torch.sum(targety*torch.log(h_adv), dim=1))
        cost = cost1 + cost2
        model1.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)

        cost.backward()
        x_adv.grad.sign_()
        x_adv = x_adv - (2/255) * x_adv.grad
        x_adv = where(x_adv>batch_x+eps, batch_x+eps, x_adv)
        x_adv = where(x_adv<batch_x-eps, batch_x-eps, x_adv)
        x_adv = torch.clamp(x_adv, -1, 1)
        x_adv = Variable(x_adv.data, requires_grad=True)
    
    torchvision.utils.save_image(x_adv, "./images/"+name, normalize=True)
    preds = model1(x_adv).argmax(dim=1)
    true_list = preds.detach().cpu().numpy() == print_label.numpy()
    true_list1 = preds.detach().cpu().numpy() == print_label1.numpy()
    res += np.sum(true_list)
    count += 1
    print(preds, print_label, print_label1)
    print(true_list)
    print(true_list1)

print(res/total_num)