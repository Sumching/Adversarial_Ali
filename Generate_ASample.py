import torch
import torch.nn as nn
from model import tmodels
from dataset import get_loader
model = tmodels.resnet50(pretrained=True)
print(model)

#model = torch.nn.DataParallel(model).cuda().eval()
def where(cond, x, y):
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)

dataloader = get_loader()

eps = 30/255
res = 0
count = 0

total_num = 0 
for idx, (xbatch, ybatch, targety_batch) in enumerate(dataloader):
    #print(torch.max(batch_x), torch.min(batch_x))
    batch_x = Variable(xbatch)
    total_num += batch_x.shape[0]
    batch_y = ybatch
    batch_x = batch_x.cuda()
    print_label = batch_y
    batch_y = f.one_hot(batch_y, 1000).float().cuda()
    #print(batch_x.shape)
    x_adv = Variable(batch_x.data, requires_grad=True)
    #x_adv = hp(x_adv)
    for i in range(10):
        h_adv = model(x_adv)
        cost = torch.mean(torch.sum(batch_y*torch.log(h_adv), dim=1))
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
    #x_adv = x_adv[:,::-1,:,:]
    torchvision.utils.save_image(x_adv, "./adv.jpg", normalize=True)
    preds = model(x_adv).argmax(dim=1)
    true_list = preds.detach().cpu().numpy() == print_label.numpy()
    res += np.sum(true_list)
    count += 1
    print(preds, print_label)
    print(true_list)

print(res/total_num)