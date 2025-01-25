"this script include the tool functions"
import numpy as np
import os
import csv
import logging

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from utils.random_seed import set_seed
from torch.distributions import Normal

EPSILON = 1e-8

# ---------------------------------------------------------------------------------
# Rrep_train
def pretrain_rep(args, Rnet, Dnet, train_loader, optimizerR, optimizerD, savefile):
    Rnet = Rnet.to(args.device)
    Dnet = Dnet.to(args.device)
    Rnet.train()
    Dnet.train()
    mseloss = nn.MSELoss().to(args.device)
    print("pretraining Representor begin")
    logger = logging.getLogger("loggerrep")
    logger.setLevel(logging.DEBUG)
    file_name = os.path.join(savefile, 'reptrain.log')
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.info(f"Start CVRTL pretraining Represent on dataset {args.data_name} with {args.split} splits in random: {args.seed}.")
    logger.info(f"the epoch is {args.epochs}, batch size is {args.bths}.")
    logger.info(f"Training with no gpu: {args.nocuda}.")
    writer = SummaryWriter(savefile)

    niter = 0
    for i in trange(args.epochs):
        if i < 50: lr = args.lr
        elif i == 50: lr = args.lr*0.5
        elif i == 100: lr = args.lr*0.1
        elif i == 150: lr = args.lr*0.01
        for batch_indx, (data, target) in enumerate(train_loader):
            data =data.to(args.device)
            target = target.to(args.device)
            onelabels = torch.ones(data.shape[0],1).to(args.device)
            zerolabels = torch.zeros(data.shape[0], 1).to(args.device)
            z = torch.randn(data.shape[0], args.latdim).to(args.device)
            z = torch.div(z, torch.t(torch.norm(z, p = 'fro', dim = 1).repeat(args.latdim, 1)))
            
            # 训练一个判别器D_{\phi}, ratio 估计与之相似
            optimizerD.zero_grad()
            w= Rnet(data)
            new_w = w.clone()
            D_real = torch.sigmoid(Dnet(new_w))
            D_fake = torch.sigmoid(Dnet(z))
            D_loss_real = F.binary_cross_entropy(D_real, onelabels)
            D_loss_fake = F.binary_cross_entropy(D_fake, zerolabels)
            D_loss = (D_loss_real + D_loss_fake)/2
            optimizerD.zero_grad()
            D_loss.backward()
            optimizerD.step()

            # 粒子法
            w.detach_()
            w_t = w.clone().requires_grad_(True)
            d = -Dnet(w_t)
            d.backward(torch.ones(w_t.shape[0],1).to(args.device), retain_graph = True)
            w = w + lr*w_t.grad

            # 更新表示
            optimizerR.zero_grad()
            target_onehot = to_onehot(target).to(args.device)
            latent = Rnet(data)
            mloss = mseloss(w, latent)
            dcorloss = cor(latent, target_onehot, data.shape[0], args.device)
            Rloss = args.lambdaa*mloss - dcorloss
            Rloss.backward()
            optimizerR.step()

            # 计算由提取的特征生成的标签判别，计算GAN损失
            D_nerual = torch.sigmoid(Dnet(latent))
            OG_loss = F.binary_cross_entropy(D_nerual, zerolabels)

            if niter % 100 == 0:
                writer.add_scalar("dCor loss", dcorloss, niter)
                writer.add_scalar("VG", mloss, niter)
                writer.add_scalar("D loss", D_loss.item(), niter)
                writer.add_scalar("OG_loss", OG_loss.item(), niter)
                for name, param in Rnet.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(name + '_grad', param.grad, niter)
                        writer.add_histogram(name + '_data', param, niter)
            niter += 1
        logger.debug(f"Epoch: {i}\tdCor loss: {dcorloss}\tRloss: {Rloss}")    


# -----------------------------------------------------------------------------
# density ration pretrain
def pretrain_dens(args, Rnet, Dens_net, optim_Dens, source_trainloader, target_trainloader, savefile):
    print("pretraining Features density ratio begin")
    Dens_net = Dens_net.to(args.device)
    Rnet = Rnet.to(args.device)
    Rnet.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    logger = logging.getLogger("loggerdens{}".format(args.Method))
    logger.setLevel(logging.DEBUG)
    file_name = os.path.join(savefile, 'train.log')
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.info(f"Start CVRTL pretraining Features Density in {args.Method} on dataset {args.data_name} with {args.split} splits in random: {args.seed}.")
    logger.info(f"the epoch is {args.epochs}, batch size is {args.bths}.")
    logger.info(f"Training with no gpu: {args.nocuda}.")
    writer = SummaryWriter(savefile)
    niter = 0

    process = _without_Rnet if args.Method in ["Method_B", "Method_C"] else _with_Rnet
    len_source_loader = len(source_trainloader)
    len_target_loader = len(target_trainloader)
    n_batch = min(len_source_loader, len_target_loader)
    
    for i in trange(100):
        iter_source = iter(source_trainloader)
        iter_target = iter(target_trainloader)
        loss1 = loss2 = 0
        for e in range(n_batch):
            data1, _ = next(iter_source)
            data2, _ = next(iter_target)
            data2 = data2.to(args.device)
            data1 = data1.to(args.device)
            pro = process(Rnet, data1, data2)
            z1, z2 = pro[0], pro[1]
            logit1 = Dens_net(z1)
            logit2 = Dens_net(z2)
            loss1 = criterion(logit1, torch.ones_like(logit1))
            loss2 = criterion(logit2, torch.zeros_like(logit2))
            loss = loss1 + loss2
            optim_Dens.zero_grad()
            loss.backward()
            optim_Dens.step()
            
            if niter % 10 == 0:
                writer.add_scalar("loss", loss.item(), niter)
                for name, param in Dens_net.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(name + "_grad", param.grad, niter)
                        writer.add_histogram(name + "data", param, niter)
            niter += 1
        logger.debug(f"density--epoch: {i}\t loss: {loss.item()}") 

def _with_Rnet(modal, *args):
    z = [modal(data) for data in args]
    return z
def _without_Rnet(modal, *args):
    return args


# -----------------------------------------------------------------------------
# train final predictor h
def pretrain_h(args, h, Rnet, Dens_net, optimal_h, source_loader, savefile):
    if args.task == "regression":
        criterion = nn.MSELoss(reduction="none")
        _reg_h(args, h, Rnet, Dens_net, optimal_h, source_loader, savefile, criterion)
    elif args.task == "classifiaction":
        criterion = nn.CrossEntropyLoss(reduction="none")
        _class_h(args, h, Rnet, Dens_net, optimal_h, source_loader, savefile, criterion)
    else:
        raise ValueError("please choose right task regression/classification !")

def _reg_h(args, h, Rnet, Dens_net, optimal_h, source_loader, savefile, criterion):
    print("pretraining predictor h begin")
    h = h.to(args.device)
    Rnet = Rnet.to(args.device)
    Dens_net = Dens_net.to(args.device)
    Rnet.eval()
    Dens_net.eval()
    h.train()

    logger = logging.getLogger("logger{}".format(args.Method))
    logger.setLevel(logging.DEBUG)
    file_name = os.path.join(savefile, 'train.log')
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.info(f"Start CVRTL pretraining predictor h in {args.Method} on dataset {args.data_name} with {args.split} splits in random: {args.seed}.")
    logger.info(f"the epoch is {args.epochs}, batch size is {args.bths}.")
    logger.info(f"Training with no gpu: {args.nocuda}.")
    writer = SummaryWriter(savefile)
    niter = 0

    process = _without_Rnet if args.Method in ["Method_B", "Method_C", "Method_D"]  else _with_Rnet
    _criterion = _without_Dens_r if args.Method == "Method_C" else _with_Dens_r
    for i in trange(args.epochs):
        train_acc = 0
        loss_avg=0
        for count, (data,target) in enumerate(source_loader):
            data = data.to(args.device)
            target = target.to(args.device)
            # Method D not use representation
            if args.Method == "Method_D":
                re = data
            else:    
                re = Rnet(data)
            logits = h(re)
            # process is to choose weather to use representation in density ratio
            z = process(Rnet, data)[0]
            loss = _criterion(Dens_net, args.r, logits, target.float(), z, criterion)
            loss_avg += loss
            optimal_h.zero_grad()
            loss.backward()
            optimal_h.step()
            if niter % 100 == 0:
                writer.add_scalar("loss", loss.item(), niter)
                for name, param in h.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(name + "_grad", param.grad, niter)
                        writer.add_histogram(name + "data", param, niter)
            niter +=1 
        train_acc /= (count +1)
        loss_avg /= (count +1)
       
        logger.debug(f"epoch : {i}\tloss avg: {loss_avg.item()}")     
            
#regression loss            
def _with_Dens_r(Dens_net, r,  logits, target, z, criterion):
    den_ratio = torch.exp(-Dens_net(z) *r).flatten()
    loss = (den_ratio*criterion(logits.flatten(), target)).mean()
    return loss
def _without_Dens_r(Dens_net, r, logits, target, z, criterion):
    loss = criterion(logits.flatten(), target).mean()
    return loss    


def _class_h(args, h, Rnet, Dens_net, optimal_h, source_loader, savefile, criterion):
    print("pretraining predictor h begin")
    h = h.to(args.device)
    Rnet = Rnet.to(args.device)
    Dens_net = Dens_net.to(args.device)
    Rnet.eval()
    Dens_net.eval()
    h.train()

    logger = logging.getLogger("logger{}".format(args.Method))
    logger.setLevel(logging.DEBUG)
    file_name = os.path.join(savefile, 'train.log')
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.info(f"Start CVRTL pretraining predictor h in {args.Method} on dataset {args.data_name} with {args.split} splits in random: {args.seed}.")
    logger.info(f"the epoch is {args.epochs}, batch size is {args.bths}.")
    logger.info(f"Training with no gpu: {args.nocuda}.")
    writer = SummaryWriter(savefile)
    niter = 0

    process = _without_Rnet if args.Method in ["Method_B", "Method_C", "Method_D"] else _with_Rnet
    # _criterion = _without_Dens_c if args.Method in ["Method_B"] else _with_Dens_c
    #To explore the representation only
    _criterion = _without_Dens_c if args.Method in ["Method_C", "Method_E"] else _with_Dens_c
    for i in trange(args.epochs):
        train_acc = 0
        loss_avg=0
        for count, (data,target) in enumerate(source_loader):
            target = target.long()
            data = data.to(args.device)
            target = target.to(args.device)
            # Method D not use representation
            if args.Method == "Method_D":
                re = data
            else:    
                re = Rnet(data)
            # process is to choose weather to use representation in density ratio
            logits = h(re)
            z = process(Rnet, data)[0]
            loss = _criterion(Dens_net, args.r, logits, target, z, criterion)
            top1 = accuracy(logits, target, topk=(1,))
            train_acc += top1[0]
            loss_avg += loss
            optimal_h.zero_grad()
            loss.backward()
            optimal_h.step()
            if niter % 100 == 0:
                writer.add_scalar("loss", loss.item(), niter)
                writer.add_scalar("top1 accuracy", top1[0], niter )
                for name, param in h.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(name + "_grad", param.grad, niter)
                        writer.add_histogram(name + "data", param, niter)
            niter +=1  
        train_acc /= (count +1)
        loss_avg /= (count +1)
       
        logger.debug(f"epoch : {i}\tloss avg: {loss_avg.item()}\ttrain_acc:{train_acc.item()}") 

#class loss
def _with_Dens_c(Dens_net, r, logits, target, z, criterion):
    den_ratio = torch.exp(-Dens_net(z) *r).flatten()
    loss = (den_ratio*criterion(logits, target)).mean()
    return loss

def _without_Dens_c(Dens_net, r, logits, target, z, criterion):
    loss = criterion(logits, target).mean()
    return loss 

# -----------------------------------------------------------------------------
# eval predictor h
def eval(args,h, Rnet, target_testloader, result_file_name):
    Rnet = Rnet.to(args.device)
    Rnet.eval()
    h  = h.to(args.device)
    h.eval()
    test_acc_av = []
    test_results=[]
    _criterion = acc_reg if args.task == "regression" else acc_class
    for i in trange(args.epochs):
        test_acc= 0
        for count, (data, target) in enumerate(target_testloader):
            data = data.to(args.device)
            target = target.to(args.device)
            if args.Method != "Method_D":
                re = Rnet(data)
            else:
                re = data
            logits = h(re)
            top1= _criterion(logits, target)
            test_acc += top1
        test_acc /= (count +1)
        test_acc_av.append(test_acc.item())
        test_results.append({"epoch": i, "test_acc":test_acc.item()})
    test_results.append({"epoch": "test_acc_mean", "test_acc": np.mean(test_acc_av)}) 
    test_results.append({"epoch": "test_acc_std", "test_acc": np.std(test_acc_av)})     

    with open(result_file_name, "w", newline='') as f:
        filedname = ["epoch", "test_acc"]
        writer = csv.DictWriter(f, filedname)
        writer.writeheader()
        writer.writerows(test_results)

def acc_reg(logits, target):
    top1 = (torch.square(logits - target)).mean()
    return top1
def acc_class(logits, target):
    top1 = accuracy(logits, target, topk=(1,))
    return top1[0]

    
    
#-----------------------------------------------------------------------------------------
# tools functions
def to_onehot(target):
    Y = np.ravel(target.cpu().numpy()).astype(int)
    Y_train = np.zeros((Y.shape[0], Y.max() - Y.min() +1))
    Y_train[np.arange(Y.size), Y-Y.min()] = 1
    target_onehot = torch.from_numpy(Y_train.astype(np.float32))
    return target_onehot

def cor(X, Y, n, device):
    # 计算distance correlation,矩阵x和Y之间的相关性
    DX = pairwise_distance(X)
    DY = pairwise_distance(Y)
    J = (torch.eye(n) - torch.ones(n, n)/ n).to(device)
    RX = J @ DX @ J
    RY = J @ DY @ J
    covXY = torch.mul(RX,RY).sum()/(n*n)
    covX = torch.mul(RX, RX).sum()/(n*n)
    covY = torch.mul(RY, RY).sum()/(n*n)
    return covXY/torch.sqrt(covX*covY + EPSILON)
    
def pairwise_distance(x, y = None):
    x_norm = (x**2).sum(1).view(-1,1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1,-1)
    else:
        y = x
        y_norm = x_norm.view(1,-1)
    dist = x_norm + y_norm - 2.0*torch.mm(x, torch.transpose(y, 0, 1))
    return dist

def test(args, epoch, Rnet, testloader):
    Rnet = Rnet.to(args.device)
    Rnet.eval()
    dCor_loss = 0
    with torch.no_grad():
        for data, target in testloader:
            data = data.to(args.device)
            target = target.to(args.device)
            target_onehot = to_onehot(target).to(args.device)
            latent = Rnet(data)
            dCor_loss += cor(latent, target_onehot, data.shape[0], args.device)
    dCor_loss /= len(testloader)
    print('\nTest set: dCor_loss:, {:.4f} \n'.format(
        dCor_loss))
    
# -----------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res    
    
def corss_entropy_loss_vec(logits, labels):
    exp_logits = torch.exp(logits)
    softmax = exp_logits/exp_logits.sum(dim=1, keepdim=True)
    prob = torch.gather(softmax, 1, labels.unsqueeze(1))
    loss = -torch.log(prob)

    return loss








