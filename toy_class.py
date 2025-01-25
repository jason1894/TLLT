"""This script include the pretrain process to get the trained representation model $R$,
and the density ration model $r$.
"""
import os

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import argparse

from dataset.data_gen import *
# from utils.util import 
from utils.random_seed import set_seed
from model.models import Densenet, Discriminator, Predict_h
from covariateshiftTL import Representation, Density_ratio, CovshiftTL

# setting args
parser = argparse.ArgumentParser(description="Python CVRTL ")
parser.add_argument("--task", type=str, default="classifiaction", help= "the task name")
parser.add_argument("--source_domain", type = str, nargs= '+',  help='source_domains')
parser.add_argument("--target_domain", type = str, nargs= '+',  help='target_domains')
parser.add_argument("--data", metavar="DOR", default='./CVRTL/dataset', help="path to dataset")
parser.add_argument("--data_name", default="circles", help="data name")
parser.add_argument("--nsamples", type = int,  default=10000, help="data name")
parser.add_argument("--lr", type=float, default=1.0, help="learning rate")
parser.add_argument("--lambdaa", type=float, default=1.0, help="the trade off parameter in representation learning")
parser.add_argument("--split", default=0.3)
parser.add_argument("--bths", type=int, default=64, help="batch size")
parser.add_argument("--epochs", type=int, default=200, help="the iter epoch number")
parser.add_argument("--depth", type=int, default=10, help="the depth of densnet")
parser.add_argument("--latdim", type=int, default=2, help="the representation dimension")
parser.add_argument("--r", type=float, default=1.0, help="the the regularization number")
parser.add_argument("--nocuda", action='store_true', help="disable the cuda")
parser.add_argument("--seed", type = int, default=0, help="the random seed")
parser.add_argument("--time", type = int, default=1, help="the loop round")
parser.add_argument("--Method", type = str, default="Method_A",  help='The transfer metod')
parser.add_argument('--device',default= torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

SEED = 2324
set_seed(SEED)
ARTI_DATA= ['Blood', 'Blood_normalize',
            'BreastCancer', "BreastCancer_normalize",
            "Haberman", "Haberman_normalize",
            "Ringnorm", "Ringnorm_normalize"]

def main():
    args = parser.parse_args()
    # check the device
    args = parser.parse_args()

    if not args.nocuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # generate the dataset
    if args.data_name == "circles" or args.data_name == "moon":
        source_train, target_train, source_test, target_test = gendata(args.data_name, args.nsamples, random_seed =args.seed, noise= 0.1, factor=0.6, psudim =1024, bias=True, alpha=0.95)
    elif args.data_name == "breast_cancer":   
        source_train, target_train, source_test, target_test = getbreast(random_seed = args.seed, feature_n = 5, psudim =400, alpha=0.95)
    elif args.data_name in ARTI_DATA:
        source_train, target_train, source_test, target_test = get_arti_data(args.data_name, random_seed = args.seed, feature_n = 0, psudim =1024, alpha=0.95)
    else:
        raise ValueError("there is no such data set!")
    
    source_trainloader = DataLoader(source_train, batch_size=args.bths, shuffle=True, drop_last=True)
    source_testloader = DataLoader(source_test, batch_size=args.bths, shuffle=True)
    target_trainloader = DataLoader(target_train, batch_size=args.bths, shuffle=True, drop_last=True) 
    target_testloader = DataLoader(target_test, batch_size=args.bths, shuffle=True)
    
    rootpath = os.path.join(os.getcwd(), args.task, "pretrainedmodel", args.data_name, str(args.time))
    if not os.path.exists(rootpath ):
        os.makedirs(rootpath)
        
    # pretrain the represent model
    # Keep the Renet is same in all three methods
    Rnet = Densenet(growthrate=12, depth=args.depth, reduction=0.5, bottleneck=True, ndim=args.latdim)
    Disnet = Discriminator(ndim=args.latdim)
    optim_Rep  = optim.Adam(Rnet.parameters(), weight_decay = 1e-4)
    optim_Disnet = optim.Adam(Disnet.parameters(), weight_decay = 1e-4) 

    rep = Representation(args, Rnet, Disnet, optim_Rep, optim_Disnet)
    rep.train(source_trainloader, rootpath)


    # pretrain the density model
    if args.Method in ["Method_B","Method_D"]:
        Dens_net = Densenet(growthrate=12, depth=30, reduction=0.5, bottleneck=True, ndim=1)
    else:
        Dens_net = Discriminator(ndim=args.latdim) 
    optim_Dens = optim.Adam(Dens_net.parameters(), weight_decay = 1e-4)

    dens = Density_ratio(args, Dens_net, optim_Dens) 
    dens.train(rep.Rnet, source_trainloader, target_trainloader, rootpath)  

    # covariate shit trainsfer learning
    if args.Method != "Method_D":
        h = Predict_h(args.latdim, 128, 2)
    else:    
        h =  Densenet(growthrate=12, depth=30, reduction=0.5, bottleneck=True, ndim=2)
    optimal_h = optim.Adam(h.parameters(), lr = 1e-3, weight_decay=1e-4)
    
    covTL = CovshiftTL(args, h, optimal_h)
    covTL.train(source_trainloader, rep.Rnet, dens.Dens_net, rootpath)
    covTL.predict(target_testloader, rep.Rnet)



if __name__ == "__main__":
    main()
