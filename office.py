import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset
import argparse

from utils.random_seed import set_seed
from covariateshiftTL import Representation, Density_ratio, CovshiftTL
from model.models import ResNetBackbone, Discriminator, Predict_h

# setting args
parser = argparse.ArgumentParser(description="Python CVRTL ")
parser.add_argument("--data_name", default="office", help="data name")
parser.add_argument("--task", type=str, default="classifiaction", help= "the task name")
parser.add_argument("--source_domain", type = str, nargs= '+',  help='source_domains')
parser.add_argument("--target_domain", type = str, nargs= '+',  help='target_domains')
parser.add_argument("--split", default=0.2)
parser.add_argument("--lr", type=float, default=1.0, help="learning rate")
parser.add_argument("--lambdaa", type=float, default=1.0, help="the trade off parameter in representation learning")
parser.add_argument("--bths", type=int, default=64, help="batch size")
parser.add_argument("--epochs", type=int, default=200, help="the iter epoch number")
parser.add_argument("--depth", type=int, default=10, help="the depth of densnet")
parser.add_argument("--latdim", type=int, default=30, help="the representation dimension")
parser.add_argument("--r", type=float, default=1.0, help="the the regularization number")
parser.add_argument("--nocuda", action='store_true', help="disable the cuda")
parser.add_argument("--seed", type = int, default=0, help="the random seed")
parser.add_argument("--time", type = int, default=1, help="the loop round")
parser.add_argument("--Method", type = str, default="Method_A",  help='The transfer metod')
# Method B is the origianl method, Mathod_C is without density ratio, Method_A is our porposed fearure density ratio method.
parser.add_argument('--device',default= torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
parser.add_argument('--pretrained', action='store_true', help='whether to load the pretrained model.')

def load_data(args):
    transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])  # 转换为张量
            ])

    Caltech_data_dir = "./dataset/Office_Caltech_DA_Dataset/Caltech" 
    amazon_data_dir = "./dataset/Office_Caltech_DA_Dataset/amazon" 
    dslr_data_dir = "./dataset/Office_Caltech_DA_Dataset/dslr" 
    webcam_data_dir = "./dataset/Office_Caltech_DA_Dataset/webcam" 

    Caltech = ImageFolder(root=Caltech_data_dir, transform=transform)
    amazon = ImageFolder(root=amazon_data_dir, transform=transform)
    dslr = ImageFolder(root=dslr_data_dir, transform=transform)
    webcam = ImageFolder(root=webcam_data_dir, transform=transform)
    data_dic = {"A":amazon, 
                "D":dslr,
                "W":webcam,
                "C":Caltech}
    source_data = ConcatDataset([data_dic[i] for i in args.source_domain])
    target_data = ConcatDataset([data_dic[i] for i in args.target_domain])
    source_data_train, source_data_test = train_test_split(source_data, test_size=args.split, random_state=0)
    target_data_train, target_data_test = train_test_split(target_data, test_size=args.split, random_state=0)
    kwargs = {'num_workers': 2, 'pin_memory': True}  
    source_train_loader = DataLoader(source_data_train, batch_size=args.bths, shuffle=True, **kwargs)
    source_test_loader  = DataLoader(source_data_test, batch_size=args.bths, shuffle=True, **kwargs)
    target_train_loader = DataLoader(target_data_train, batch_size=args.bths, shuffle=True, **kwargs)
    target_test_loader = DataLoader(target_data_test, batch_size=args.bths, shuffle=True, **kwargs)
    return source_train_loader, source_test_loader, target_train_loader, target_test_loader


def main():
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    if not args.nocuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1 
    
    # loader the data set ['C', 'D', 'W]--->['A']
    # source_domain = ['C', 'D', 'W']
    # target_domain = ['A']
    source_train_loader, source_test_loader, target_train_loader, target_test_loader\
     = load_data(args)
    
    # model setting
    rootpath = os.path.join(os.getcwd(),"pretrainedmodel", args.data_name, "{}2{}".format("".join(args.source_domain), "".join(args.target_domain)), str(args.time))
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    
    # pretraining R
    #Keep the Renet is same in all three methods 
    Rnet = ResNetBackbone(ndim=args.latdim, pretrained= args.pretrained)
    optim_Rep = optim.Adam(params=Rnet.parameters(), weight_decay=1e-4)
    Disnet = Discriminator(ndim=args.latdim)
    optim_Disnet = optim.Adam(Disnet.parameters(), weight_decay = 1e-4)

    rep = Representation(args, Rnet, Disnet, optim_Rep, optim_Disnet)
    rep.train(source_train_loader, rootpath)

    # pretrain the density model
    if args.Method in ["Method_B","Method_D"]:
        Dens_net = ResNetBackbone(ndim=1)
    else:
        Dens_net = Discriminator(ndim=args.latdim) 
    optim_Dens = optim.Adam(Dens_net.parameters(), weight_decay = 1e-4)

    dens = Density_ratio(args, Dens_net, optim_Dens) 
    dens.train(rep.Rnet, source_train_loader, target_train_loader, rootpath)  

    # covariate shit trainsfer learning
    if args.Method != "Method_D":
        h = Predict_h(args.latdim, 128, 10)
    else:
        h = models.resnet18(weights=None)
        # in_features = h.classifier.in_features
        # h.classifier = nn.Linear(in_features, 10)  
        in_features = h.fc.in_features
        h.fc = nn.Linear(in_features, 10)
    optimal_h = optim.Adam(h.parameters(), lr = 1e-3, weight_decay=1e-4)
    
    covTL = CovshiftTL(args, h, optimal_h)
    covTL.train(source_train_loader, rep.Rnet, dens.Dens_net, rootpath)
    covTL.predict(target_test_loader, rep.Rnet)



if __name__ == "__main__":
    main()    
