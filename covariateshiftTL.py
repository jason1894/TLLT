"""
This scripts is the class of covariate shift transfer learning
"""
import os

import torch

from utils.util import *
from utils.random_seed import set_seed

def _save(modal, file_path):
        torch.save(modal.state_dict(), file_path) 

def _load(modal, file_path):
    modal.load_state_dict(torch.load(file_path))

"""
Representation class is to get the representation of data
"""
class Representation:
    def __init__(self, args, Rnet, Dnet, optimizerR, optimizerD):
        self.args = args
        self.Rnet = Rnet
        self.Dnet = Dnet
        self.optimizerR = optimizerR
        self.optimizerD = optimizerD

    def train(self, train_loader, root_path):
        file_path = os.path.join(root_path, "Representor")
        flilename_Ret = "Repnet_round_{}_depth{}_ep{:03}_bt{:03}_latdim{:04}_lr{}_lambda{}_r{}.pt.tar".format(self.args.time, self.args.depth, 
                                                                                            self.args.epochs, self.args.bths, self.args.latdim,
                                                                                              self.args.lr,self.args.lambdaa, self.args.r)
        dict_path = os.path.join(file_path, flilename_Ret)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if not os.path.exists(dict_path):
            # pretrain the representation function and save.
            pretrain_rep(args=self.args, Rnet=self.Rnet, Dnet=self.Dnet, train_loader= train_loader, optimizerR=self.optimizerR,
                        optimizerD=self.optimizerD, savefile = file_path)    
            _save(self.Rnet, dict_path)   
        else:
            print("{} already exist, just loadin!!".format(flilename_Ret))
            _load(self.Rnet, dict_path)     

"""
Density_ratio class is to get the Density ratio of source and target data
"""
class Density_ratio:
    def __init__(self, args, Dens_net, optim_Dens):
        self.args = args
        self.Dens_net = Dens_net
        self.optim_Dens = optim_Dens

    def train(self, Rnet, source_trainloader, target_trainloader, root_path):
        if self.args.Method != "Method_C":
            pretrain_den = pretrain_dens
        else:
            print("{} no need density ratio".format(self.args.Method))
            return    

        if self.args.Method !=  "Method_D":
            file_path = os.path.join(root_path, "Density_ratio", self.args.Method)
        else:
            #Method_D use the same density ratio with Method_B  
            file_path = os.path.join(root_path, "Density_ratio", "Method_B")
        flilename_Dens = "Denratio_round_{}_latdim{}_depth{}_ep{:03}_bt{:03}_lr{}_lambda{}_r{}.pt.tar".format(self.args.time, self.args.latdim, self.args.depth, self.args.epochs, self.args.bths,
                                                                                                     self.args.lr, self.args.lambdaa, self.args.r) 
        dict_path = os.path.join(file_path, flilename_Dens)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if not os.path.exists(dict_path):
            pretrain_den(self.args, Rnet, self.Dens_net, self.optim_Dens, source_trainloader, target_trainloader, file_path)  
            _save(self.Dens_net, dict_path)
        else:
            print("{} already exist, just loadin!!".format(flilename_Dens))
            _load(self.Dens_net, dict_path)    

"""
CovshiftTL class is to train the predictor h on source domian and test it on target domain
"""
class CovshiftTL:
    def __init__(self, args, h, optimal_h):
        self.args = args
        self.h = h
        self.optim_h = optimal_h
       
    def train(self, source_data, Rnet, Dnet, rootpath):
        file_path = os.path.join(rootpath, "Predictor_h", self.args.Method)
        filename_h = "h_latdim{}_round_{}_depth{}_ep{:03}_bt{:03}_lr{}_lambda{}_r{}.pt.tar".format(self.args.latdim, self.args.time, self.args.depth, self.args.epochs, self.args.bths,
                                                                                          self.args.lr, self.args.lambdaa, self.args.r)
        dict_path = os.path.join(file_path, filename_h)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if not os.path.exists(dict_path):
            pretrain_h(self.args, self.h, Rnet, Dnet, self.optim_h, source_data, file_path)
            _save(self.h, dict_path)
        else:
            print("{} already exist, just loadin!!".format(filename_h))
            _load(self.h, dict_path)


    def predict(self, target_data, Rnet):
        if self.args.source_domain and self.args.target_domain is not None:
            result_path = os.path.join(os.getcwd(),"Eval_resuls", self.args.task, self.args.data_name, "{}2{}".format("".join(self.args.source_domain), "".join(self.args.target_domain)), self.args.Method)
        else:
            result_path = os.path.join(os.getcwd(),"Eval_resuls", self.args.task, self.args.data_name, self.args.Method)    
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        file_name = "round{}_latdim{}_depth{}_epoch{}_batch{}_lr{}_lambda{}_r{}.csv".format(self.args.time,self.args.latdim, self.args.depth, self.args.epochs, self.args.bths,
                                                                              self.args.lr, self.args.lambdaa, self.args.r)
        result_file_name = os.path.join(result_path, file_name)
        eval(self.args, self.h, Rnet, target_data, result_file_name)