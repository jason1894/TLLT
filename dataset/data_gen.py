"""This script is to generate the simulaition datas"""
import os
import numpy as np

from sklearn.datasets import make_circles, make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split
from utils.random_seed import set_seed

import torch 
import torchvision.transforms as transforms
import torchvision.datasets as data
from torch.utils.data import TensorDataset

# dataname = ['circels', 'moon']

def _gendata(data_name, nsamples, random_seed, noise, factor, bias):
    if data_name == "circles":
        X, y = make_circles(n_samples=nsamples, random_state=random_seed, noise=noise, factor=factor)
    elif data_name == 'moon':
        X, y = make_moons(n_samples=nsamples, random_state=random_seed, noise=noise) 
    else:
        raise ValueError("There is no such dataname, plz check it!")
    if bias:
        X[:, 0] += 0.5
        X[:, 1] += 0.5
    return X, y 
     
def gendata(data_name, nsamples=1000, random_seed=0, noise=0.1, factor=0.5, psudim = 400, bias = False, alpha =0.5):
    # check wheather the data is already exist
    data_file = "{}_pdim{:03}_nois{:0.1}_factor{:0.2}_ns{:04}".format(data_name, psudim, noise,
                                                                      factor, nsamples)
    filepath = os.path.join(os.getcwd(),'dataset', 'simul_data', data_file)
    if os.path.exists(filepath):
        X = np.genfromtxt(os.path.join(filepath,'X.csv'), delimiter=",", dtype=float)
        y = np.genfromtxt(os.path.join(filepath,'y.csv'), delimiter=",", dtype=int)
    else:
        os.makedirs(filepath, exist_ok=True)
        X, y  = _gendata(data_name, nsamples, random_seed, noise, factor, bias)
        np.savetxt(os.path.join(filepath,'X.csv'), X, delimiter=',', fmt='%f')
        np.savetxt(os.path.join(filepath,'y.csv'), y, delimiter=',', fmt='%d')

    # shift the data
    source_X, target_X, source_y, target_y = covariate_shift(X, y ,feature_n=0, alpha=alpha)    

    # map the data from 2 dim into high dim
    if psudim > 2:
        x_dim = int(np.sqrt(psudim))
        y_dim = psudim // x_dim
        source_X = np.matmul(np.random.rand(x_dim*y_dim, 2), source_X.T).T
        source_X = source_X.reshape(source_X.shape[0], 1, x_dim, y_dim)
        target_X = np.matmul(np.random.rand(x_dim*y_dim, 2), target_X.T).T
        target_X = target_X.reshape(target_X.shape[0], 1, x_dim, y_dim)

    source_X_train, source_X_test, source_y_train, source_y_test = train_test_split(source_X, source_y, test_size=0.2, random_state=random_seed)
    target_X_train, target_X_test, target_y_train, target_y_test = train_test_split(target_X, target_y, test_size=0.2, random_state=random_seed)
    source_train = TensorDataset(torch.from_numpy(source_X_train).float(), torch.from_numpy(source_y_train).float())
    source_test = TensorDataset(torch.from_numpy(source_X_test).float(), torch.from_numpy(source_y_test).float())
    target_train = TensorDataset(torch.from_numpy(target_X_train).float(), torch.from_numpy(target_y_train).float())
    target_test = TensorDataset(torch.from_numpy(target_X_test).float(), torch.from_numpy(target_y_test).float())
    
    return source_train, target_train, source_test, target_test

    
    
def genmnist_usps(source_domain, target_domain, transform = False):
    def add_noise(x):
        noise = torch.randn_like(x)*0.2
        noise_x = x + noise
        return torch.clamp(noise_x, 0, 1)
    if transform:
        # transforms.Normalize((1.1251,), (0.3740))
        Transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomApply([transforms.GaussianBlur(kernel_size = 3, sigma = (0.1, 2.0))]),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1051,), (0.1340)),
                                        transforms.Lambda(lambda x : add_noise(x))])
    else:
        Transform = transforms.Compose([transforms.ToTensor()])

    mnist_train = data.MNIST(root = './dataset/mnist', train= True, download= True, transform= Transform)
    mnist_test = data.MNIST(root="./dataset/mnist", train=False, download=True, transform=Transform) 
    usps_train = data.USPS(root = './dataset/usps', train= True, download= True, transform= Transform)
    usps_test = data.USPS(root="./dataset/usps", train=False, download=True, transform=Transform) 


    if source_domain == "mnist" and target_domain == "usps":
        source_train = mnist_train
        source_test = mnist_test 
        target_train = usps_train
        target_test = usps_test
        return source_train, target_train, source_test, target_test
    elif source_domain == "usps" and target_domain == "mnist" :
        source_train = usps_train
        source_test = usps_test 
        target_train = mnist_train
        target_test = mnist_test
        return source_train, target_train, source_test, target_test
    else:
        raise ValueError("please assign right source and target domain!")



def gensin(nsamples, mean, std, rdim = 1,  pdim = 196, error = "Gaussian", random_seed = None):
    data_file = "{}_rdim{}_mean{:0.1}_std{:0.2}_ns{:04}".format("sin", rdim, mean,
                                                                      std, nsamples)
    filepath = os.path.join(os.getcwd(),'dataset', 'simul_data', data_file)

    # save or reloade the orignianl data，datatype is float
    if not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)
        X, y, fx = make_sin(n_samples=nsamples, std = std, mean=mean, random_seed=random_seed, error=error, rdim=rdim)
        np.savetxt(os.path.join(filepath,'X.csv'), X, delimiter=',', fmt='%f')
        np.savetxt(os.path.join(filepath,'y.csv'), y, delimiter=',', fmt='%f')
        np.savetxt(os.path.join(filepath,'fx.csv'), fx, delimiter=',', fmt='%f')
    else:
        X = np.genfromtxt(os.path.join(filepath,'X.csv'), delimiter=",", dtype=float)
        y = np.genfromtxt(os.path.join(filepath,'y.csv'), delimiter=",", dtype=float)
        fx = np.genfromtxt(os.path.join(filepath,'fx.csv'), delimiter=",", dtype=float)
        
    if pdim > rdim: 
        # high dim data with noise dim  
        x_dim = int(np.sqrt(pdim))
        y_dim = pdim // x_dim
        noise= np.random.rand(x_dim*y_dim, rdim)
        X = X.reshape(-1, rdim)
        X = np.matmul(noise, X.T).T
        X = X.reshape(nsamples, 1, x_dim, y_dim)
    else:    
        #one dim regression ,low instric dim data
        X = X.reshape(-1, rdim)
        y = y.reshape(-1, 1)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    test_data = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

    return train_data, test_data

def make_sin(n_samples , std , mean, random_seed = None, error = "Gaussian", rdim = 2):
    # make the regression data
    ind = torch.bernoulli(0.9*torch.ones([n_samples,1]))
    normal = torch.randn([n_samples, rdim])
    x = normal*std + mean
    errors ={'Gaussian':torch.normal(0,0.3,size=(n_samples,1)),
             'outline':ind*torch.normal(0,0.2,size =(n_samples,1)) + (1-ind)*(4*torch.rand(n_samples,1)+1),
             'skew':torch.from_numpy(np.random.exponential(0.3,(n_samples,1))),
             'nostationary':torch.abs(torch.cos(x.data[0][0]*np.pi))*torch.from_numpy(np.random.exponential(0.3,(n_samples,1))),
             't':torch.from_numpy(0.025*np.random.standard_t(2,[n_samples,1]))             
            }
    eps = errors[error]
    fx = _sin(x).float()
    y = fx + eps

    # transfer to numpy
    X = x.data.numpy()
    Y = y.data.numpy()
    FX = fx.data.numpy()

    return X, Y, FX

def _sin(input):
    dx = input.size(dim=1)
    t = torch.sum(input,dim = 1)
    fx = torch.sin(0.5*np.pi*t/dx) - 0.5
    output = -fx.reshape(-1,1)
    return output


def getbreast(random_seed,feature_n, psudim, alpha):
    """https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic ;
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html;
    Feature: 30,
    nsample: 569
    """
    X, y  = load_breast_cancer(return_X_y=True)
    source_X, target_X, source_y, target_y = covariate_shift(X, y ,feature_n=feature_n, alpha=alpha)

    #TODO rase 30 dim to 400 dim will casue issue
    if psudim > 30:
        x_dim = int(np.sqrt(psudim))
        y_dim = psudim // x_dim

        source_X = np.matmul(np.random.rand(x_dim*y_dim, 30), source_X.T).T
        source_X = source_X.reshape(source_X.shape[0], 1, x_dim, y_dim)

        target_X = np.matmul(np.random.rand(x_dim*y_dim, 30), target_X.T).T
        target_X = target_X.reshape(target_X.shape[0], 1, x_dim, y_dim)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    # train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    # test_data = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    source_X_train, source_X_test, source_y_train, source_y_test = train_test_split(source_X, source_y, test_size=0.2, random_state=random_seed)
    target_X_train, target_X_test, target_y_train, target_y_test = train_test_split(target_X, target_y, test_size=0.2, random_state=random_seed)
    source_train = TensorDataset(torch.from_numpy(source_X_train).float(), torch.from_numpy(source_y_train).float())
    source_test = TensorDataset(torch.from_numpy(source_X_test).float(), torch.from_numpy(source_y_test).float())
    target_train = TensorDataset(torch.from_numpy(target_X_train).float(), torch.from_numpy(target_y_train).float())
    target_test = TensorDataset(torch.from_numpy(target_X_test).float(), torch.from_numpy(target_y_test).float())
    return source_train, target_train, source_test, target_test
    
def get_arti_data(data_name, random_seed, feature_n, psudim, alpha):
    """
    https://github.com/MachineLearningBCAM/MRCs-for-Covariate-Shift-Adaptation/tree/main/Datasets
    """

    root = "./dataset/arti_data"
    file_path = os.path.join(root,"{}.csv".format(data_name))
    if not os.path.exists(file_path):
        raise ValueError("there is no such dataset: {}, plz check!".format(file_path))
    
    data_set = np.genfromtxt(file_path, delimiter=",")
    data_dim = data_set.shape[1] -1
    X = data_set[:, : data_dim]
    y_orginal = data_set[:, -1]
    y = y_orginal-1
    

    source_X, target_X, source_y, target_y = covariate_shift(X, y ,feature_n=feature_n, alpha=alpha)

    #TODO rase 30 dim to 400 dim will casue issue
    if psudim > data_dim:
        x_dim = int(np.sqrt(psudim))
        y_dim = psudim // x_dim

        source_X = np.matmul(np.random.rand(x_dim*y_dim, data_dim), source_X.T).T
        source_X = source_X.reshape(source_X.shape[0], 1, x_dim, y_dim)

        target_X = np.matmul(np.random.rand(x_dim*y_dim, data_dim), target_X.T).T
        target_X = target_X.reshape(target_X.shape[0], 1, x_dim, y_dim)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    # train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    # test_data = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    source_X_train, source_X_test, source_y_train, source_y_test = train_test_split(source_X, source_y, test_size=0.2, random_state=random_seed)
    target_X_train, target_X_test, target_y_train, target_y_test = train_test_split(target_X, target_y, test_size=0.2, random_state=random_seed)
    source_train = TensorDataset(torch.from_numpy(source_X_train).float(), torch.from_numpy(source_y_train).float())
    source_test = TensorDataset(torch.from_numpy(source_X_test).float(), torch.from_numpy(source_y_test).float())
    target_train = TensorDataset(torch.from_numpy(target_X_train).float(), torch.from_numpy(target_y_train).float())
    target_test = TensorDataset(torch.from_numpy(target_X_test).float(), torch.from_numpy(target_y_test).float())
    return source_train, target_train, source_test, target_test


def covariate_shift(X, y , feature_n = 0, alpha = 0.95, n = None, t = None):
    """ To generate the covariate shift source and target data with features shift,
    https://github.com/MachineLearningBCAM/MRCs-for-Covariate-Shift-Adaptation
    """  
    nsample = X.shape[0]
    median = np.median(X[:, feature_n])
    source_indx =[]
    target_indx =[]

    " shift with P_s = 0.7, p_t = 0.3 for x_i < median."
    for i in range(nsample):
        if X[i, feature_n] <= median:
            if np.random.rand() <= alpha:
                source_indx.append(i)
            else:
                target_indx.append(i)
        else:
            if np.random.rand() <= 1-alpha:
                source_indx.append(i)
            else:
                target_indx.append(i)
    source_X = X[source_indx]  
    source_y = y[source_indx]  
    target_X = X[target_indx]  
    target_y = y[target_indx]

    return source_X, target_X, source_y, target_y

    













