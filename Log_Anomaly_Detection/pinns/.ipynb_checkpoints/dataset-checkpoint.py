import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import itertools


class DomainDataset(Dataset):
    def __init__(self, xmin, xmax, n, rand = True, shuffle = True, period = 1):
        self.xmin = np.array(xmin, dtype="f")
        self.xmax = np.array(xmax, dtype="f")
        self.dim = len(xmin)
        self.n = n
        self.side_length = self.xmax - self.xmin
        self.volume = np.prod(self.side_length)
        self.rand = rand
        self.shuffle = shuffle
        self.period = period
        if self.rand:
            self.compute_items_rand()
        else:
            self.compute_items_sequential()   
        if shuffle:
            self.counter = 0 

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        ret = self.x[idx]
        if self.shuffle:
            self.counter += 1
            if self.counter == self.__len__()*self.period:
                self.compute_items_rand()
                self.counter = 0
        return ret
    
    def compute_items_sequential(self):
        n_points_per_axis = int(np.ceil(self.n ** (1/self.dim)))
        xi = []
        for i in range(self.dim):
            s = np.linspace(self.xmin[i], self.xmax[i], num = n_points_per_axis + 1, endpoint=False)[1:]
            xi.append(s)
        self.x = np.array(list(itertools.product(*xi)), dtype = "f")
        return

    def compute_items_rand(self):
        n_points_per_axis = int(np.ceil(self.n ** (1/self.dim)))
        xi = []
        for i in range(self.dim):
            s = np.random.uniform(low=self.xmin[i], high=self.xmax[i], size=(n_points_per_axis, ))
            xi.append(s)
        self.x = np.array(list(itertools.product(*xi)), dtype = "f")
        return
    
    def __str__(self):
        s = f"DomainDataset({self.xmin}, {self.xmax}, n={self.n}, rand={self.rand}, shuffle={self.shuffle}, period={self.period})"
        return s
    
    def _compute_items(self):    
        dx = (self.volume / self.n) ** (1 / self.dim)
        xi = []
        for i in range(self.dim):
            ni = int(np.ceil(self.side_length[i] / dx))
            s = np.linspace(self.xmin[i],self.xmax[i],num=ni + 1,endpoint=False)[1:]
            xi.append(s)
        x = np.array(list(itertools.product(*xi)), dtype="f")
        #x = np.array(xi)
        """ if self.n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(self.n, len(x))
            ) """
        self.x = x
    

class ICDataset(DomainDataset):
    def __init__(self, xmin, xmax, n, rand=True, shuffle = True, period = 1):
        super().__init__(xmin, xmax, n, rand = rand, shuffle = shuffle, period = period)
        
    def compute_items_sequential(self):
        n_points_per_axis = int(np.ceil(self.n ** (1/self.dim)))
        xi = []
        for i in range(self.dim):
            s = np.linspace(self.xmin[i], self.xmax[i], num = n_points_per_axis + 1, endpoint=False)[1:]
            xi.append(s)
        xi.append([0.0])
        self.x = np.array(list(itertools.product(*xi)), dtype = "f")
        return

    def compute_items_rand(self):
        n_points_per_axis = int(np.ceil(self.n ** (1/self.dim)))
        xi = []
        for i in range(self.dim):
            s = np.random.uniform(low=self.xmin[i], high=self.xmax[i], size=(n_points_per_axis, ))
            xi.append(s)
        xi.append([0.0])
        self.x = np.array(list(itertools.product(*xi)), dtype = "f")
        return
    
    def __str__(self):
        s = f"ICDataset({self.xmin}, {self.xmax}, n={self.n}, rand={self.rand}, shuffle={self.shuffle}, period={self.period})"
        return s
    
    def _compute_items(self):
        dx = (self.volume / self.n) ** (1 / self.dim)
        xi = []
        for i in range(self.dim):
            ni = int(np.ceil(self.side_length[i] / dx))
            s = np.linspace(self.xmin[i],self.xmax[i],num=ni + 1,endpoint=False)[1:]
            xi.append(s)
        ni = int(np.ceil(self.side_length[0] / dx))
        xi.append([0.0]*ni)          

        x = np.array(list(itertools.product(*xi)), dtype="f")
        #if self.n != len(x):
            #print(
            #    "Warning: {} points required, but {} points sampled.".format(self.n, len(x))
            #)
        self.x = x


class ValidationDataset(DomainDataset):
    def __init__(self, xmin, xmax, n, rand=True, shuffle=False, period=1):
        super().__init__(xmin, xmax, n, rand, shuffle, period)
    
    def compute_items_sequential(self):
        n_points_per_axis = self.n
        xi = np.zeros(shape = (n_points_per_axis, 1))
        for i in range(self.dim):
            s = np.linspace(self.xmin[i], self.xmax[i], num = n_points_per_axis, endpoint = True).reshape(-1, 1)
            xi = np.hstack((xi, s))
        self.x = np.array(xi[:, 1:], dtype="f")
        return

    def compute_items_rand(self):
        n_points_per_axis = self.n
        xi = np.zeros(shape = (n_points_per_axis, 1))
        for i in range(self.dim):
            s = np.random.uniform(low=self.xmin[i], high=np.nextafter(self.xmax[i], self.xmax[i]+1), size=(n_points_per_axis, )).reshape(-1, 1)
            xi = np.hstack((xi, s))
        self.x = np.array(xi[:, 1:], dtype="f")
        return
    
    def __str__(self):
        s = f"ValidationDataset({self.xmin}, {self.xmax}, n={self.n}, rand={self.rand}, shuffle={self.shuffle}, period={self.period})"
        return s
    
class ValidationICDataset(DomainDataset):
    def __init__(self, xmin, xmax, n, rand=True, shuffle=False, period=1):
        super().__init__(xmin, xmax, n, rand, shuffle, period)
    
    def compute_items_sequential(self):
        n_points_per_axis = self.n
        xi = np.zeros(shape = (n_points_per_axis, 1))
        for i in range(self.dim):
            s = np.linspace(self.xmin[i], self.xmax[i], num = n_points_per_axis, endpoint = True).reshape(-1, 1)
            xi = np.hstack((xi, s))
        z = np.zeros((n_points_per_axis, 1))
        xi = np.hstack((xi, z))
        self.x = np.array(xi[:, 1:], dtype="f")
        return

    def compute_items_rand(self):
        n_points_per_axis = self.n
        xi = np.zeros(shape = (n_points_per_axis, 1))
        for i in range(self.dim):
            s = np.random.uniform(low=self.xmin[i], high=np.nextafter(self.xmax[i], self.xmax[i]+1), size=(n_points_per_axis, )).reshape(-1, 1)
            xi = np.hstack((xi, s))
        z = np.zeros((n_points_per_axis, 1))
        xi = np.hstack((xi, z))
        self.x = np.array(xi[:, 1:], dtype="f")
        return
    
    def __str__(self):
        s = f"ValidationICDataset({self.xmin}, {self.xmax}, n={self.n}, rand={self.rand}, shuffle={self.shuffle}, period={self.period})"
        return s


class BCDataset(DomainDataset):
    def __init__(self, xmin, xmax, n):
        super().__init__(xmin, xmax, n)
    
    #TO BE IMPLEMENTED
