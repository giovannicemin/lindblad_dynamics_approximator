import torch
import numpy as np

def one_negative(x):
        it = False
        if any(t < 0 for t in x):
                it = True
        return it

def simplexion(x):
        I =[]
        while one_negative(x): 
                s_x_i = x.sum()-x[I].sum()
                for i in range(len(x)):     
                        if not i in I:
                                x[i] = x[i]+(1-s_x_i)/(len(x) - len(I)) 
                                if x[i]<0:
                                          I = list(set(I)|set([i]))             
                        else:
                                x[i] = 0
        
        return x

