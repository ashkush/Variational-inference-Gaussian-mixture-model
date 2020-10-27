#!/usr/bin/env python
# coding: utf-8




"""Bayesian inference of gaussian mixture model"""

from sklearn.datasets import make_spd_matrix
import numpy as np
from scipy.special import psi, gammaln, multigammaln
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
import itertools
from scipy import linalg
import matplotlib as mpl
import numpy.random as rnd





class prior_initilisation():
    
    def __init__(self,data):
        self.alpha_o = 1e-2
        self.D = data.shape[1]
        self.w_o = np.eye(self.D) 
        self.nu_o = self.D
        self.mu_o = np.ones(self.D)
        self.beta_o = 1




class VI_GMM(prior_initilisation):
    
    def __init__(self,data,components,max_iter = 500,threshold = 1e-5):
        
        prior_initilisation.__init__(self,data)
        self.data = data
        self.N = data.shape[0]
        self.k = components
        self.alpha = np.ones(self.k)
        self.beta = np.ones(shape=self.k)
        self.nu = np.array([self.D]*self.k)
        self.m = np.random.multivariate_normal(np.ones(self.D),np.eye(self.D),size = self.k)
        self.w_inv = np.array([np.eye(self.D)]*self.k)
        self.r_nk = np.random.randn(self.N,self.k)
        self.max_iter = max_iter
        self.threshold = threshold
        self.n_iters = 0
        self.weights = []
    
    def update_alpha(self,nk):

        for k in range(self.k):
            self.alpha[k] = self.alpha_o + nk[k]
    
    def update_beta(self,nk):
        
        for k in range(self.k):
            self.beta[k] = self.beta_o + nk[k]
            
    def update_m(self,xk,nk):
        
        for k in range(self.k):
            second_term = np.zeros(self.D)
            second_term += nk[k]*xk[k]
            
            self.m[k] = (self.beta_o*self.mu_o+second_term)/self.beta[k]
                
    def update_w(self,nk,xk,sk):
        
        for k in range(self.k):
            second_term = np.zeros(shape = (self.D,self.D))
            c = xk[k]-self.mu_o
            second_term+= nk[k]*sk[k]
            second_term+= (self.beta_o*nk[k]/(self.beta_o+nk[k]))*np.outer(c,c)
                             
            
            self.w_inv[k] = np.linalg.inv(self.w_o)+second_term
            self.w_inv[k] = np.linalg.inv(self.w_inv[k])
                                                                                             
        
    def update_nu(self,nk):
        for k in range(self.k):
            self.nu[k] = self.nu_o+nk[k]
    
                                                                                             
    def update_r_nk(self):
        
        for n in range(self.N):
            for k in range(self.k):
                self.r_nk[n][k] = psi(self.alpha[k])-psi(np.sum(self.alpha)) + .5*self.D*np.log(2)
                
                for d in range(1,self.D+1):
                    self.r_nk[n][k]+=.5*psi((self.nu[k]+1-d)/2)
                
                self.r_nk[n][k] += .5*np.log(np.linalg.det(self.w_inv[k]))
                self.r_nk[n][k] -= .5*(self.D/self.beta[k] + self.nu[k]*(np.dot(np.dot(self.data[n]-self.m[k],self.w_inv[k]),self.data[n]-self.m[k])))
                self.r_nk[n][k] -= .5*self.D*np.log(2*math.pi) 
                
                self.r_nk[n][k] = np.exp(self.r_nk[n][k])
                
        
        # normalisation
        self.r_nk = self.r_nk/self.r_nk.sum(axis = 1)[:,np.newaxis]
    
    # Evidence lower bound computation
    
    def ln_pi(self,k):
        return psi(self.alpha[k])-psi(sum(self.alpha))
    
    def ln_precision(self,k):
        s = sum([psi((self.nu[k]+1-d)/2) for d in range(1,self.D+1)])
        s+=self.D*np.log(2)
        s+= np.log(np.linalg.det(self.w_inv[k]))  
        return s
    
    def b_w_v(self,k):
        return -0.5*self.D*self.nu[k]*np.log(2.0) - 0.5*self.nu[k]*np.log(np.linalg.det(self.w_inv[k])) - multigammaln(self.nu[k], self.D) 
        
        
    def entropy_wishart(self,k):
        return -self.b_w_v(k)-.5*(self.nu[k]-self.D-1)*self.ln_precision(k)+self.nu[k]*self.D/2
        
    
    def elbo_computation(self,nk,sk,xk):
        
        # E[ln p(X|Z,u,L)]
        def first_term(nk,sk,xk):
            expectation = 0
            for k in range(self.k):
                expectation+=nk[k]*(self.ln_precision(k)-self.D/self.beta[k]-self.nu[k]*np.trace(np.dot(sk[k],np.linalg.inv(self.w_inv[k]))-                                         self.nu[k]*np.dot(np.dot(xk[k]-self.m[k],self.w_inv[k]),xk[k]-self.m[k])-                                          self.D*np.log(2*math.pi)))
                
            return expectation*.5
        
        
        # E[ln p(Z|π)] 
        def second_term():
            s=0
            for n in range(self.N):
                for k in range(self.k):
                    s+=self.r_nk[n][k]*self.ln_pi(k)
            return s  
        
        # E[ln p(π)]
        def third_term():
            a = gammaln(self.alpha_o*self.k)-self.k*gammaln(self.alpha_o)
            b = (self.alpha_o-1)*sum([self.ln_pi(k) for k in range(self.k)])
            return a+b
        
        # E[ln p(μ, Λ)] 
        def fourth_term():
            a= 0
            for k in range(self.k):
                a += self.D*np.log(self.beta_o/(2*math.pi))+self.ln_precision(k)-self.D*self.beta_o/self.beta[k]-                     self.beta_o*self.nu[k]*np.dot(np.dot(self.m[k]-self.mu_o,self.w_inv[k]),self.m[k]-self.mu_o)
            
            a = a*.5
            
            b_w_v = np.linalg.det(self.w_o)**(-self.nu_o/2)*(2**(-self.nu_o*self.D/2))*(multigammaln(self.nu_o,self.D))**-1
            a+=self.k*np.log(b_w_v)
            a+=.5*(self.nu_o-self.D-1)*sum([self.ln_precision(k) for k in range(self.k)])
            a-=.5*sum([self.nu[k]*np.trace(np.dot(self.w_o,self.w_inv[k])) for k in range(self.k)])
            
            return a
        
        # E[ln q(Z)]
        def fifth_term():
            return sum([self.r_nk[n][k]*np.log(self.r_nk[n][k]) for n in range(self.N) for k  in range(self.k)])
            
        
        
        # E[ln q(π)] 
        def sixth_term():
            a = sum([(self.alpha[k]-1)*self.ln_pi(k) for k in range(self.k)])
            b = gammaln(sum(self.alpha))-sum([gammaln(self.alpha[k]) for k in range(self.k)])
            return a+b
        
        
        # E[ln q(μ, Λ)] 
        def seventh_term():
            return sum([.5*self.ln_precision(k)+self.D*.5*np.log(self.beta[k]/(2*np.pi))-self.D*.5-self.entropy_wishart(k) for k in range(self.k)])
            
        
        a=first_term(nk,sk,xk)
        b=second_term()
        c=third_term()
        d=fourth_term()
        e=fifth_term()
        f=sixth_term()
        g=seventh_term()
                
        
        
        
        return a+b+c+d-e-f-g
            
                    
    def mixture_weights(self,nk):
        for k in range(self.k):
            self.weights.append(self.alpha[k]+nk[k])
        self.weights = self.weights/sum(self.weights)
    
    
    def fit(self):
        lower_bound=[]
        rsp = {}
        while True:
            
            # E-step
            
            
            self.update_r_nk()
            
            
            
            # M-step
            nk = np.sum(self.r_nk,0)
            xk = np.diag(1/nk).dot(np.transpose(self.r_nk).dot(self.data))
            sk = np.zeros((self.k,self.D,self.D))
            for k in range(self.k):
                for n in range(self.N):
                    t = self.data[n]- xk[k]
                    sk[k]+=self.r_nk[n][k]*np.outer(t,t)
                sk[k] /=nk[k]
                
                
            self.update_alpha(nk)
            self.update_beta(nk)
            self.update_m(xk,nk)
            self.update_w(nk,xk,sk)
            self.update_nu(nk)
            
            if self.n_iters%20==0:
                rsp[self.n_iters] = self.r_nk
            
            # elbo-computation
            lb = self.elbo_computation(nk,sk,xk)
            lower_bound.append(lb)
            
            
            
            
            # Convergence criterion
            improve = (lb - lower_bound[self.n_iters-1]) if self.n_iters> 0 else lb
            if self.n_iters>0 and 0<improve<self.threshold:
                plt.plot(lower_bound)
                print('Converged at iteration {}'.format(self.n_iters))
                print('lower bound',lower_bound[-1])
                self.mixture_weights(nk)
                rsp[self.n_iters] = self.r_nk
                break
                
            if self.n_iters>self.max_iter:
                print('Maximum iteration reached, change hyperparameters')
                break
            self.n_iters+=1
        return rsp
     
    def cluster_prediction(self,rsp):
        d = [np.argmax(r) for r in rsp]
        return d
        
    
    
    def plot_results(self,rsp,data):
        
        color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
        
        fig,axes = plt.subplots(len(rsp),figsize = (10,20))
        iterations = list(rsp.keys())
        for i,r in enumerate(rsp.values()):
            for cluster,color in zip(range(self.k),color_iter):
                predictions = self.cluster_prediction(r)
                cluster_specific_points = [data[i] for i in range(len(data)) if predictions[i]==cluster]
                
                x,y = [i[0] for i in cluster_specific_points],[i[1] for i in cluster_specific_points]
                axes[i].scatter(x,y,c = color)
                
            axes[i].set_title("iteration {}".format(iterations[i]))   
                
        
            
            
            
            
            
            
        
     
        



