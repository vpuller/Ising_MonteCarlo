#!/usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import sys
import seaborn as sns
sns.set()

np.random.seed()

def energy(S_ij, J = 1.):
    return -J*(np.sum(S_ij[1:, :]*S_ij[:-1, :]) + np.sum(S_ij[:, 1:]*S_ij[:, :-1]))

def run_MC(Lx, Ly, T, Nit = 10**4, S_init = None, Nburn = 10**2):
    if S_init is None:
        #Initializing lattice
        S_init = np.ones((Lx, Ly))
        S_init[np.random.rand(Lx, Ly) > 0.5] = -1

    #MC looping
    S_old = np.copy(S_init)
    E_old = energy(S_old)
    M_old = np.sum(S_old)

    Esum = 0
    E2sum = 0
    Msum = 0
    Mabs_sum = 0
    M2sum = 0
    Ssum = np.zeros(S_init.shape)

    N = Nit*Lx*Ly
    iix = np.random.randint(0, high = Lx, size = N)
    iiy = np.random.randint(0, high = Ly, size = N)
    pp = np.random.rand(N)
    k = 0
    for n, (ix, iy, p) in enumerate(zip(iix, iiy, pp)): 
        S_new = np.copy(S_old)
        S_new[ix,iy] = - S_new[ix,iy]
        E_new = energy(S_new)

        if np.exp(-(E_new - E_old)/T) > p:
            if E_new > E_old:
                k +=1
            S_old = S_new
            E_old = E_new

        if n > Nburn:
            Esum += E_old
            E2sum += E_old**2
            M = np.sum(S_old)
            Msum += M
            M2sum += M**2
            Mabs_sum += np.abs(M)
            Ssum += S_old
    return Esum/N, E2sum/N, Msum/N, M2sum/N, Mabs_sum/N, Ssum/N, S_init, S_old

if __name__ == "__main__":
    '''
    Monte Carlo approach to Ising model
    '''

    #Setting parameters
    Lx = 8 #lattice dimensions
    Ly = 8
    N = Lx*Ly

    Tmin = 0.01 #temperature range
    Tmax = 10.
    nT = 51 #number of temperature points

    Nit = 10**4 #maximum number of iterations per site
    Nburn = 10**3 #burn-in iterations

    t0 = time()
    TT = np.linspace(Tmin, Tmax, nT)

    E_T = np.array([])
    E2_T = np.array([])
    M_T = np.array([])
    M2_T = np.array([])
    Mabs_T = np.array([])
    for jT, T in enumerate(TT):
        print(f'{jT}, T = {T}')
        t1 = time()
        E_mean, E2_mean, M_mean, M2_mean, Mabs_mean, _, _, _ = run_MC(Lx, Ly, T, Nit = Nit, Nburn = Nburn)
        t2 = time()
        print(f'simulation time: {t2-t1}\n')
        E_T = np.append(E_T, E_mean)
        E2_T = np.append(E2_T, E2_mean)
        M_T = np.append(M_T, M_mean)
        M2_T = np.append(M2_T, M2_mean)
        Mabs_T = np.append(Mabs_T, Mabs_mean)

    t3 = time()
    print(f'Total simulation time: {t3-t0}\n')
    fig, axs = plt.subplots(3, 2, figsize = (12, 8))
    axs[0, 0].plot(TT, E_T/N)
    axs[0, 0].set_ylabel(r'$\langle E\rangle$')
    axs[0, 1].plot(TT, (E2_T - E_T**2)/N)
    axs[0, 1].set_ylabel(r'$\langle E^2\rangle - \langle E\rangle^2$')
    axs[1, 0].plot(TT, M_T/N)
    axs[1, 0].set_ylabel(r'$\langle M\rangle$')
    axs[1, 1].plot(TT, (M2_T - M_T**2)/N)
    axs[1, 1].set_ylabel(r'$\langle M^2\rangle - \langle M\rangle^2$')
    axs[2, 0].plot(TT, Mabs_T/N)
    axs[2, 0].set_ylabel(r'$\langle |M|\rangle$')
    axs[2, 1].plot(TT, (M2_T - Mabs_T**2)/N)
    axs[2, 1].set_ylabel(r'$\langle M^2\rangle - \langle M\rangle^2$')
    for ax in axs:
        ax[0].set_xlabel('T')
        ax[1].set_xlabel('T')
    plt.tight_layout()
    plt.savefig('./figs/Temperature.jpg')
    plt.close()

