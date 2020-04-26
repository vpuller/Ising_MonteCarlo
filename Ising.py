#!/usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import sys

# import seaborn as sns
# sns.set()

np.random.seed()

def energy(S_ij, J = 1.):
    return -J*(np.sum(S_ij[1:, :]*S_ij[:-1, :]) + np.sum(S_ij[:, 1:]*S_ij[:, :-1]))

def sites_loop(S_old, E_old, T):
    #looping over sites
    iix = np.random.randint(0, high = Lx, size = N)
    iiy = np.random.randint(0, high = Ly, size = N)
    pp = np.random.rand(N)
    S_new = np.copy(S_old)
    for ix, iy, p in zip(iix, iiy, pp):
        # S_new = np.copy(S_old)
        S_new[ix,iy] = - S_new[ix,iy]
        E_new = energy(S_new)
        # print ("\n", E_old, E_new, (E_new - E_old)/T, np.exp(-(E_new - E_old)/T), p)
        if np.exp(-(E_new - E_old)/T) > p:
            S_old = S_new
            E_old = E_new
            # print (np.exp(-(E_new - E_old)/T), p, E_old)
    return S_old, E_old

def run_MC(Lx, Ly, T, Nit = 10**4, S_init = None, Nburn = 10**2):
    if S_init is None:
        #Initializing lattice
        S_init = np.ones((Lx, Ly))
        S_init[np.random.rand(Lx, Ly) > 0.5] = -1

    #MC looping
    S_old = np.copy(S_init)
    E_old = energy(S_old)
    M_old = np.sum(S_old)

    # EE = [E_old]
    # MM = [M_old]
    # SS = [S_old]
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
    # for n in range(Nit*Lx*Ly):
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
            # EE.append(E_old)
            # MM.append(np.sum(S_old))
            # SS.append(S_old)
    print(k, N, T)
    # return np.array(EE), np.array(MM), np.array(SS)
    return Esum/N, E2sum/N, Msum/N, M2sum/N, Mabs_sum/N, Ssum/N, S_init, S_old

if __name__ == "__main__":
    '''
    Monte Carlo approach to Ising model
    '''

    #Setting parameters
    Lx = 8
    Ly = 8
    N = Lx*Ly
    T = 1.
    Nit = Lx*Ly*10**2

    # t0 = time()
    # # EE, MM, SS = run_MC(Lx, Ly, T, Nit = 10**5)
    # E_mean, E2_mean, M_mean, M2_mean, Mabs_mean, S_mean, S_init, S_final = run_MC(Lx, Ly, T, Nit = 10**2, Nburn = 0)
    # t1 = time()
    # print(f'simulation time: {t1 - t0}')

    # # NN = np.arange(len(EE))
    # # fig, axs = plt.subplots(2, 1, figsize = (12, 6))
    # # axs[0].plot(NN, EE/N)
    # # axs[0].set_ylabel('Energy/site')
    # # axs[1].plot(NN, MM/N)
    # # axs[1].set_ylabel('Magnetization/site')
    # # for ax in axs:
    # #     ax.set_xlabel('Iterations')
    # # plt.tight_layout()
    # # plt.savefig('./figs/evol.jpg')
    # # plt.close()

    # fig, axs = plt.subplots(1, 3, figsize = (12, 6))
    # im0 = axs[0].imshow(S_init.T, origin = 'lower', cmap = 'bwr', vmin = -1., vmax = 1.)
    # im1 = axs[1].imshow(S_final.T, origin = 'lower', cmap = 'bwr', vmin = -1., vmax = 1.)
    # im2 = axs[2].imshow(S_mean.T, origin = 'lower', cmap = 'bwr', vmin = -1., vmax = 1.)
    # # fig.colorbar(im1, ax = axs[0])
    # axs[0].set_title('Initial configuration')
    # axs[1].set_title('Final configuration')
    # for ax in axs:
    #     ax.set_xticks(range(Lx))
    #     # ax.set_xticklabels(self.labels_iso)
    #     ax.set_yticks(range(Ly))
    #     # ax.set_yticklabels(self.labels_frag)    
    # plt.savefig('./figs/init.jpg')
    # plt.close()

    # sys.exit()

    t0 = time()
    Tmin = 0.01
    Tmax = 10.
    TT = np.linspace(Tmin, Tmax, 51)
    # TT = np.array([.01, .1, 1., 10., 100., 1000., 10000.])
    Nburn = 100
    E_T = np.array([])
    E2_T = np.array([])
    M_T = np.array([])
    M2_T = np.array([])
    Mabs_T = np.array([])
    for jT, T in enumerate(TT):
        print(f'{jT}, T = {T}')
        t1 = time()
        # EE, MM, SS = run_MC(Lx, Ly, T, Nit = 10**4)
        E_mean, E2_mean, M_mean, M2_mean, Mabs_mean, _, _, _ = run_MC(Lx, Ly, T, Nit = 10**4, Nburn = 10**3)
        t2 = time()
        print(f'simulation time: {t2-t1}\n')
        E_T = np.append(E_T, E_mean)
        E2_T = np.append(E2_T, E2_mean)
        M_T = np.append(M_T, M_mean)
        M2_T = np.append(M2_T, M2_mean)
        Mabs_T = np.append(Mabs_T, Mabs_mean)

    t3 = time()
    print(f'simulation time: {t3-t0}\n')
    fig, axs = plt.subplots(3, 2, figsize = (18, 12))
    axs[0, 0].plot(TT, E_T/N)
    axs[0, 0].set_ylabel('E')
    axs[0, 1].plot(TT, (E2_T - E_T**2)/N)
    axs[0, 1].set_ylabel('var(E)')
    axs[1, 0].plot(TT, M_T/N)
    axs[1, 0].set_ylabel('M')
    axs[1, 1].plot(TT, (M2_T - M_T**2)/N)
    axs[1, 1].set_ylabel('var(M)')
    axs[2, 0].plot(TT, Mabs_T/N)
    axs[2, 0].set_ylabel('|M|')
    axs[2, 1].plot(TT, (M2_T - Mabs_T**2)/N)
    axs[2, 1].set_ylabel('var(|M|)')
    for ax in axs:
        ax[0].set_xlabel('T')
        ax[1].set_xlabel('T')
    plt.tight_layout()
    plt.savefig('./figs/Temperature.jpg')
    plt.close()

