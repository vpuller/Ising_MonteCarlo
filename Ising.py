#!/usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

import seaborn as sns
sns.set()

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

def run_MC(Lx, Ly, T, Nit = 10**4, S_init = None):
    if S_init is None:
        #Initializing lattice
        S_init = np.ones((Lx, Ly))
        S_init[np.random.rand(Lx, Ly) > 0.5] = -1

    #MC looping
    S_old = np.copy(S_init)
    E_old = energy(S_old)
    M_old = np.sum(S_old)

    EE = [E_old]
    MM = [M_old]
    SS = [S_old]
    for n in range(Nit*Lx*Ly):
        ix = np.random.randint(Lx)
        iy = np.random.randint(Ly)
        p = np.random.rand()

        S_new = np.copy(S_old)
        S_new[ix,iy] = - S_new[ix,iy]
        E_new = energy(S_new)

        if np.exp(-(E_new - E_old)) > p:
            S_old = S_new
            E_old = E_new

        EE.append(E_old)
        MM.append(np.sum(S_old))
        SS.append(S_old)

    return np.array(EE), np.array(MM), np.array(SS)

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

    # #Initializing lattice
    # S_old = np.ones((Lx, Ly))
    # S_old[np.random.rand(Lx, Ly) > 0.5] = -1
    # E_old = energy(S_old)
    # print(S_old)
    # print(energy(S_old))


    # #looping over sites
    # iix = np.random.randint(0, high = Lx, size = N)
    # iiy = np.random.randint(0, high = Ly, size = N)
    # pp = np.random.rand(N)
    # S_new = np.copy(S_old)
    # for ix, iy, p in zip(iix, iiy, pp):
    #     # S_new = np.copy(S_old)
    #     S_new[ix,iy] = - S_new[ix,iy]
    #     E_new = energy(S_new)
    #     print ("\n", E_old, E_new, (E_new - E_old)/T, np.exp(-(E_new - E_old)/T), p)
    #     if np.exp(-(E_new - E_old)/T) > p:
    #         S_old = S_new
    #         E_old = E_new
    #         print (np.exp(-(E_new - E_old)/T), p, E_old)

    # EE = [E_old]
    # MM = [np.sum(S_old)]
    # SS = [S_old]
    # for n in range(Nit):
    #     S_old, E_old = sites_loop(S_old, E_old, T = .1)
    #     EE.append(E_old)
    #     MM.append(np.sum(S_old))
    #     SS.append(S_old)

    # print(S_old)
    # print(energy(S_old))    
    # EE = np.array(EE)
    # MM = np.array(MM)

    # #One loop approach
    # for j in range(Nit):
    #     ix = np.random.randint(Lx)
    #     iy = np.random.randint(Ly)
    #     p = np.random.rand()

    #     S_new = np.copy(S_old)
    #     S_new[ix,iy] = - S_new[ix,iy]
    #     E_new = energy(S_new)

    #     if np.exp(-(E_new - E_old)) > p:
    #         S_old = S_new
    #         E_old = E_new

    #     EE.append(E_old)
    #     MM.append(np.sum(S_old))
    #     SS.append(S_old)

    # print(S_old)
    # print(energy(S_old), MM[-1])
    # EE = np.array(EE)
    # MM = np.array(MM)

    EE, MM, SS = run_MC(Lx, Ly, T, Nit = 10**4)

    NN = np.arange(len(EE))
    fig, axs = plt.subplots(2, 1, figsize = (12, 6))
    axs[0].plot(NN, EE/N)
    axs[0].set_ylabel('Energy/site')
    axs[1].plot(NN, MM/N)
    axs[1].set_ylabel('Magnetization/site')
    for ax in axs:
        ax.set_xlabel('Iterations')
    plt.tight_layout()
    plt.savefig('./figs/evol.jpg')
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize = (6,6))
    im = ax.imshow(SS[-1].T, origin = 'lower', cmap = 'Greens')
    fig.colorbar(im, ax = ax)
    ax.set_title('Initial configuration')
    ax.set_xticks(range(Lx))
    # ax.set_xticklabels(self.labels_iso)
    ax.set_yticks(range(Ly))
    # ax.set_yticklabels(self.labels_frag)    
    plt.savefig('./figs/init.jpg')
    plt.close()


    Tmin = 0.01
    Tmax = 10.
    TT = np.linspace(Tmin, Tmax, 11)
    Nburn = 100
    E_T = np.array([])
    E2_T = np.array([])
    M_T = np.array([])
    M2_T = np.array([])
    Mabs_T = np.array([])
    for jT, T in enumerate(TT):
        print(f'{jT}, T = {T}')
        t0 = time()
        EE, MM, SS = run_MC(Lx, Ly, T, Nit = 10**4)
        t1 = time()
        print(f'simulation time: {t1-t0}\n')
        E = EE[Nburn:]
        M = MM[Nburn:]
        E_T = np.append(E_T, E.mean())
        E2_T = np.append(E2_T, (E**2).mean())
        M_T = np.append(M_T, M.mean())
        M2_T = np.append(M2_T, (M**2).mean())
        Mabs_T = np.append(Mabs_T, np.abs(M).mean())

    fig, axs = plt.subplots(5, 1, figsize = (12, 30))
    axs[0].plot(TT, E_T/N)
    axs[0].set_ylabel('E')
    axs[1].plot(TT, (E2_T - E_T**2)/N)
    axs[1].set_ylabel('var(E)')
    axs[2].plot(TT, M_T/N)
    axs[2].set_ylabel('M')
    axs[3].plot(TT, Mabs_T/N)
    axs[3].set_ylabel('|M|')
    axs[4].plot(TT, (M2_T - M_T**2)/N)
    axs[4].set_ylabel('var(M)')
    for ax in axs:
        ax.set_xlabel('T')
    plt.tight_layout()
    plt.savefig('./figs/Temperature.jpg')
    plt.close()

