#!/usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    '''
    Monte Carlo approach to Ising model
    '''

    #Setting parameters
    Lx = 8
    Ly = 8
    N = Lx*Ly
    T = .001
    Nit = 10**4

    #Initializing lattice
    S_old = np.ones((Lx, Ly))
    S_old[np.random.rand(Lx, Ly) > 0.5] = -1
    E_old = energy(S_old)
    print(S_old)
    print(energy(S_old))


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

    EE = [E_old]
    MM = [np.sum(S_old)]
    SS = [S_old]
    for n in range(Nit):
        S_old, E_old = sites_loop(S_old, E_old, T = .1)
        EE.append(E_old)
        MM.append(np.sum(S_old))
        SS.append(S_old)

    print(S_old)
    print(energy(S_old))    
    EE = np.array(EE)
    MM = np.array(MM)

    fig, axs = plt.subplots(2, 1, figsize = (12, 6))
    axs[0].plot(range(Nit+1), EE/N)
    axs[0].set_ylabel('Energy/site')
    axs[1].plot(range(Nit+1), MM/N)
    axs[1].set_ylabel('Magnetization/site')
    for ax in axs:
        ax.set_xlabel('Iterations')
    plt.tight_layout()
    plt.savefig('./figs/evol.jpg')
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize = (6,6))
    im = ax.imshow(S_old.T, origin = 'lower', cmap = 'Greens')
    fig.colorbar(im, ax = ax)
    ax.set_title('Initial configuration')
    ax.set_xticks(range(Lx))
    # ax.set_xticklabels(self.labels_iso)
    ax.set_yticks(range(Ly))
    # ax.set_yticklabels(self.labels_frag)    
    plt.savefig('./figs/init.jpg')
    plt.close()
