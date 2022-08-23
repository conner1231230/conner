
import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt
import random
#environment setting
num_epi = 300


#parameter setting
myalpha_w = 0.1
myalpha_theta = 0.05
del_t = 0.05
mygarmma = 1
sig = 2

#initialization
N = 8 #num of tilings
p = [10, 15]
numTilesTiling = p[0] * p[1]

#function
def env_Ex9_1(S, u, del_t):
    S_copy = S.copy()
    g = 9.8
    l = 4

    F = np.array([[0, 1], [g/l, 0]])
    H = np.array([[0],[-1/l]])
    temp = integralexpm(F,0,del_t)
    temp2 = np.matmul(temp, H)
    S_copy = np.matmul(sp.expm(F*del_t), S_copy[0]) + temp2.T * u

    if S_copy[0][0] >= np.pi/2 or S_copy[0][0] <=-np.pi/2:
        R = 0
    else:
        R = 1

    return [S_copy, R]

def integralexpm(F,down,up):
    detF = F[0][0]*F[1][1]-F[0][1]*F[1][0]
    adjF = np.array([[F[1][1],-F[1][0]],[-F[0][1],F[0][0]]])
    reverseF = adjF/detF
    temp1 = sp.expm(F*up) - sp.expm(F*down)

    return np.matmul(reverseF,temp1)

def featurState(S, N, p):
#N is the number  of tilings
#p is a vector consisting of p_i and p_i is the number of partitions in the i-th dim
    S_copy1 = S.copy()
    S_copy1 = state_normal_Ex9_1(S_copy1)
    sz = np.array((N,p),dtype=object)
    tiles = np.zeros((N * numTilesTiling))
    L = p[0]/(p[0]-1+(1/N))
    xi = (L-1)/(N-1)
    blocklength = L/p[0]
    for n in range(N):
        tempS = S_copy1 + n *xi
        mysub = np.floor(tempS/blocklength)
        myind = sub2ind(sz, n, mysub[0][0], mysub[0][1])
        tiles[int(myind)] = 1

    return tiles

def sub2ind(array_shape, n, rows, cols):
    return (rows*array_shape[0]+n)*array_shape[1][1] + cols

def state_normal_Ex9_1(S_copy1):
    S_copy1[0][0] = (S_copy1[0][0] + np.pi/2)/np.pi
    S_copy1[0][1] = (S_copy1[0][1] + 2*np.pi)/(4*np.pi)

    return S_copy1
#Algorithm Actor-Critic methods for episodic tasks
times = 10
avgRlist = []
for time in range(times):
    w = np.zeros(N*numTilesTiling)
    Rlist = []
    theta_mu = np.zeros(N*numTilesTiling)
    for epi in range(num_epi):
        G = 0
        StepCount = 0
        S = np.array([[0.6, -1.3]])
        R = 1
        while R > 0:
            feaVec_S = featurState(S, N, p)
            feaVec_S_copy = feaVec_S.copy()
            mu = np.dot(theta_mu, feaVec_S_copy)
            A = random.gauss(mu, sig)
            [S_prime, R] = env_Ex9_1(S, A, del_t)
            OldEst = np.dot(w, feaVec_S_copy)
            if R == 0:
                delta = -OldEst
            else:
                delta = R + mygarmma * np.dot(w,featurState(S_prime, N, p)) - OldEst
            w += myalpha_w * delta * feaVec_S
            grad_mu = (A-mu)/(sig*sig)*feaVec_S
            theta_mu += myalpha_theta * delta * grad_mu
            S = S_prime
            G += R
            if G > 100:
                print("G:",G,", S:",S)
        Rlist.append(G)
    avgRlist.append(Rlist)
    print(time)
print(avgRlist)
ans = []
for i in range(len(avgRlist[0])):
    temp = 0
    for j in range(times):
        temp += avgRlist[j][i]
    ans.append(temp/times)

x = np.linspace(0,len(ans)-1,len(ans))
plt.plot(x,ans,label='Actor-Critic Method')

plt.legend()
plt.show()


