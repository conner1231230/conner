
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=190,threshold=sys.maxsize)

Sg = np.array([0.5, 0])
num_epi = 200

#parameter setting
myalpha = 0.06
eps = 0.1

#initialization
N = 8
p = np.array([10, 10])
K = 3
w = np.random.randn(N*p[0]*p[1]*K)

#Fucntion
def EX6_1_eps_greedy(w, S, eps, numTilings, numPartition, num_action):
    if random.random() < eps:
        A=random.randint(0,2)
    else:
        feaVec1 = featureVEC(S, -1, numTilings, numPartition, num_action)
        feaVec2 = featureVEC(S, 0, numTilings, numPartition, num_action)
        feaVec3 = featureVEC(S, 1, numTilings, numPartition, num_action)
        templist = [w.dot(feaVec1),w.dot(feaVec2),w.dot(feaVec3)]
        maxnumber = max(templist)
        A = templist.index(maxnumber)
    A -= 1
    return A

def featureVEC(S, A, N, p, num_action):

    S1 = Ex6_1_state_normal(S)
    numTilesInTiling = p[0] * p[1]
    tiles = np.zeros(N * numTilesInTiling * num_action)
    L = p[0]/(p[0]-1+(1/N))
    xi = (L-1)/(N-1)
    blocklength = L/p[0]
    #print("1")
    #print(L)
    #print(xi)
    #print(blocklength)
    for n in range(N):
        tempS = S1 + n *xi
        mysub = np.ceil(tempS/blocklength)
        mysub -= 1
        #print("||")
        #print(mysub)
        sz = np.array((N,p,num_action),dtype=object)
        myind = sub2ind(sz, n, mysub[0], mysub[1], A)
        #myind.astype(int)
        #print(sz[1])
        #print(sz[2])
        tiles[int(myind)] = 1
    return tiles

def sub2ind(array_shape, n, rows, cols, A):
   # print(n, rows, cols, A)
    return (rows*array_shape[0]+n)*array_shape[1][1]*array_shape[2] + ((A+1)*array_shape[1][1]+cols)
    #return n*array_shape[0] + (rows+1)*array_shape[1][0] + (cols+1)*array_shape[1][1] + (A+2)#*array_shape[2]

def Ex6_1_state_normal(S):
    temp = np.array([S[0],S[1]])
    temp[0] = (temp[0]+1.2)/1.7
    temp[1] = (S[1]+0.07)/0.14
    return temp

def env_Ex6_1(S, A):
    x_dot = S[1] + 0.001*A - 0.0025*np.cos(3*S[0])
    x_dot = max(min(x_dot,0.07),-0.07)
    x = S[0] + x_dot
    x = max(min(x,0.5),-1.2)
    if x == -1.2 or x ==0.5:
        x_dot = 0
    S = np.array([x, x_dot])
    #reward setting
    if x == 0.5:
        R = 0
    else:
        R = -1

    return [S, R]

#Episodic Semi-gradient Sarsa
times = 200
steplist = []
avgRlist = []
while times>0:
    step = 0
    avgR = 0
    S = np.array([random.random()*0.2-0.7, 0])
    A = EX6_1_eps_greedy(w, S, eps, N, p, K)
    while S[0] != Sg[0]:
        [S_prime, R] = env_Ex6_1(S, A)
        avgR += R
        feaVec_S = featureVEC(S, A, N, p, K)
        OldEst = w.dot(feaVec_S)
        if S[0]==Sg[0]:
            w = w + myalpha*(R-OldEst)*feaVec_S
            break
        else:
            A_prime = EX6_1_eps_greedy(w, S_prime, eps, N, p, K)
            feaVec_S_prime = featureVEC(S_prime, A_prime, N, p, K)
            w = w + myalpha*(R+w.dot(feaVec_S_prime)-OldEst)*feaVec_S
            step += 1
        S = S_prime
        A = A_prime
    avgRlist.append(avgR)
    steplist.append(step)
    times -= 1
    print(times)
ans = steplist
ans1 = avgRlist
#print(w)
print(steplist)
x = np.linspace(0,len(ans)-1,len(ans))
plt.plot(x,ans,label='Step')
plt.plot(x,ans1,label='Reward')
plt.legend()
plt.show()
