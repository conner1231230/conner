import random

from sklearn.preprocessing import scale
import numpy as np
from numpy import random

L = 10
D = 5
So = [1 , 0]
Sg = [1 , L-1]

#parameter setting
myalpha = 0.1
eps = 0.1
garmma = 1
#initialization
tempQ = np.random.randn(D, L, 4)
tempQ[Sg[0]][Sg[1]][:] = 0

#function
##epsilon greedy action selection
def eps_greedy(Q, S, eps):
    A = np.argmax(Q[S[0]][S[1]][:])
    if random.random()<eps:
        while True:
            temp = random.randint(0,3)
            if temp != A:
                A = temp
                break
    return A

#gridworld environment
def env_SW(S, A, L, D):
    #state transition
    if A == 0 and S[0]-1 >= 0:#up
        S[0] = S[0] - 1
    if A == 1 and S[1]+1 < L:#right
        S[1] = S[1] + 1
    if A == 2 and S[0]+1 < D:#down
        S[0] = S[0] + 1
    if A == 3 and S[1] - 1 >= 0:#left
        S[1] = S[1] - 1
    #reward setting
    R = random.normal(loc=-0.5,scale=2,size=(1))

    return [S, R]

"""
#Sarsa
Q = tempQ.copy()
i=300000
while i>0:
    S = So.copy()
    A = eps_greedy(Q, S, eps)
    while S != Sg:
        [S_prime, R] = env_SW(S, A, L, D)
        A_prime = eps_greedy(Q, S_prime, eps)
        Q[S[0]][S[1]][A] = Q[S[0]][S[1]][A] + myalpha*(R+ garmma* Q[S_prime[0]][S_prime[1]][A_prime]- Q[S[0]][S[1]][A])
        S = S_prime
        A = A_prime
    i-=1
np.set_printoptions(suppress=True)
print(Q)
"""
#Q-learning
ansQ=np.zeros((D,L,4))
times = 100
while times>0:
    Q = tempQ.copy()
    i=500
    while i>0:
        S = So.copy()
        A = eps_greedy(Q, S, eps)
        while S != Sg:
            [S_prime, R] = env_SW(S, A, L, D)
            A_prime = eps_greedy(Q, S_prime, eps)
            Q[S[0]][S[1]][A] = Q[S[0]][S[1]][A] + myalpha*(R+ garmma* np.max(Q[S_prime[0]][S_prime[1]][:])- Q[S[0]][S[1]][A])
            S = S_prime
            A = A_prime
        i-=1
    ansQ += Q
    times-=1
ansQ /= 100
np.set_printoptions(suppress=True)
print(ansQ)
