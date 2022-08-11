import random
import numpy as np
import sys

L = 10
D = 5
So = [1, 0]
Sg = [1, L-1]

#parameter setting
myalpha = 0.1
eps = 0.3
n = 4

#initialization
tempQ = np.random.randn(D, L, 4)
tempQ[Sg[0]][Sg[1]][:] = 0

#Function

def Ex7_1_env(S, A, L, D):
    Sc = S
    #state transition
    if A == 0 and S[0]-1 >= 0: #up
        S[0] = S[0] - 1
    if A == 1 and S[1]+1 < L: #right
        S[1] = S[1] + 1
    if A == 2 and S[0]+1 < D: #down
        S[0] = S[0] + 1
    if A == 3 and S[1]-1 >= 0: #left
        S[1] = S[1] - 1
    #block detection
    if S[0] >= 1 and S[0] <= 3 and S[1] == 3:
        S = Sc
    if S[0] >= 2 and S[0] <= 5 and S[1] == 7:
        S = Sc
    #reward setting
    if S[0] == 0:
        R = -100
    else:
        R = -1

    return [S, R]

def eps_greedy(Q, S, eps):
    A = np.argmax(Q[S[0]][S[1]][:])
    if random.random() < eps:
        A = random.randint(0,3)

    return A

def find(Model_visit):
    templist=[]
    for i in range(D):
        for j in range(L):
            for p in range(4):
                if Model_visit[i][j][p] == 1:
                    templist.append([i, j ,p])

    return templist

#Dyna-Q
times = 100
Q = tempQ.copy()
Model = np.zeros((D, L, 4, 3))
Model_visit = np.zeros((D, L, 4))
steplist =[]
while times > 0:
    step = 0
    S_prime = So.copy()
    while S_prime != Sg:
        S = S_prime.copy()
        A = eps_greedy(Q, S, eps)
        [S_prime, R] = Ex7_1_env(S, A, L, D)
        #reward += R
        Q[S[0]][S[1]][A] =Q[S[0]][S[1]][A] + myalpha*(R+max(Q[S_prime[0]][S_prime[1]][:])-Q[S[0]][S[1]][A])
        Model_visit[S[0]][S[1]][A] = 1 #record visited pairs
        Model[S[0]][S[1]][A][0] = R
        Model[S[0]][S[1]][A][1] = S_prime[0]
        Model[S[0]][S[1]][A][2] = S_prime[1]
        for k in range(n):
            #find visited state-action pairs
            Ind_visit = find(Model_visit)
            Ind_select = random.randint(0,len(Ind_visit)-1)
            temp = Model[Ind_visit[Ind_select][0]][Ind_visit[Ind_select][1]][Ind_visit[Ind_select][2]]
            R = temp[0]
            S_double_prime = [int(temp[1]), int(temp[2])]
            Q[S[0]][S[1]][A] =Q[S[0]][S[1]][A]+ myalpha*(R+max(Q[S_double_prime[0]][S_double_prime[1]][:])-Q[S[0]][S[1]][A])
        step += 1
    steplist.append(step)
    times -= 1
print(steplist)
np.set_printoptions(linewidth=190,threshold=sys.maxsize)
print(Q)
