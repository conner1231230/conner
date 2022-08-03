import random
import numpy as np

#environment setting
L = 10
D = 5
Sg = [1, L-1]
So = [1, 0]
mygarmma = 1

#parameter
myalpha = 0.1
eps = 0.1
n_step = 5
garmma_list = np.zeros((n_step,1))
for i in range(n_step):
    garmma_list[i] = pow(mygarmma,i)


tempQ = np.random.randn(D, L, 4)
tempQ[Sg[0]][Sg[1]][:] = 0

#function
##gridworld environment
def env_SW(S, A, L, D):
    #state transition
    if A == 0 and S[0]-1 >=0:#up
        S[0] = S[0] - 1
    if A == 1 and S[1]+1 < L:#right
        S[1] = S[1] + 1
    if A == 2 and S[0]+1 < D: #down
        S[0] = S[0] + 1
    if A == 3 and S[0]-1 >=0:#left
        S[1] = S[1] - 1
    #reward setting
    if S[0] == 1:
        R = -100
    else:
        R = -1
    return [S, R]

def eps_greedy(Q, S, eps):
    A = np.argmax(Q[S[0]][S[1]][:])
    if random.random() < eps:
        while True:
            temp = random.randint(0,3)
            if temp != A:
                A = temp
                break
    return A

#n_step Sarsa
Q = tempQ
S = So
S_queue = []
R_queue = []
A_queue = []
A = eps_greedy(Q, S, eps)
while S != Sg:
    S_queue.append(S)
    A_queue.append(A)
    [S, R] = env_SW(S, A, L, D)
    A = eps_greedy(Q, S, eps)
    R_queue.append(R)
    if len(R_queue) == n_step:
        G_n = sum([x*y for x,y in zip(R_queue,garmma_list)]) + pow(mygarmma,n_step) * Q[S[0]][S[1]][A]
        OldEst = Q[S_queue[0][0]][S_queue[0][1]][A_queue[0]]
        Q[S_queue[0][0]][S_queue[0][1]][A_queue[0]] = OldEst + myalpha * (G_n - OldEst)
        S_queue.pop(0)
        R_queue.pop(0)
        A_queue.pop(0)

#update the remaining state-actuin pairs(at most n-1 updates)
tempStep = len(R_queue)
G_n = 0
for i in range(tempStep):
    G_n = R_queue[i] + G_n * mygarmma
    OldEst = Q[S_queue[i][0]][S_queue[i][1]][A_queue[i]]
    Q[S_queue[i][0]][S_queue[i][1]][A_queue[i]] = OldEst + myalpha * (G_n - OldEst)

print(Q)
