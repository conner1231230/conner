import random
import numpy as np
import matplotlib.pyplot as plt
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
    S_prime = S.copy()
    #state transition
    if A == 0 and S[0]-1 >=0:#up
        S_prime[0] = S[0] - 1
    if A == 1 and S[1]+1 < L:#right
        S_prime[1] = S[1] + 1
    if A == 2 and S[0]+1 < D: #down
        S_prime[0] = S[0] + 1
    if A == 3 and S[1]-1 >=0:#left
        S_prime[1] = S[1] - 1
    #reward setting
    if S_prime[0] == 1:
        R = -100
    else:
        R = -1
    return [S_prime, R]

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
avgtimes = 10
avgnsteplist = []
while avgtimes > 0:
    times = 300
    Q = tempQ.copy()
    avgRlist = []
    while times > 0:
        avgR = 0
        S = So.copy()
        S_queue = []
        R_queue = []
        A_queue = []
        A = eps_greedy(Q, S, eps)
        while S != Sg:
            S_queue.append(S)
            A_queue.append(A)
            [S, R] = env_SW(S, A, L, D)
            avgR += R
            A = eps_greedy(Q, S, eps)
            R_queue.append(R)
            if len(R_queue) == n_step:
                G_n = sum([x*y for x,y in zip(R_queue,garmma_list)]) + pow(mygarmma,n_step) * Q[S[0]][S[1]][A]
                OldEst = Q[S_queue[0][0]][S_queue[0][1]][A_queue[0]]
                Q[S_queue[0][0]][S_queue[0][1]][A_queue[0]] = OldEst + myalpha * (G_n - OldEst)
                S_queue.pop(0)
                R_queue.pop(0)
                A_queue.pop(0)
        avgRlist.append(avgR)

        #update the remaining state-actuin pairs(at most n-1 updates)
        tempStep = len(R_queue)
        G_n = 0
        for i in range(tempStep):
            G_n = R_queue[i] + G_n * mygarmma
            OldEst = Q[S_queue[i][0]][S_queue[i][1]][A_queue[i]]
            Q[S_queue[i][0]][S_queue[i][1]][A_queue[i]] = OldEst + myalpha * (G_n - OldEst)
        times -= 1
    avgnsteplist.append(avgRlist)
    avgtimes -= 1
    print(avgtimes)
ans = []
for i in range(len(avgnsteplist[0])):
    temp = 0
    for j in range(10):
        temp += avgnsteplist[j][i]
    ans.append(temp/10)

#sarsa

avgtimes = 10
avgsteplist = []
while avgtimes > 0:
    i=300
    steplist = []
    Q = tempQ.copy()
    while i>0:
        avgR = 0
        S = So.copy()
        A = eps_greedy(Q, S, eps)
        while S != Sg:
            [S_prime, R] = env_SW(S, A, L, D)
            avgR += R
            A_prime = eps_greedy(Q, S_prime, eps)
            Q[S[0]][S[1]][A] = Q[S[0]][S[1]][A] + myalpha*(R+ mygarmma* (Q[S_prime[0]][S_prime[1]][A_prime])- Q[S[0]][S[1]][A])
            S = S_prime
            A = A_prime
        steplist.append(avgR)
        i-=1
    avgsteplist.append(steplist)
    avgtimes -= 1
    print(avgtimes)
ans1 = []

for i in range(len(avgsteplist[0])):
    temp = 0
    for j in range(10):
        temp += avgsteplist[j][i]
    ans1.append(temp/10)

x = np.linspace(0,len(avgRlist)-1,len(avgRlist))
plt.plot(x,ans,label='n-step sarsa')
plt.plot(x,ans1,label='sarsa')
plt.legend()
plt.show()
