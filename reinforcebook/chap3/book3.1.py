import random
import numpy as np
import matplotlib.pyplot as plt
L = 4
D = 3
Sg = [1,L-1]
So = [1,1]

#parameter setting
myalpha = 0.1
eps = 0.1
mygamma = 0.1

#initialization
tempQ = np.zeros((D,L,4))

#Function
def eps_greedy(Q,S,eps):
    if random.random() < eps:
        A = random.randint(0,3)
    else:
        A = Q[S[0]][S[1]][:].argmax()

    return A

#Gridworld Environment
def env_SW(S,A,L,D):
    #state transition
    S_prime = S.copy()
    if A == 1 and S[0]-1 >= 0:#up
        S_prime[0] = S[0]-1
    elif A == 2 and S[1]+1 < L:#right
        S_prime[1] = S[1]+1
    elif A == 3 and S[0]+1 < D:#down
        S_prime[0] = S[0]+1
    elif A == 4 and S[1]-1 >= 0:#left
        S_prime[1] = S[1]-1
    #reward setting
    if S_prime[0] == 1:
        R=-100
    else:
        R=-1
    return [S_prime,R]

#Monte Carlo Control
avgsteplist = []
avgtimes=40
while avgtimes > 0:
    Q=tempQ.copy()
    times=150
    avgRlist = []
    while(times>0):
        avgR = 0
        S=So.copy()
        S_queue=np.array([0,0])
        R_queue=np.array([])
        A_queue=np.array([])

        while S!=Sg:
            A=eps_greedy(Q,S,eps)
            S_queue = np.vstack((S_queue,S))
            A_queue = np.append(A_queue,A)
            [S,R] = env_SW(S,A,L,D)
            R_queue = np.append(R_queue,R)
            avgR += R

        T=R_queue.size-1
        G=0

        for t in range(T,-1,-1):
            G=R_queue[t]+mygamma*G
            Q[[int(S_queue[t+1,0])],[int(S_queue[t+1,1])],[int(A_queue[t])]]+=myalpha*(G-Q[[int(S_queue[t+1,0])],[int(S_queue[t+1,1])],[int(A_queue[t])]])
        avgRlist.append(avgR)
        times-=1
    avgtimes -= 1
    print(avgtimes)
    avgsteplist.append(avgRlist)

ans = []
for i in range(len(avgsteplist[0])):
    temp = 0
    for j in range(40):
        temp += avgsteplist[j][i]
    ans.append(temp/40)

x = np.linspace(0,len(avgRlist)-1,len(avgRlist))
plt.plot(x,avgRlist,label='Monte Carlo Control')

plt.legend()
plt.show()

