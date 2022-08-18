from audioop import avg
import random
import numpy as np
import matplotlib.pyplot as plt
L = 10
D = 5
So = [1 , 0]
Sg = [1 , L-1]

#parameter setting
myalpha = 0.1
eps = 0.3
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
    S_prime=S.copy()
    #state transition
    if A == 0 and S[0]-1 >= 0:#up
        S_prime[0] = S[0] - 1
    if A == 1 and S[1]+1 < L:#right
        S_prime[1] = S[1] + 1
    if A == 2 and S[0]+1 < D:#down
        S_prime[0] = S[0] + 1
    if A == 3 and S[1] - 1 >= 0:#left
        S_prime[1] = S[1] - 1
    #reward setting
    if S_prime[0] == 0:
        R=-100
    else:
        R=-1

    return [S_prime, R]

"""
#Sarsa
Q = np.zeros((D, L, 4))
avgtimes = 40
avgsteplist = []
while avgtimes > 0:
    i=300
    steplist = []
    while i>0:
        avgR = 0
        S = So.copy()
        A = eps_greedy(Q, S, eps)
        while S != Sg:
            [S_prime, R] = env_SW(S, A, L, D)
            avgR += R
            A_prime = eps_greedy(Q, S_prime, eps)
            Q[S[0]][S[1]][A] = Q[S[0]][S[1]][A] + myalpha*(R+ garmma* (Q[S_prime[0]][S_prime[1]][A_prime])- Q[S[0]][S[1]][A])
            S = S_prime
            A = A_prime
        steplist.append(avgR)
        i-=1
    avgsteplist.append(steplist)
    avgtimes -= 1
ans = []

for i in range(len(avgsteplist[0])):
    temp = 0
    for j in range(10):
        temp += avgsteplist[j][i]
    ans.append(temp/10)

"""
#Q-learning
Q = np.zeros((D, L, 4))
avgtimes = 40
avgsteplist = []
while avgtimes > 0:
    i=300
    steplist = []
    while i>0:
        avgR = 0
        S = So.copy()
        A = eps_greedy(Q, S, eps)
        while S != Sg:
            [S_prime, R] = env_SW(S, A, L, D)
            avgR += R
            A_prime = eps_greedy(Q, S_prime, eps)
            Q[S[0]][S[1]][A] = Q[S[0]][S[1]][A] + myalpha*(R+ garmma* np.max(Q[S_prime[0]][S_prime[1]][:])- Q[S[0]][S[1]][A])
            S = S_prime
            A = A_prime
        steplist.append(avgR)
        i-=1
    avgsteplist.append(steplist)
    avgtimes -= 1
ans = []

for i in range(len(avgsteplist[0])):
    temp = 0
    for j in range(10):
        temp += avgsteplist[j][i]
    ans.append(temp/10)

print(ans)
x = np.linspace(0,len(ans)-1,len(ans))
plt.plot(x,ans,label='Normal weight')

plt.legend()
plt.show()
