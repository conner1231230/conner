import numpy as np
import random
import matplotlib.pyplot as plt

#environment setting
D = 5
L = 10
So = [1, 0]
Sg = [1, L-1]


#parameter setting
myalpha = 0.1
eps = 0.3
lamda = 0.5
garmma = 1

#initialization
tempQ = np.random.randn(D, L, 4)
tempQ[Sg[0]][Sg[1]][:] = 0

def eps_greedy(Q, S, eps):
    A = np.argmax(Q[S[0]][S[1]][:])
    if random.random() < eps:
        A = random.randint(0,3)

    return A

def Ex7_1_env(S, A, L, D):
    Sc = S.copy()
    temp =S.copy()
    #state transition
    if A == 0 and S[0]-1 >= 0: #up
        temp[0] = S[0] - 1
    if A == 1 and S[1]+1 < L: #right
        temp[1] = S[1] + 1
    if A == 2 and S[0]+1 < D: #down
        temp[0] = S[0] + 1
    if A == 3 and S[1]-1 >= 0: #left
        temp[1] = S[1] - 1
    #block detection
    if temp[0] >= 1 and temp[0] <= 3 and temp[1] == 3:
        temp = Sc
    if temp[0] >= 2 and temp[0] <= 4 and temp[1] == 6:
        temp = Sc
    #reward setting
    if temp[0] == 0:
        R = -100
    else:
        R = -1

    return [temp, R]

#Sarsa(lambda)
avgtimes = 40
avgRlist = []
while avgtimes > 0:
    Rlist = []
    Q = tempQ.copy()
    times = 200
    while times > 0:
        avgR = 0
        S = So
        z = np.zeros((D, L, 4))
        A = eps_greedy(Q, S, eps)
        while S != Sg:
            [S_prime, R] =Ex7_1_env(S, A, L, D)
            avgR += R
            A_prime = eps_greedy(Q, S_prime, eps)
            delta = R + Q[S_prime[0]][S_prime[1]][A_prime] - Q[S[0]][S[1]][A]
            z[S[0]][S[1]][A] += 1
            Q += myalpha * delta * z
            z = z * lamda
            S = S_prime
            A = A_prime

        Rlist.append(avgR)
        times -= 1
    avgRlist.append(Rlist)
    avgtimes -= 1
    print(avgtimes)

ans = []
for i in range(len(avgRlist[0])):
    temp = 0
    for j in range(10):
        temp += avgRlist[j][i]
    ans.append(temp/10)

#Sarsa
avgtimes = 40
avgRlist = []
while avgtimes > 0:
    Rlist = []
    Q = tempQ.copy()
    times = 200
    while times > 0:
        avgR = 0
        S = So
        A = eps_greedy(Q, S, eps)
        while S != Sg:
            [S_prime, R] = Ex7_1_env(S, A, L, D)
            avgR += R
            A_prime = eps_greedy(Q, S_prime, eps)
            Q[S[0]][S[1]][A] += myalpha*(R + garmma*Q[S_prime[0]][S_prime[1]][A_prime] - Q[S[0]][S[1]][A])
            S = S_prime
            A = A_prime

        Rlist.append(avgR)
        times -= 1
    avgRlist.append(Rlist)
    avgtimes -= 1
    print(avgtimes)

ans1 = []
for i in range(len(avgRlist[0])):
    temp = 0
    for j in range(10):
        temp += avgRlist[j][i]
    ans1.append(temp/10)




x = np.linspace(0,len(ans)-1,len(ans))
plt.plot(x,ans,label='Sarsa(lambda)')
plt.plot(x,ans1,label='Sarsa')
plt.legend()
plt.show()
