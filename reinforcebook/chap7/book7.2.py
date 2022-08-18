
import random
import numpy as np
import matplotlib.pyplot as plt

L = 5
D = 9
So = [3, 0]
Sg = [0, L-1]

#parameter setting
myalpha = 0.1
eps = 0.3
n = 4
kappa = 0.1

#initialization
tempQ = np.random.randn(D, L, 4)
tempQ[Sg[0]][Sg[1]][:] = 0
Model_plus = np.zeros((D, L, 4, 3))
Model_visit_plus =np.zeros((D, L))
tau = np.zeros((D, L,4))

#Function
def Ex7_2_env(S, A, L, D, step):
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
    #nonstationary environment -> block detection
    if step <= 200:
        if (temp[1] == 2 and temp[0] <= 7) or (temp[0] == 7 and temp[1] == 3):
            temp = Sc
    else:
        if (temp[1] == 2 and temp[0] <= 7 and temp[0] > 0) or (temp[0] == 7 and temp[1] == 3):
            temp = Sc
    #reward setting
    R = -10
    return [temp, R]

def eps_greedy(Q, S, eps):
    A = np.argmax(Q[S[0]][S[1]][:])
    if random.random() < eps:
        A = random.randint(0,3)

    return A

#Dyna-Q+
Q = tempQ.copy()
avgsteplist = []
avgtime = 10
while avgtime > 0:
    avgRlist = []
    times = 1000
    while times>0:
        avgR = 0
        S_prime = So.copy()
        step = 0
        while S_prime!=Sg:
            S = S_prime
            A = eps_greedy(Q, S, eps)
            [S_prime, R] = Ex7_2_env(S, A, L, D, step)
            avgR += R
            Q[S[0]][S[1]][A] += myalpha*(R+np.max(Q[S_prime[0]][S_prime[1]][:]-Q[S[0]][S[1]][A]))
            tau += 1
            tau[S[0]][S[1]][A] = 0
            Model_visit_plus[S[0]][S[1]] = 1
            Model_plus[S[0]][S[1]][A][0] = R
            Model_plus[S[0]][S[1]][A][1] = S_prime[0]
            Model_plus[S[0]][S[1]][A][2] = S_prime[1]
            for k in range(n):
                while True:
                    tempi = random.randint(0,D-1)
                    tempj = random.randint(0,L-1)
                    if Model_visit_plus[tempi][tempj] == 1:
                        break
                tempA = random.randint(0,3)
                temp = Model_plus[tempi][tempj][tempA]
                S_double_prime = [0,0]
                if temp[0] == 0:
                    tempR = 0
                    S_double_prime = [tempi, tempj]
                else:
                    tempR = temp[0]
                    S_double_prime =[int(temp[1]), int(temp[2])]
                tempR += kappa*np.power(tau[tempi][tempj][tempA],0.5)
                Q[tempi][tempj][tempA] += myalpha*(tempR+np.max(Q[S_double_prime[0]][S_double_prime[1]][:])-Q[tempi][tempj][tempA])
            step += 1
        avgRlist.append(avgR)
        times -= 1
    avgsteplist.append(avgRlist)
    avgtime -= 1
    print(avgtime)
ans = []
for i in range(len(avgsteplist[0])):
    temp = 0
    for j in range(5):
        temp += avgsteplist[j][i]
    ans.append(temp/5)


#Dyna-Q
Q = tempQ.copy()

avgtime = 10
avgsteplist = []
while avgtime > 0:
    times = 1000
    steplist =[]

    while times > 0:
        Model = np.zeros((D, L, 4, 3))
        Model_visit = np.zeros((D, L, 4))
        step = 0
        avgR = 0
        S_prime = So.copy()
        while S_prime != Sg:
            S = S_prime.copy()
            A = eps_greedy(Q, S, eps)
            [S_prime, R] = Ex7_2_env(S, A, L, D,step)
            avgR += R
            #reward += R
            Q[S[0]][S[1]][A] =Q[S[0]][S[1]][A] + myalpha*(R+np.max(Q[S_prime[0]][S_prime[1]][:])-Q[S[0]][S[1]][A])
            Model_visit[S[0]][S[1]][A] = 1 #record visited pairs
            Model[S[0]][S[1]][A][0] = R
            Model[S[0]][S[1]][A][1] = S_prime[0]
            Model[S[0]][S[1]][A][2] = S_prime[1]
            step += 1
            for k in range(n):
                while True:
                    tempi = random.randint(0,D-1)
                    tempj = random.randint(0,L-1)
                    tempp = random.randint(0,3)
                    if Model_visit[tempi][tempj][tempp] == 1:
                        break
                temp = Model[tempi][tempj][tempp]
                R = temp[0]
                S_double_prime = [int(temp[1]), int(temp[2])]
                Q[tempi][tempj][tempp] =Q[tempi][tempj][tempp]+ myalpha*(R+np.max(Q[S_double_prime[0]][S_double_prime[1]][:])-Q[tempi][tempj][tempp])
        steplist.append(avgR)
        times -= 1
    avgsteplist.append(steplist)
    avgtime -= 1
    print(avgtime)

ans1 = []
for i in range(len(avgsteplist[0])):
    temp = 0
    for j in range(5):
        temp += avgsteplist[j][i]
    ans1.append(temp/5)




x = np.linspace(0,len(ans)-1,len(ans))
plt.plot(x,ans,label='Dyna-Q+')
plt.plot(x,ans1,label='Dyna-Q')
plt.legend()
plt.show()