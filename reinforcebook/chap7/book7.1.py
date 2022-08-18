from audioop import avg
import random
import numpy as np
import matplotlib.pyplot as plt

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
#tempQ = np.zeros((D,L,4))
#Function

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
Q = tempQ.copy()

avgtime = 10
avgsteplist = []
while avgtime > 0:
    times = 80
    steplist =[]

    while times > 0:
        Model = np.zeros((D, L, 4, 3))
        Model_visit = np.zeros((D, L, 4))
        step = 0
        S_prime = So.copy()
        while S_prime != Sg:
            S = S_prime.copy()
            A = eps_greedy(Q, S, eps)
            [S_prime, R] = Ex7_1_env(S, A, L, D)
            #reward += R
            Q[S[0]][S[1]][A] =Q[S[0]][S[1]][A] + myalpha*(R+np.max(Q[S_prime[0]][S_prime[1]][:])-Q[S[0]][S[1]][A])
            Model_visit[S[0]][S[1]][A] = 1 #record visited pairs
            Model[S[0]][S[1]][A][0] = R
            Model[S[0]][S[1]][A][1] = S_prime[0]
            Model[S[0]][S[1]][A][2] = S_prime[1]
            step += R
            for k in range(n):
                #find visited state-action pairs
                #Ind_visit = find(Model_visit)
                #Ind_select = random.randint(0,len(Ind_visit)-1)
                while True:
                    tempi = random.randint(0,D-1)
                    tempj = random.randint(0,L-1)
                    tempp = random.randint(0,3)
                    if Model_visit[tempi][tempj][tempp] == 1:
                        break
                temp = Model[tempi][tempj][tempp]
                #temp = Model[Ind_visit[Ind_select][0]][Ind_visit[Ind_select][1]][Ind_visit[Ind_select][2]]
                R = temp[0]
                S_double_prime = [int(temp[1]), int(temp[2])]
                Q[tempi][tempj][tempp] =Q[tempi][tempj][tempp]+ myalpha*(R+np.max(Q[S_double_prime[0]][S_double_prime[1]][:])-Q[tempi][tempj][tempp])
        steplist.append(step)
        times -= 1
    avgsteplist.append(steplist)
    avgtime -= 1
    print(avgtime)
ans = []
for i in range(len(avgsteplist[0])):
    temp = 0
    for j in range(5):
        temp += avgsteplist[j][i]
    ans.append(temp/5)

x = np.linspace(0,len(ans)-1,len(ans))
plt.plot(x,ans,label='Normal weight')

plt.legend()
plt.show()
#np.set_printoptions(linewidth=190,threshold=sys.maxsize)
#print(Q)
