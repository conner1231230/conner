import random
import numpy as np
import matplotlib.pyplot as plt

times=1000
L = 7
D = 5
Sg = [1,0]
So = [3,0]

fig = plt.figure() #定義一個圖像窗口
x = np.linspace(0,times,times)
y=[]
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
    #reward setting
    if S[0] == 1 and S[1] == L-1:
        R=40
        return [Sg,R]
    else:
        R=0


    #state transition
    if A == 1 and S[0]-1 >= 0:#up
        S[0] = S[0]-1
    elif A == 2 and S[1]+1 < L:#right
        S[1] = S[1]+1
    elif A == 3 and S[0]+1 < D:#down
        S[0] = S[0]+1
    elif A == 4 and S[1]-1 >= 0:#left
        S[1] = S[1]-1
    return [S,R]

#Monte Carlo Control
Q=tempQ

while(times>0):
    S=So
    S_queue=np.array([0,0])
    R_queue=np.array([])
    A_queue=np.array([])
    step=0
    while S!=Sg:
        A=eps_greedy(Q,S,eps)
        S_queue = np.vstack((S_queue,S))
        A_queue = np.append(A_queue,A)
        [S,R] = env_SW(S,A,L,D)
        R_queue = np.append(R_queue,R)
        step+=1
    y.append(step)
    T=R_queue.size-1
    G=0

    for t in range(T,-1,-1):
        G=R_queue[t]+mygamma*G
        Q[[int(S_queue[t+1,0])],[int(S_queue[t+1,1])],[int(A_queue[t])]]+=myalpha*(G-Q[[int(S_queue[t+1,0])],[int(S_queue[t+1,1])],[int(A_queue[t])]])
    times-=1

plt.plot(x,y,'.')
plt.show()
print(Q)








