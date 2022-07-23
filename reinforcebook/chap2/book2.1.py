import random
import numpy as np

L=7
D=5
Sg=[2,L]

theta=0.01
mygamma=0.9

#v1=random.randint(0,D)
#v2=random.randint(0,L)
V=np.zeros((D,L))

S_prime=np.zeros((4,2))
R=np.zeros((4,1))

def Ex2_1_env(S,A,L,D):
    if S[0]==1 and S[1]==L-1:
        R=40
        S=[1,0]
    else:
        if A==1 and S[0]-1>=0:
            S[0]=S[0]-1
        if A==2 and S[1]+1<L:
            S[1]=S[1]+1
        if A==3 and S[0]+1<D:
            S[0]=S[0]+1
        if A==4 and S[1]-1>=0:
            S[1]=S[1]-1
        R=0
    return S,R

delta=theta+1
while delta>theta:
    delta=0
    for i in range(D):
        for j in range(L):
            v=V[i,j]
            for k in range(4):
                S_prime[k],R[k] = Ex2_1_env([i,j],k,L,D)
                R[k]=R[k]+mygamma*V[int(S_prime[k,0]),int(S_prime[k,1])]
            V[i,j]=sum(R)/4
            delta=max(delta,abs(v-V[i,j]))

np.set_printoptions(linewidth=1000)
print(V)

