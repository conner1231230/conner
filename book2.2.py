import random
import numpy as np

L=15
D=5
Sg=[2,L]

theta=0.01
mygamma=1

V=np.zeros((D,L))

S_prime=np.zeros((4,2))
R=np.zeros((4,1))

def env_SW(S,A,L,D):
    if A==1 and S[0]-1>=0:
        S[0]=S[0]-1
    if A==2 and S[1]+1<L:
        S[1]=S[1]+1
    if A==3 and S[0]+1<D:
        S[0]=S[0]+1
    if A==4 and S[1]-1>=0:
        S[1]=S[1]-1
    if S[0]==0:
        R=-100
    else:
        R=-1
    return S,R

delta=theta+1
while delta>theta:
    delta=0
    for i in range(D):
        for j in range(L):
            if i==1 and j==14:
                break
            else:
                v=V[i,j]
                for k in range(4):
                    S_prime[k],R[k] = env_SW([i,j],k,L,D)
                    R[k]=R[k]+mygamma*V[int(S_prime[k,0]),int(S_prime[k,1])]
                V[i,j]=max(R)
                delta=max(delta,abs(v-V[i,j]))
V=V.astype(int)
print(V)




