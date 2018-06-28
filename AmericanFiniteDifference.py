import numpy as np

t = 5/12
S0 = 50
K = 50
r = 0.1
sig = 0.4

M = 20
N = 10
dS = 5
def finiteDifference(S0,K,r,sig,t,M,N,dS):
	dt = t/N

	Smax = dS*M

	f = [ [-1]*(M+1) for _ in range(N+1) ]
	for i in range(M+1):
	    f[N][i] = max(K-i*dS,0)
	for i in range(N+1):
	    f[i][0] = K
	    f[i][M] = 0

	for idx in range(N):
	    a = []
	    b = []
	    i = N-1-idx
	    for jdx in range(M-1):
	        j = jdx+1
	        at = [0]*(M-1)
	        if jdx>0:
	            at[jdx-1] = 1/2*(r*j*dt) - 1/2*(sig*sig*j*j*dt)
	        at[jdx] = 1 + r*dt + sig*sig*j*j*dt
	        if jdx<M-2:
	            at[jdx+1] = -1/2*(r*j*dt) - 1/2*(sig*sig*j*j*dt)
	        bt = f[i+1][j]
	        if jdx==0:
	            bt = bt - (1/2*(r*j*dt) - 1/2*(sig*sig*j*j*dt))*f[i][0]
	        if jdx==M-2:
	            bt = bt - (-1/2*(r*j*dt) - 1/2*(sig*sig*j*j*dt))*f[i][M]
	        a.append(at)
	        b.append(bt)
	    a = np.array(a)
	    b = np.array(b)
	    x = np.linalg.solve(a, b)
	    for jdx in range(M-1):
	        j = jdx+1
	        f[i][j] = max(x[jdx],K-j*dS)
	return f[0][int(S0/dS)] 
print (finiteDifference(S0,K,r,sig,t,M,N,dS))