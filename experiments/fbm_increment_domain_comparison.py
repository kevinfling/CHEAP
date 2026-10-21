#!/usr/bin/env python3
"""Matrix-free fBm DCT-PCG vs increment-domain fGn Chan-PCG.
Requires numpy/scipy. Prints true residuals and iterations.
This is an exploratory comparison, not an end-to-end FFTW timing study.
"""
import numpy as np
from scipy.fft import fft,ifft,dct,idct
from scipy.sparse.linalg import cg,LinearOperator
def trial(n=4096,h=.3,reg=1e-4,seed=0):
 t=np.arange(1,n+1)/n;u=.5*t**(2*h);c=-.5*(np.arange(n)/n)**(2*h)
 def spectrum(lags):return fft(np.r_[lags,0,lags[:0:-1]])
 def toep(v,s):return ifft(fft(np.r_[v,np.zeros(n)])*s).real[:n]
 fc=spectrum(c)
 def a0(v):return toep(v,fc)+u*v.sum()+u@v
 v=np.ones(n)/np.sqrt(n)
 for _ in range(50):v=a0(v);v/=np.linalg.norm(v)
 rho=reg*(v@a0(v))
 def av(v):return a0(v)+rho*v
 b=np.random.default_rng(seed).normal(size=n)
 p=np.zeros(2*n);q=p.copy();p[1:n]=(n-np.arange(1,n))*c[1:];q[1:n]=c[1:]
 fp,fq=fft(p),fft(q);w=np.empty(n);w[0]=c[0]+2*p.sum()/n+2*u.sum()+rho
 k=np.arange(1,n);w[1:]=c[0]+2/n*(fp.real[k]+fq.imag[k]/np.sin(np.pi*k/n))+rho
 assert np.min(w)>0
 def dctpre(v):return idct(dct(v,norm='ortho')/w,norm='ortho')
 d=np.arange(n);g=.5*(np.abs(d-1)**(2*h)-2*d**(2*h)+(d+1)**(2*h))/n**(2*h)
 fg=spectrum(g)
 def D(v):return np.r_[v[0],np.diff(v)]
 def DT(v):return np.r_[v[:-1]-v[1:],v[-1]]
 def gv(v):return toep(v,fg)+rho*D(DT(v))
 j=np.arange(n);cc=np.empty(n);cc[0]=g[0]+rho*(2-1/n)
 cc[1:]=(1-j[1:]/n)*g[1:]+j[1:]/n*g[:0:-1]
 cc[1]-=rho*(n-1)/n;cc[-1]-=rho*(n-1)/n
 eig=fft(cc).real;assert eig.min()>0
 def chan(v):return ifft(fft(v)/eig).real
 for name,op,rhs,pre,back,tol in (
  ('CG',av,b,None,lambda y:y,1e-9),
  ('DCT-PCG',av,b,dctpre,lambda y:y,1e-9),
  ('fGn-Chan',gv,D(b),chan,DT,1e-12)):
  cnt=[0]
  def cb(_):cnt[0]+=1
  A=LinearOperator((n,n),matvec=op)
  M=None if pre is None else LinearOperator((n,n),matvec=pre)
  y,info=cg(A,rhs,M=M,rtol=tol,atol=0,callback=cb,maxiter=5000)
  res=np.linalg.norm(av(back(y))-b)/np.linalg.norm(b)
  print(n,h,reg,seed,name,cnt[0],res,info)
  assert info==0 and res<=1.1e-9
if __name__=='__main__':
 for n in (256,1024,4096):
  for h in (.3,.7):
   for reg in (1e-4,1e-3):
    for seed in (0,1,2):trial(n,h,reg,seed)
