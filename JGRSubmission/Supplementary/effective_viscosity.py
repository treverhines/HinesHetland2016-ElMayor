#!/usr/bin/env python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def ivt(uhat,s,n):
  ''' 
  Extension of the initial value theorem which computes the n'th
  derivative of u(t) evaluated at t=0 from the Laplace transform of
  u(t), uhat(s).

  This is eq. (A.5) in Appendix A

  PARAMETERS
  ----------
    uhat: Laplace transform of u. This is a symbolic expression
      containing s
    s: Laplace domain variable
    n: the derivative order

  RETURNS
  -------
    u_n: symbolic expression for the nth derivative of u evaluated at
      t=0
  '''
  if n == 0:
    expr = s*uhat
    u_0 = expr.limit(s,np.inf)
    return u_0

  elif n > 0:
    expr = s**(n+1)*uhat - sum(s**m*ivt(uhat,s,n-m) for m in range(1,n+1))
    u_n = expr.limit(s,np.inf)
    return u_n

def ilt(uhat,s,t,N):
  ''' 
  Evaluates the inverse Laplace transform of uhat(s) through a Taylor
  series expansion

  This is a combination of eqs. (A.7) and (A.5) in Appendix A

  PARAMETERS
  ----------
    uhat: Laplace transform of u. This is a symbolic expression
      containing s
    s: Laplace domain variable
    t: time domain variable
    N: order of the Taylor series expansion

  RETURNS
  -------
    series: symbolic expression for the series expansion of the u(x)
     about x=0
  '''
  series = sum((ivt(uhat,s,n)*t**n)/sp.factorial(n) for n in range(N+1))
  return series

s = sp.symbols('s')
t = sp.symbols('t')
mu = sp.symbols('mu')
muk = sp.symbols('mu_K')
muk1 = sp.symbols('mu_K1')
muk2 = sp.symbols('mu_K2')
etam = sp.symbols('eta_M')
etak = sp.symbols('eta_K')
etak1 = sp.symbols('eta_K1')
etak2 = sp.symbols('eta_K2')
sigma = sp.symbols('sigma_0')

# COMPUTE CREEP COMPLIANCE FOR MAXWELL MATERIAL
eps_hat = sigma*(1/mu + 1/(etam*s))/s

# COMPUTE CREEP COMPLIANCE FOR KELVIN MATERIAL
eps_hat = sigma/((etak*s + muk)*s)

# COMPUTE CREEP COMPLIANCE FOR ZENER MATERIAL
eps_hat = sigma/(mu*s) + sigma/((etak*s + muk)*s)

# COMPUTE CREEP COMPLIANCE FOR BURGERS MATERIAL
eps_hat = sigma*(1/mu + 1/(etam*s))/s + sigma/((etak*s + muk)*s)

# COMPUTE CREEP COMPLIANCE FOR GENERAL KELVIN MATERIAL
eps_hat = sigma*(1/mu + 1/(etam*s))/s + sigma/((etak1*s + muk1)*s) + sigma/((etak2*s + muk2)*s)

eps_t = ilt(eps_hat,s,t,5)

#numeps = eps_t.subs(muk,1)
#numeps = numeps.subs(etak,1)
#numeps = numeps.subs(sigma,1)
#numeps = sp.lambdify(t,numeps)
#times = np.linspace(0,10,100)
#u = numeps(times)
#plt.plot(times,u)
#plt.show()
eta_eff = (sigma/eps_t.diff(t)).subs(t,0).simplify().expand()

sp.pprint(eta_eff)
sp.pprint(eps_t)


