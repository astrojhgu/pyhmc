import numpy as np
from numpy.random import normal, uniform



def leapfrog(q, p, last_hq_q, epsilon, h_q, h_p):
    p_mid=p-last_hq_q*epsilon/2.0
    q=q+h_p(p_mid)*epsilon
    last_hq_q=h_q(q)
    p=p_mid-last_hq_q*(epsilon/2.0)
    return (q,p, last_hq_q)

def kinetic(p, m=1):
    return np.dot(p,p)/m/2

class SamplerState:
    def __init__(self, epsilon, target_accept_ratio, adj_param):
        self.epsilon=epsilon
        self.target_accept_ratio=target_accept_ratio
        self.adj_param=adj_param

def sample(flogprob, grad_logprob, q0, lp, last_grad, l, sc: SamplerState):
    p=normal(size=q0.shape)
    current_k=kinetic(p)
    q=q0
    h_p=lambda p:p
    h_q=lambda q: -grad_logprob(q)

    last_hq_q=-last_grad

    for i in range(l):
        q, p, last_hq_q=leapfrog(q, p, last_hq_q, sc.epsilon, h_q, h_p)

    current_u=-lp
    proposed_u=-flogprob(q)
    proposed_k=kinetic(p)
    accepted=False
    if uniform()<np.exp(current_u-proposed_u+current_k-proposed_k):
        q0=q
        lp=-proposed_u
        last_grad=-last_hq_q
        accepted=True
        if uniform() < 1-sc.target_accept_ratio:
            sc.epsilon*=1.0+sc.adj_param
    else:
        if uniform()< sc.target_accept_ratio:
            sc.epsilon/=(1.0+sc.adj_param)

    return (q0, lp, last_grad, accepted)

def rosenbrock(x):
    return -np.sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2)

def delta(i,j):
    return 1 if i==j else 0

def diff_rosenbrock(x):
    result=np.zeros_like(x)
    for j in range(len(x)):
        for i in range(len(x)-1):
            result[j]-=200.0*(x[i+1]-x[i]**2)*(delta(j,i+1) -2*x[i]*delta(i,j))+2.0*(x[i]-1)*delta(i,j)

    return result



if __name__=='__main__':
    x=np.array([1.,1.])
    lp=rosenbrock(x)
    last_grad=diff_rosenbrock(x)
    sc=SamplerState(0.01, 0.9, 0.0001)
    of=open('a.txt','w')
    accept_cnt=0
    x_sum=np.zeros(2)
    for i in range(0, 10000000):
        x, lp, last_grad, accepted=sample(rosenbrock, diff_rosenbrock, x, lp, last_grad, 3, sc)
        x_sum+=x
        if accepted:
            accept_cnt+=1
        if i%1000==0:
            print(sc.epsilon, accept_cnt/(i+1), x_sum/(i+1))
        if i%100==0:
            of.write("{0} {1}\n".format(x[0], x[1]))
