import jax, jax.numpy as jnp, numpy as np
def init(key,sizes):
    ps=[]
    for a,b in zip(sizes[:-1],sizes[1:]):
        key,k=jax.random.split(key); ps.append((jax.random.normal(k,(a,b))*np.sqrt(2/a), jnp.zeros(b)))
    return ps
def fwd(ps,x):
    for w,b in ps[:-1]: x=jax.nn.gelu(x@w+b)
    w,b=ps[-1]; return (x@w+b)[:,0]
def train(X,Y,Xv=None,Yv=None,hidden=(128,128),steps=6000,bs=2048,lr=2e-3,wd=1e-5,seed=0,loss='mse',verbose=True):
    mu=X.mean(0); sd=X.std(0)+1e-6
    Xn=jnp.array((X-mu)/sd); Yj=jnp.array(Y)
    ps=init(jax.random.PRNGKey(seed),[X.shape[1],*hidden,1])
    m=jax.tree_util.tree_map(jnp.zeros_like,ps); v=jax.tree_util.tree_map(jnp.zeros_like,ps)
    def lf(ps,x,y):
        r=fwd(ps,x)-y
        l = jnp.mean(r**2) if loss=='mse' else jnp.mean(jnp.where(jnp.abs(r)<1, 0.5*r**2, jnp.abs(r)-0.5))
        return l
    g=jax.jit(jax.value_and_grad(lf))
    @jax.jit
    def step(ps,m,v,t,key,lr_t):
        idx=jax.random.randint(key,(bs,),0,Xn.shape[0])
        l,gr=g(ps,Xn[idx],Yj[idx])
        m=jax.tree_util.tree_map(lambda a,b:0.9*a+0.1*b,m,gr); v=jax.tree_util.tree_map(lambda a,b:0.999*a+0.001*b*b,v,gr)
        ps=jax.tree_util.tree_map(lambda p,a,b:p-lr_t*((a/(1-0.9**t))/(jnp.sqrt(b/(1-0.999**t))+1e-8)+wd*p),ps,m,v)
        return ps,m,v,l
    key=jax.random.PRNGKey(seed+1)
    for t in range(1,steps+1):
        key,k=jax.random.split(key)
        lr_t=lr*0.5*(1+np.cos(np.pi*t/steps))
        ps,m,v,l=step(ps,m,v,t,k,lr_t)
        if verbose and (t%2000==0 or t==steps) and Xv is not None:
            pv=predict((ps,mu,sd),Xv); print(f"  step {t} train_loss={float(l):.4f} val_rmse={np.sqrt(np.mean((pv-Yv)**2)):.4f}",flush=True)
    return (ps,mu,sd)
def predict(model,X):
    ps,mu,sd=model; out=[]
    for i in range(0,len(X),65536): out.append(np.array(fwd(ps,jnp.array((X[i:i+65536]-mu)/sd))))
    return np.concatenate(out)
