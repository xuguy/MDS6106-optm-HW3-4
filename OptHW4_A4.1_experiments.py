#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# def main function
def f(x1,x2):
    #return np.sqrt(1+x1**2+x2**2)
    return 0.5*x1**4-x1**3-x1**2+x1**2*x2**2+0.5*x2**4-x2**2

# 有2种计算grad的方式，一种精确，另一种数值，这里默认用数值法

def grad2d(f,x1,x2,h=1e-5, numerical = True):
    '''
    输入2个scalars，分别为x1 x2，输出一个gradient ndarray
    '''
    if numerical:
        grad_1 = (f(x1+h,x2)-f(x1-h,x2))/(2*h)
        grad_2 = (f(x1,x2+h)-f(x1,x2-h))/(2*h)
    else:
        grad_1 = 2*x1**3-3*x1**2-2*x1+2*x1*x2**2#
        grad_2 = 2*x1**2*x2+2*x2**3-2*x2#
    return np.stack([grad_1,grad_2],axis=-1)#.astype('float32')

def golden_section_minimizer(func, a, b, tol=1e-6, maxit=100):
    """
    Finds the minimum point of a scalar function `func` within the interval [a, b]
    using the Golden Section Search method.
    
    Parameters:
    func : callable
        The scalar function to minimize.
    a, b : float
        The interval [a, b] in which to search for the minimum.
    tol : float, optional
        The tolerance for convergence. The function will stop when the interval size is less than `tol`.
        
    Returns:
    float
        The estimated location of the minimum point.
    """
    # Golden ratio constant
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi  # Equivalent to (3 - sqrt(5)) / 2

    # Define the two interior points
    c = a + resphi * (b - a)
    d = b - resphi * (b - a)
    fc, fd = func(c), func(d)
    iternum = 0
    trace = []

    while abs(b - a) > tol and iternum<maxit:
        iternum+=1
        if fc < fd:
            b, d, fd = d, c, fc
            c = a + resphi * (b - a)
            fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = b - resphi * (b - a)
            fd = func(d)
        trace.append((a+b)/2)
    x_min = (a + b) / 2
    # The midpoint of the final interval is our best estimate for the minimum
    #print(f"interation number: {iternum}, x_min={x_min}")
    # 2 other results that cen be returned: iternum trace
    return x_min


#----------------- GD with armijo-----------------
class gradient_method:

    # class specified variables
    tol = 1e-5
    gamma = 0.1
    sigma = 0.5 # stepsize parameter for armijo
    max_iter = 10000
    

    def __init__(self,obj,grad):

        # function can be passed as a variable
        ''' 
        obj: callable function
            input: x1, x2, both are 1d scalars, can be extend to n-dim but here leave it alone.

            ouput: value at (x,y), 1d scalar

        grad: callable function
            input: (f,x1,x2,h=1e-5, numerical = True), where f is callable function defined as obj above, x1,x2 all ascalar as obj above, h=1e-5, the accuracy of calculating gradient using numerical method, numerical = True, use numerical method or accurate gradient.

            output: 2d ndarray point like (x1,x2)
        
            
        '''
        self.f=obj
        self.grad2d=grad
    
    def descent_backtrack(self,x0 ,max_backtrack = 100,gamma=gamma, sigma=sigma, tol=tol,max_iter=max_iter):

        #xk is a 2dim point ndarray
        xk=x0.copy()
        grad_k=self.grad2d(self.f,xk[0],xk[1])
        descent_direction_k=-1*grad_k
        trace = []
        
        for i in range(max_iter):
            backtrack = 0
            stepsize=1#2/3/4 might converge to different stationary point

            # checking armijo condition, if no, shrink step size then re-check armijo condition until armijo condition ture.
            while self.f((xk + stepsize*descent_direction_k)[0],(xk + stepsize*descent_direction_k)[1])-self.f(xk[0],xk[1]) > gamma*stepsize*(grad_k.T@descent_direction_k):
                stepsize *= sigma
                backtrack +=1
                # "while True" protect
                if backtrack>max_backtrack:
                    print('loop overflow')
                    break
            
            # store current location before moving to next
            trace.append(xk)

            # update location
            xk = xk + stepsize*descent_direction_k

            # calculate new grad and new descent direction based on the updated location
            grad_k = self.grad2d(self.f,xk[0],xk[1])
            descent_direction_k = -1*grad_k#/np.linalg.norm(grad), regularize or not
            
            # print location, stepsize and current norm of grad
            #print(f'x_{i}: ({xk[0]:.4f},{xk[1]:.4f}), stepsize: {stepsize},gradNorm: {np.linalg.norm(grad_k):.5f}')
            
            # check if gradient small enough to meet the stopping criterion
            if np.linalg.norm(grad_k)<=tol:
                trace=np.array(trace)
                #print(f'converged point: ({xk[0]:.4f},{xk[1]:.4f}),\ngrad norm now: {np.linalg.norm(grad_k):.5f},\nfunction value at stationary point: {f(xk[0],xk[1]):.4f}')
                output = f'&({x0[0]:.2f},{x0[1]:.2f}) & {i} & ({xk[0]:.2f},{xk[1]:.2f}) & {np.linalg.norm(x0-xk):.2f} \\\\ % backtrack'
                break
        return output,trace
    
    def exact_line_search(self,x0, tol=tol, search_upper = 1 ,max_iter=max_iter):
        a=search_upper
        xk=x0.copy()
        grad_k = grad2d(f,xk[0],xk[1])
        descent_direction_k =-1.*grad_k
        trace = []

        # def phi(x):
        #     return f((xk + x*descent_direction_k)[0],(xk + x*descent_direction_k)[1])

        # start GD with golden section
        for i in range(max_iter):
            #backtrack = 0
            a=1# golden search interval [0,a]

            # f((xk + stepsize*descent_direction_k)[0],(xk + stepsize*descent_direction_k)[1])
            stepsize = golden_section_minimizer(lambda x: f((xk + x*descent_direction_k)[0],(xk + x*descent_direction_k)[1]), 0 ,a)

            # store current location before moving to next
            trace.append(xk)
            
            xk = xk + stepsize*descent_direction_k
            grad_k = grad2d(f,xk[0],xk[1])
            descent_direction_k = -1.*grad_k#/np.linalg.norm(grad_k)
            
            #print(f'x_{i}: ({xk[0]:.4f},{xk[1]:.4f}), stepsize: {stepsize:.4f},gradNorm: {np.linalg.norm(grad_k):.5f}')
            
            if np.linalg.norm(grad_k)<=tol:
                trace=np.array(trace)
                output = f'&({x0[0]:.2f},{x0[1]:.2f}) & {i} & ({xk[0]:.2f},{xk[1]:.2f}) & {np.linalg.norm(x0-xk):.2f} \\\\ % exactlineS'
                # print(f'converged point: ({xk[0]:.4f},{xk[1]:.4f}),\ngrad norm now: {np.linalg.norm(grad_k):.5f},\nfunction value at stationary point: {f(xk[0],xk[1]):.4f}')
                break
        return output,trace
    
    def diminishing_stepsize(self,x0 , tol=tol,max_iter=max_iter):

        xk=x0.copy()
        grad_k = grad2d(f,xk[0],xk[1],numerical=False)
        descent_direction_k =-1.*grad_k
        trace = []
        # start GD with golden section
        for i in range(max_iter):
            #backtrack = 0
            #a, b = 0, 1# golden search interval

            # f((xk + stepsize*descent_direction_k)[0],(xk + stepsize*descent_direction_k)[1])
            stepsize = 1./np.sqrt(i+3)

            trace.append(xk)
            xk = xk + stepsize*descent_direction_k
            grad_k = grad2d(f,xk[0],xk[1],numerical=True)
            descent_direction_k = -1.*grad_k#/np.linalg.norm(grad_k)
            
            print(f'x_{i}: ({xk[0]:.4f},{xk[1]:.4f}), stepsize: {stepsize:.4f},gradNorm: {np.linalg.norm(grad_k):.5f}')
            
            if np.linalg.norm(grad_k,2)<=tol:
                trace=np.array(trace)
                
                output = f'&({x0[0]:.2f},{x0[1]:.2f}) & {i} & ({xk[0]:.2f},{xk[1]:.2f}) & {np.linalg.norm(x0-xk):.2f} \\\\ % diminishing'
                # print(f'converged point: ({xk[0]:.4f},{xk[1]:.4f}),\ngrad norm now: {np.linalg.norm(grad_k):.5f},\nfunction value at stationary point: {f(xk[0],xk[1]):.4f}')
                break
        return output,trace
    


#-------------- momentum --------------------

class inertia_gradient_method:

    # class specified variables
    tol = 1e-5
    max_iter = 10000

    def __init__(self,obj,grad):

        # function can be passed as a variable
        ''' 
        obj: callable function
            input: x1, x2, both are 1d scalars, can be extend to n-dim but here leave it alone.

            ouput: value at (x,y), 1d scalar

        grad: callable function
            input: (f,x1,x2,h=1e-5, numerical = True), where f is callable function defined as obj above, x1,x2 all ascalar as obj above, h=1e-5, the accuracy of calculating gradient using numerical method, numerical = True, use numerical method or accurate gradient.

            output: 2d ndarray point like (x1,x2)
        
            
        '''
        self.f=obj
        self.grad2d=grad
    
    def momentum_GD(self, x0, beta=np.array(0.7), max_iter = max_iter, tol=tol, search_upper=100):

        # initialize
        xk_minus_1 = x0.copy()
        xk=x0.copy()
        #beta = beta_candidate[beta_selector].copy()
        ell = beta.copy()
        alpha = 1.99*(1-beta)/ell
        grad_k = self.grad2d(self.f,xk[0],xk[1])
        trace=[]
        # iterate:

        for i in range(max_iter):
            momentum = (xk-xk_minus_1).copy()
            trace.append(xk)
            xk_minus_1 = xk.copy()

            alpha = 1.99*(1-beta)/ell
            xk = xk-alpha*self.grad2d(self.f,xk[0],xk[1])+beta*momentum
            grad_k = self.grad2d(self.f,xk_minus_1[0],xk_minus_1[1])

            # variables for estimating ell:

            # xkp1_ell = xk.copy()
            # xk_ell = xk_minus_1.copy()
            loop = 0
            #ell = beta.copy()
            while self.f(xk[0],xk[1])-self.f(xk_minus_1[0],xk_minus_1[1])>=grad_k.T@(xk-xk_minus_1)+ell*0.5*(np.linalg.norm(xk-xk_minus_1))**2 and loop < search_upper:
                loop+=1
                #print(xk,xk_minus_1)
                ell *= 2
                #print(ell,loop)
                
                alpha_ell = 1.99*(1-beta)/ell

                grad_k = self.grad2d(self.f,xk_minus_1[0],xk_minus_1[1])
                xk = xk_minus_1-alpha_ell*grad_k+beta*momentum
            

            if np.linalg.norm(grad_k)<=tol:
                trace=np.array(trace)
                output = f'&({x0[0]:.2f},{x0[1]:.2f}) & {i} & ({xk[0]:.2f},{xk[1]:.2f}) & {np.linalg.norm(x0-xk):.2f} \\\\ % momentum beta {beta:.1f}'
                # print(f'iter {i}, beta: {beta},\nconverged point: ({xk[0]:.4f},{xk[1]:.4f}),\ngrad norm now: {np.linalg.norm(grad_k):.5f},\nfunction value at stationary point: {self.f(xk[0],xk[1]):.4f}')
                break
        return output,trace
    

#%% plot

# initial points 
x0 = np.array([[-0.5,1],[-0.5,0.5],[-0.25,-0.5],[0.5,-0.5],[0.5,1]])

# stationary points of objective function:
# 0:local max, 1:global min, 2/3/4: saddle
x_star = np.array([[0.,0.],[2,0],[-0.5,0],[0,1],[0,-1]])
#-------------- grid search stepsize strat ---------------
# list to save trace
back_track_seq = []
exact_seq = []
diminish_seq = []
# list to save stats
back_track_stats = []
exact_stats = []
diminish_stats = []

stepsize_GD = gradient_method(f,grad2d)

for initial_point in x0:
    baack_track_stats_tmp, back_track_seq_tmp = stepsize_GD.descent_backtrack(initial_point)

    exact_stats_tmp, exact_seq_tmp = stepsize_GD.exact_line_search(initial_point)

    diminish_stats_tmp, diminish_seq_tmp = stepsize_GD.diminishing_stepsize(initial_point)

    back_track_seq.append(back_track_seq_tmp)
    back_track_stats.append(baack_track_stats_tmp)

    exact_seq.append(exact_seq_tmp)
    exact_stats.append(exact_stats_tmp)

    diminish_seq.append(diminish_seq_tmp)
    diminish_stats.append(diminish_stats_tmp)


df_stepsize_GD_trace =pd.DataFrame({'backtrack':back_track_seq,'exact':exact_seq,'diminish':diminish_seq})

df_stepsize_GD_stats = pd.DataFrame({'backtrack':back_track_stats,'exact':exact_stats,'diminish':diminish_stats})

# save data to local file
df_stepsize_GD_stats.to_csv('df_stepsize_GD_stats.csv')
df_stepsize_GD_trace.to_csv('df_stepsize_GD_trace.csv')


# define object finction contour plot
x1 = np.linspace(-2,3,100)
x2 = np.linspace(-2,2,100)
X, Y = np.meshgrid(x1,x2)
z = f(X,Y)
#%%

def plot_trace(X,Y,z,x0,x_star,trace,method='back track', filePathRegularize = False):
    colors = ['red','pink','blue','yellow','cyan']
    line_style = 'solid'
    marker_size = 100
    if filePathRegularize:
        image_path = method.replace('.','')+'.png'

    else:
        image_path = method+'.png'
    
    fig1, ax1 = plt.subplots(1,1,figsize = (10,10))
    contour = ax1.contour(X,Y,z,6,levels=30,colors='k',alpha=0.3)
    ax1.clabel(contour,inline=True)

    
    ax1.scatter(x0[:,0],x0[:,1],marker='o',s = marker_size,color='black',label='initial point')

    ax1.scatter(x_star[:,0],x_star[:,1],marker='x',s = marker_size,color='red', label = 'stationary points')
    for sta in x_star:
        ax1.text(sta[0],sta[1]+0.15,f'({sta[0]:.1f},{sta[1]:.1f})',horizontalalignment = 'center')
    # for trace in back_track:
    #     ax1.plot(trace[:,0],trace[:,1],linestyle='dashed',color='green')
    for i,trace in enumerate(trace):
        ax1.plot(trace[:,0],trace[:,1],linestyle=line_style,marker='.',color=colors[i],label = f'{x0[i]}')
        
    ax1.legend()
    ax1.set_xlabel(f'$x_1$')
    ax1.set_ylabel(f'$x_2$')
    ax1.set_title(method,fontweight="bold")
    fig1.savefig(image_path,bbox_inches='tight')
    
    plt.show()

plot_trace(X,Y,z,x0,x_star,df_stepsize_GD_trace['backtrack'],method = 'Back Tracking')
plot_trace(X,Y,z,x0,x_star,df_stepsize_GD_trace['exact'],method = 'Exact Line Search')
plot_trace(X,Y,z,x0,x_star,df_stepsize_GD_trace['diminish'],method = 'Diminishing Stepsize')

# print stats （in latex form):

for i in df_stepsize_GD_stats['backtrack']:
    print(i)

for i in df_stepsize_GD_stats['exact']:
    print(i)

for i in df_stepsize_GD_stats['diminish']:
    print(i)


#%%
#---------------- Momentum ----------------

# define contour plot:
x1 = np.linspace(-2,4,100)
x2 = np.linspace(-2,2,100)
X, Y = np.meshgrid(x1,x2)
z = f(X,Y)

# choose beta from 0-3
beta_candidate = np.array([0.3,0.5,0.7,0.9])

momentum = inertia_gradient_method(f,grad2d)

for beta_selector in range(4):
    momentum_seq = []
    momentum_stats = []
    for initial_point in x0:
        momentum_stats_tmp, momentum_seq_tmp = momentum.momentum_GD(initial_point,beta=beta_candidate[beta_selector])

        momentum_stats.append(momentum_stats_tmp)
        momentum_seq.append(momentum_seq_tmp)

    beta_tag = f'GD with momentum, beta={beta_candidate[beta_selector]}'


    plot_trace(X,Y,z,x0,x_star,momentum_seq,method = beta_tag,filePathRegularize=True)

    for i in momentum_stats:
        print(i)

#%% 
#-------------- extra calculation --------------
# calculate averge iteration it takes to converge
# backtrack_avg = (13+325+ 467 +12+ 10)/5
# exact_avg = (295+296+375+9+6)/5
# diminish_avg = (47+8523+8501+47+47)/5

# b03 = (26+33+24+24)/4
# b05 = (39+56+88+37+39)/5
# b07 = (105+88+102+77+79)/5
# b09 = (268+258+276+297+5151)/5

# backtrack_avg
# exact_avg
# diminish_avg
# b03
# b05
# b07
# b09


























#----------------- trash code --------------------
#%%

# def descent_backtrack(self,x0 , max_iter=10000):

#     #xk is a 2dim point ndarray
#     self.xk=x0.copy()
#     self.grad=self.grad2d(self.f,self.xk[0],self.xk[1])
#     self.descent_direction_k=-1*self.grad
#     self.trace = []
    
#     for i in range(max_iter):
#         backtrack = 0
#         stepsize=1#2/3/4 might converge to different stationary point

#         # checking armijo condition, if no, shrink step size then re-check armijo condition until armijo condition ture.
#         while self.f((self.xk + stepsize*self.descent_direction_k)[0],(self.xk + stepsize*self.descent_direction_k)[1])-self.f(self.xk[0],self.xk[1]) > gamma*stepsize*(self.grad.T@self.descent_direction_k):
#             stepsize *= sigma
#             backtrack +=1
#             # "while True" protect
#             if backtrack>max_backtrack:
#                 print('loop overflow')
#                 break
        
#         # store current location before moving to next
#         self.trace.append(self.xk)

#         # update location
#         self.xk = self.xk + stepsize*self.descent_direction_k

#         # calculate new grad and new descent direction based on the updated location
#         self.grad = self.grad2d(self.f,self.xk[0],self.xk[1])
#         self.descent_direction_k = -1*self.grad#/np.linalg.norm(grad), regularize or not
        
#         # print location, stepsize and current norm of grad
#         print(f'x_{i}: ({self.xk[0]:.4f},{self.xk[1]:.4f}), stepsize: {stepsize},gradNorm: {np.linalg.norm(self.grad):.5f}')
        
#         # check if gradient small enough to meet the stopping criterion
#         if np.linalg.norm(self.grad)<=tol:
#             self.trace=np.array(self.trace)
#             print(f'converged point: ({self.xk[0]:.4f},{self.xk[1]:.4f}),\ngrad norm now: {np.linalg.norm(self.grad):.5f},\nfunction value at stationary point: {f(self.xk[0],self.xk[1]):.4f}')
#             break



#%%

#---------------- template for momentum ----------------
# class gradient_method:

#     # class specified variables
#     tol = 1e-5
#     gamma = 0.1
#     sigma = 0.5 # stepsize parameter for armijo
#     max_iter = 10000
    

#     def __init__(self,obj,grad):

#         # function can be passed as a variable
#         ''' 
#         obj: callable function
#             input: x1, x2, both are 1d scalars, can be extend to n-dim but here leave it alone.

#             ouput: value at (x,y), 1d scalar

#         grad: callable function
#             input: (f,x1,x2,h=1e-5, numerical = True), where f is callable function defined as obj above, x1,x2 all ascalar as obj above, h=1e-5, the accuracy of calculating gradient using numerical method, numerical = True, use numerical method or accurate gradient.

#             output: 2d ndarray point like (x1,x2)
        
            
#         '''
#         self.f=obj
#         self.grad2d=grad
