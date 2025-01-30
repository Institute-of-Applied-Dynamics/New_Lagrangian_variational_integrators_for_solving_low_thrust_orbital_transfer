#########################################
### The field functions for the low-thrust orbital transfer problem
#########################################
###########


import numpy as np
import Helper_functions as hf

#Base functions needed to evaluate the problem
def f(q,dq,params):
    ''' Cartesian Vector field depending on q=(x,y), dq=(dx,dy), for kepler/low thrust
    params= list of parameters = (γ,M)
    returns the vetor field f=(r,phi)'''
    x,y = q.flatten()
    dx,dy = dq.flatten()
    grav = params["grav"]
    Mass = params["M"]
    r = np.sqrt(x**2+y**2)
    return np.array([[- x], [- y]])*grav*Mass/r**3


def D1f(q,dq,params):
    '''derivative w.r.t. first argument q=(x,y), done by hand for simplicity
        D2 f = [(d/dx f_1, d/dx f_2),
           [(d/dy f_1, d/dy f_2)]'''
    x,y = q.flatten()
    dx,dy = dq.flatten()
    grav = params["grav"]
    Mass = params["M"]
    r = np.sqrt(x**2+y**2)
    return np.array([[-1 + 3*x**2/r**2, 3*y*x/r**2], [3*x*y/r**2, -1+3*y**2/r**2]])*grav*Mass/r**3

def D2f(q,dq,params):
    '''derivative/jacobian w.r.t. second argument dq=(dr,dφ), done by hand for simplicity
    D2 f = [(d/ddr f_1, d/ddr f_2),
           [(d/ddphi f_1, d/ddphi f_2)]
    '''
    x,y = q.flatten()
    dx,dy = dq.flatten()
    return np.array([[0,0], [0,0]])

#######
# #Terminal cost Φ(q,dq) and its derivatives
def Phi(q,dq,params):
    '''terminal cost function. Softly enforces the end position of the low thrust transfer
        Phi = a_q(q-q_end)^2 + a_dq(dq-dq_end)^2 = a_q((x-x_end)^2 + (y-yend)^2) + a_dq((dx-dy)^2 + (dx-dy)^2). 
        However, that here is the naive approach that does not respect the rotational nature of the kepler paths.
    returns a scalar'''
    x,y = q.flatten()
    dx,dy = dq.flatten()
    final_x,final_y = params["q_T"].flatten()
    final_dx,final_dy = params["dq_T"].flatten()
    Phi = params["q_T_weight"]*(((x-final_x)**2+ (y-final_y)**2)) 
    Phi+= params["dq_T_weight"]*((dx-final_dx)**2 + (dy-final_dy)**2)
    return Phi
def D1Phi(q,dq,params):
    '''terminal cost derivative first argument q, done by hand'''
    # x,y = q.flatten()
    # final_x,final_y = params["q_T"].flatten()
    # return  params["q_T_weight"]*np.array([[2*(x-final_x)],[2*(y-final_y)]])
    return  params["q_T_weight"]*2* (q - params["q_T"] ) # np.array([[2*(dx-final_dx)],[2*(dy-final_dy)]])


def D2Phi(q,dq,params):
    '''terminal cost derivative first argument dq, done by hand'''
    # dx,dy = dq.flatten()
    # final_dx,final_dy = params["dq_T"].flatten()
    return  params["dq_T_weight"]*2* (dq - params["dq_T"] ) # np.array([[2*(dx-final_dx)],[2*(dy-final_dy)]])

#Actuation matrix ρ(q)
def rho(q,params):
    '''matrix translating between control and state space
    from 1D to 2D, underactuated so not a symmetric matrix'''
    #fixit wrong cartesian force rho
    x,y = q.flatten()
    r = np.sqrt(x**2+y**2)
    return np.array([[-y],[x]])/r
#######
#######
#Control space metric g(q)
def g(q,params):
    '''metric of the control space, induced by the metric in Q and the linear anchor rho'''
    x,y = q.flatten()
    return np.array([1])
######
#b matrix definition
def b_mat(q,params):
    rho_mat = rho(q,params)
    g_mat = g(q,params)
    if len(g_mat) == 1:
        g_mat_inv = np.array([1/g_mat[0]]) #invert the 1D matrix 
    else:
        g_mat_inv = np.linalg.inv(g_mat) #use linalg to invert all dimensions
    return rho_mat *(g_mat_inv*rho_mat.transpose())

def Db_vec(q,lam,params):
    '''calculate D_{q} lam^T b lam used for covector field
        here we have 1/2 rho^T g^{-1} rho 
     '''

    x,y = q.flatten()
    lam_x,lam_y = lam.flatten()
    r2 = x**2+y**2
    diff_term = (x*lam_y - y*lam_x)
    sol= diff_term*np.array([[lam_y],[-lam_x]])/r2 - diff_term**2 *q /r2**2
    return sol

#here the analog for the u-dependent case, for easier evaluation
def  Db_u_vec(q,lam,u,params):
    '''The analog term to Db in the u-dependent case d/dq(λ^Tρu - 1/2 U^TgU)
    '''
    x,y = q.flatten()
    uval = u.flatten()[0]
    lam_x,lam_y = lam.flatten()
    diff_term = (x*lam_y - y*lam_x)
    r = np.sqrt(x**2+y**2)
    sol = np.array([[lam_y],[-lam_x]])*uval/r - (diff_term*uval/r**3)*q
    return  sol


###############
#u-λ relation
def lambda_u_relation(q,lam,u,params):
    '''returns the term ρ^Tλ -gu'''
    g_mat = g(q,params)
    rho_mat_T = rho(q,params).transpose()
    return rho_mat_T@lam - g_mat@u

def get_u_from_lambda(lam,q,params):
    g_mat = g(q,params)
    rho_mat_T = rho(q,params).transpose()
    if len(g_mat) == 1:
        g_mat_inv = np.array([1/g_mat[0]]) #invert the 1D matrix 
    else:
        g_mat_inv = np.linalg.inv(g_mat) #use linalg to invert all dimensions
    return g_mat_inv@(rho_mat_T@lam)


##############
#Running cost integrand I = int_0^T 1/2*(u^Tgu)
def evaluate_running_cost_integrand(q,u,parameters):
    return u.transpose()@(g(q,parameters)@u)



def calculate_control_Hamilton_function(q,lam,p_q,p_lam,params):
    '''explicitly evaluated hamilton function wihtout u-dependence'''
    hamiltonian = p_q.transpose()@p_lam  
    hamiltonian -= lam.transpose()@ f(q,p_lam,params)
    hamiltonian -= lam.transpose()@(b_mat(q,params) @ lam)/2
    return hamiltonian[0,0]

def  calculate_control_Hamilton_function_u(q,lam,p_q,p_lam,u,params):
    '''explicitly evaluated hamilton function wiht u-dependence'''
    hamiltonian = p_q.transpose()@p_lam  
    hamiltonian -= lam.transpose()@ (f(q,p_lam,params) + rho(q,params)@u )
    hamiltonian +=  u.transpose()@(g(q,params) @ u)/2
    return hamiltonian[0,0]