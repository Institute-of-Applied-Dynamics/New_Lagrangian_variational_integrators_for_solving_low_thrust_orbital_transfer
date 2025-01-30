##################################################################################
### Some generic helper functions needed to evaluate the low-order numerical methods
##################################################################################
import numpy as np
import matplotlib.pyplot as plt
def weighted_avg(x1,x2,wght):
    return wght*x1 + (1-wght)*x2

def Delta_x(x1,x2,params):
    return (x2 - x1)/params["h"]

def end_Time_calc(d,M,gamma,r0,rT):
    return d*np.sqrt(4*np.pi**2*(r0+rT)**3/(8*gamma*M)) 

def Cartesian_trafo(q):
    r,phi = q.flatten()
    return np.array([[r*np.cos(phi)],[r*np.sin(phi)]])

def cartesian_velocity_trafo(x,dx):
    r,phi=x
    dr,dphi=dx
    # return np.array([[dr*np.cos(phi) - r*np.sin(phi)*dphi],[dr*np.sin(phi) + r*np.cos(phi)*dphi]])
    return np.array([[- r*np.sin(phi)*dphi],[ r*np.cos(phi)*dphi]])


def stack_variables(q_d,lambda_d,mu,nu,U_d,use_u=True):
    '''stack here the variables into a single line needed for root finding'''
    N_term = len(q_d)
    dim_q = len(q_d[0])
    if use_u:
        stacked_vector =np.hstack( (q_d.reshape(N_term*dim_q),lambda_d.reshape(N_term*dim_q),mu.reshape(dim_q),nu.reshape(dim_q),U_d.reshape(N_term)))
    else:
        stacked_vector =np.hstack( (q_d.reshape(N_term*dim_q),lambda_d.reshape(N_term*dim_q),mu.reshape(dim_q),nu.reshape(dim_q) ))

    return stacked_vector

def unstack_variables(stacked_data,dim_q,dim_u,N_val,u_in_list=True):
    '''inverse function to stack_variables that outputs q_d,lambda_d,U_d,nu,mu from a given correctly sized numpy array like one would get from stack_variables'''
    if u_in_list:
        q_d,lam_d,mu,nu,U_d = (stacked_data[:dim_q*(N_val+1)].reshape([N_val+1,dim_q,1])
                           ,stacked_data[dim_q*(N_val+1):2*dim_q*(N_val+1)].reshape([N_val+1,dim_q,1])
                           ,stacked_data[2*dim_q*(N_val+1):2*dim_q*(N_val+1)+2].reshape([1,dim_q])
                           ,stacked_data[2*dim_q*(N_val+1)+2:2*dim_q*(N_val+1)+4].reshape([1,dim_q])
                           ,stacked_data[2*dim_q*(N_val+1)+4:].reshape([N_val+1,dim_u,1])
        )
    else:
        q_d,lam_d,mu,nu = (stacked_data[:dim_q*(N_val+1)].reshape([N_val+1,dim_q,1])
                           ,stacked_data[dim_q*(N_val+1):2*dim_q*(N_val+1)].reshape([N_val+1,dim_q,1])
                           ,stacked_data[2*dim_q*(N_val+1):2*dim_q*(N_val+1)+2].reshape([1,dim_q])
                           ,stacked_data[2*dim_q*(N_val+1)+2:2*dim_q*(N_val+1)+4].reshape([1,dim_q])
        )
        U_d = np.zeros([len(q_d),1])
    return q_d,lam_d,mu,nu,U_d


def stack_variables_two_Us(q_d,lambda_d,mu,nu,U_d_1,U_d_2,use_u=True):
    '''stack here the variables into a single line needed for root finding'''
    tmp_list = []
    solution=np.array([])
    for x in q_d:
        tmp_list.append(x.flatten())
    solution=np.append(solution,np.concatenate(tmp_list))
    tmp_list=[]
    for x in lambda_d:
        tmp_list.append(x.flatten())
    solution=np.append(solution,np.concatenate(tmp_list))
    solution=np.append(solution,np.concatenate(mu))
    solution=np.append(solution,np.concatenate(nu))
    if use_u:
        tmp_list=[]
        for x   in U_d_1:
            tmp_list.append(x.flatten())
        solution=np.append(solution,np.concatenate(tmp_list))
        tmp_list=[]
        for x   in U_d_2:
            tmp_list.append(x.flatten())
        solution=np.append(solution,np.concatenate(tmp_list))
    return solution

def unstack_variables_two_Us(stacked_data,dim_q,dim_u,N_val,u_in_list=True):
    '''inverse function to stack_variables that outputs q_d,lambda_d,U_d,nu,mu from a given correctly sized numpy array like one would get from stack_variables'''
    q_d = [np.zeros((dim_q,1)) for i in range(N_val+1)]
    lambda_d = [np.zeros((dim_q,1)) for i in range(N_val+1)]
    U_d_1 = [np.zeros((dim_u,1)) for i in range(N_val+1)]
    U_d_2 = [np.zeros((dim_u,1)) for i in range(N_val+1)]
    nu= np.zeros((1,dim_q))
    mu= np.zeros((1,dim_q))
    for i in range(N_val +1):
        for j in range(dim_q):
            q_d[i][j] = stacked_data[(dim_q)*i+j]
            lambda_d[i][j] = stacked_data[dim_q*(N_val +1 +i)+ j ]
    if u_in_list:
        for i in range(N_val +1):
            for j in range(dim_u):
                U_d_1[i][j] = stacked_data[2*dim_q+2*dim_q*(N_val+1) + dim_u*i+ j ]
                U_d_2[i][j] = stacked_data[2*dim_q+2*dim_q*(N_val+1)+dim_u*(N_val+1) + dim_u*i+ j ]

   
    for i in range(dim_q):
        mu[0,i] =  stacked_data[2*dim_q*(N_val+1) + i ]
        nu[0,i] =  stacked_data[dim_q+ 2*dim_q*(N_val+1) + i ]
    return q_d,lambda_d,mu,nu,U_d_1, U_d_2


def dphi_from_radius(radius,grav,Mass):
    return  np.sqrt(grav* Mass/radius**3)

def stack_variable(q_d):
    '''stack here variables, only q'''
    tmp_list = []
    solution=np.array([])
    for x in q_d:
        tmp_list.append(x.flatten())
    solution=np.append(solution,np.concatenate(tmp_list))
    return solution

def unstack_variable(stacked_data,dim_q,N_val):
    '''inverse function to stack_variables that outputs q_d from a given correctly sized numpy array like one would get from stack_variables
       only q
    '''
    q_d = [np.zeros((dim_q,1)) for i in range(N_val+1)]
    for i in range(N_val +1):
        for j in range(dim_q):
            q_d[i][j] = stacked_data[(dim_q)*i+j]
    return q_d

def rescale_stepsize_parameters(parameters, stepsize):
    parameters["h"] = stepsize
    parameters["N"] = int((parameters["t_N"]- parameters["t_0"])/parameters["h"])
    parameters["times"] = np.array([parameters["t_0"] + parameters["h"]*i for i in range(parameters["N"]+1)])

def plot_curve(q_vals,name,params):
    phase_space_evo = [[],[]]
    for x in q_vals:
        tmp = x
        phase_space_evo[0].append(tmp[0,0])
        phase_space_evo[1].append(tmp[1,0])
    plt.plot(phase_space_evo[0],phase_space_evo[1],'--*',label=name)    
    return q_vals    

