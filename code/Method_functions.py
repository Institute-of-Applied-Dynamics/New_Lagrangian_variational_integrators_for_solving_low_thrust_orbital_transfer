import numpy as np
from scipy import optimize as op
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import scipy.optimize as opt
######
# #Define discrete paths
from Field_functions import *
from Helper_functions import *


def vector_field_u_eval(q_d,lam_d,U_d, k,gamma,beta,invert,params):
    '''Evaluate here the vector field with explicit U-dependence for a given gamma, beta choice
    invert is a boolean that decides whether the global gamma prefactor is inverted or not - needed for the four evaluations/chain rule evaluation of d/dq_k'''
    q_wht = weighted_avg(q_d[k], q_d[k+1], gamma)
    U_wht = U_d[k]      # using the fact that we work with the U_d^1, U_d^2, give as argument only the relevant one and treat both U_d explicitly
    Delta_q = Delta_x(q_d[k],q_d[k+1],params)
    f_gamma = f(q_wht,Delta_q,params)
    rho_gamma = rho(q_wht,params)
    sol = (f_gamma + rho_gamma @ U_wht)
    # sol = f_gamma ##for testing purposes used 
    if invert:
        sol *= (1-gamma)
    else:
        sol *= gamma
    return sol
    
def vector_field_eval(q_d,lam_d,U_d, k,gamma,beta,invert,params):
    '''Evaluate here the vector field withou U-dependence for a given gamma, beta choice
    For transferability the arguments stay the same, so only another vector field needs to be supplied
    invert is a boolean that decides whether the global gamma prefactor is inverted or not - needed for the four evaluations/chain rule evaluation of d/dq_k
    '''
    q_wht = weighted_avg(q_d[k], q_d[k+1], gamma)
    lam_wht = weighted_avg(lam_d[k],lam_d[k+1],gamma)
    Delta_q = Delta_x(q_d[k],q_d[k+1],params)
    f_gamma = f(q_wht,Delta_q,params)
    b_gamma = b_mat(q_wht,params)
    sol = f_gamma + b_gamma @ lam_wht
    if invert:
        sol *= (1-gamma)
    else:
        sol *= gamma
    return sol
    


######################
# Co-Vectorfield evaluations
def covector_field_u_eval(q_d,lam_d,U_d, k,gamma,beta,invert,params):
    '''Evaluate here the co-vector field with explicit U-dependence for a given gamma, beta choice'''
    q_wht = weighted_avg(q_d[k], q_d[k+1], gamma)
    lam_wht = weighted_avg(lam_d[k], lam_d[k+1],gamma)
    U_wht = U_d[k]
    Delta_lam = Delta_x(lam_d[k],lam_d[k+1],params)
    Delta_q = Delta_x(q_d[k],q_d[k+1],params)
    
    D1f_eval = D1f(q_wht,Delta_q,params)
    D2f_eval = D2f(q_wht,Delta_q,params)
    Db_eval = Db_u_vec(q_wht,lam_wht,U_wht,params)
    sol= np.array([[0],[0]])
    if invert:
        sol = ( (1-gamma)* D1f_eval + D2f_eval/params["h"])@lam_wht + Db_eval *(1-gamma)
    else:
        sol= ( gamma* D1f_eval - D2f_eval/params["h"])@lam_wht + Db_eval *gamma
    return sol

def covector_field_eval(q_d,lam_d,U_d, k,gamma,beta,invert,params):
    '''Evaluate here the co-vector field with explicit U-dependence for a given gamma, beta choice'''
    q_wht = weighted_avg(q_d[k], q_d[k+1], gamma)
    lam_wht = weighted_avg(lam_d[k], lam_d[k+1],gamma)
    # U_wht = weighted_avg(U_d[k],U_d[k+1],beta)
    Delta_lam = Delta_x(lam_d[k],lam_d[k+1],params)
    Delta_q = Delta_x(q_d[k],q_d[k+1],params)
    
    D1f_eval = D1f(q_wht,Delta_q,params)
    D2f_eval = D2f(q_wht,Delta_q,params)
    Db_eval = Db_vec(q_wht,lam_wht,params)
    sol= np.array([[0],[0]])
    if invert:
        sol = ( (1-gamma)* D1f_eval +D2f_eval/params["h"])@lam_wht + Db_eval *(1-gamma)
    else:
        sol = ( gamma* D1f_eval -D2f_eval/params["h"])@lam_wht + Db_eval *gamma
    return sol

def calculate_total_running_cost(q_d,U_d_1,U_d_2,parameters):
    total_cost = 0
    alpha=parameters["alpha"]
    beta = parameters["beta"]
    gamma = parameters["gamma"]
    for k in range(parameters["N"]):
        q_wht = weighted_avg(q_d[k], q_d[k+1], gamma)
        U_wht =U_d_1[k]
        total_cost += parameters["h"]* alpha * evaluate_running_cost_integrand(q_wht,U_wht,parameters)/2
        q_wht = weighted_avg(q_d[k], q_d[k+1], 1-gamma)
        U_wht = U_d_2[k]
        total_cost += parameters["h"]*(1-alpha) * evaluate_running_cost_integrand(q_wht,U_wht,parameters)/2
    return total_cost    

def calculate_conserved_quantity(q,dq,lam,dlam,parameters):
    x,y = q.flatten()
    dx,dy = dq.flatten()
    lamx,lamy = lam.flatten()
    dlamx,dlamy = dlam.flatten()
    return x*dlamy - y*dlamx - lamy*dx + lamx*dy

########
#Boundary velocity - legendre pair definitions
def dq0(q_d, lam_d, U_d_1,U_d_2,vectorfield,params):
    '''supply as vectorfield either vector_field_eval or vector_field_u_eval for the two cases'''
    DeltaX = Delta_x(q_d[0],q_d[1],params)
    h = params["h"]
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    sol = DeltaX - h * alpha *  vectorfield(q_d,lam_d,U_d_1,0,gamma,beta,False,params) 
    sol -= h* (1-alpha)* vectorfield(q_d,lam_d, U_d_2,0, 1-gamma,1-beta,False,params)
    return sol

def dqT(q_d, lam_d, U_d_1,U_d_2,vectorfield,params):
    '''supply as vectorfield either vector_field_eval or vector_field_u_eval for the two cases'''
    Nval= params['N']
    DeltaX = Delta_x(q_d[Nval-1],q_d[Nval],params)
    h = params["h"]
    alpha = params["alpha"]
    gamma = params["gamma"]
    beta = params["beta"]
    sol = DeltaX + h * alpha  * vectorfield(q_d,lam_d,U_d_1,Nval-1,gamma,beta,True,params) 
    sol += h* (1-alpha)* vectorfield(q_d,lam_d, U_d_2,Nval-1, 1-gamma,1-beta,True,params)
    return sol

def dqk(q_d, lam_d, U_d_1,U_d_2,k,vectorfield,params):
    '''supply as vectorfield either vector_field_eval or vector_field_u_eval for the two cases'''
    h = params["h"]
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    Nval= params['N']
    if k < params["N"]:
        DeltaX = Delta_x(q_d[k],q_d[k+1],params)
        sol = DeltaX - h * alpha *  vectorfield(q_d,lam_d,U_d_1,k,gamma,beta,False,params) 
        sol -= h* (1-alpha)* vectorfield(q_d,lam_d, U_d_2,k, 1-gamma,1-beta,False,params)
    else:
        DeltaX = Delta_x(q_d[Nval-1],q_d[Nval],params)
        sol = DeltaX + h * alpha  * vectorfield(q_d,lam_d,U_d_1,Nval-1,gamma,beta,True,params) 
        sol += h* (1-alpha)* vectorfield(q_d,lam_d, U_d_2,Nval-1, 1-gamma,1-beta,True,params)
    # print(vectorfield(q_d,lam_d,U_d,k,gamma,beta,False,params))
    return sol


def dlam_k(q_d,lam_d,U_d_1,U_d_2,k, covector_field,parameters):
    '''Define here the lambda-velocities needed for the conserved quantity calculation.
    for k=0,...,N'''
    alpha = parameters["alpha"]
    beta = parameters["beta"]
    gamma = parameters["gamma"]
    if k < parameters["N"]:
        cofield1 = covector_field(q_d,lam_d,U_d_1,k,gamma,beta,False,parameters)
        cofield3 = covector_field(q_d,lam_d,U_d_2,k, 1-gamma,1-beta,False,parameters)
        DeltaX = Delta_x(lam_d[k],lam_d[k+1],parameters)
        sol = DeltaX - parameters["h"]*(alpha * (cofield1 ) + (1-alpha)*(cofield3 ))
    else:
        cofield2 = covector_field(q_d,lam_d,U_d_1,k-1,gamma,beta,True,parameters)
        cofield4=  covector_field(q_d,lam_d,U_d_2,k-1,1-gamma,1-beta,True,parameters)
        DeltaX = Delta_x(lam_d[k-1],lam_d[k],parameters)
        sol = DeltaX + parameters["h"]*(alpha * ( cofield2) + (1-alpha)*( cofield4))
    return sol
        





def eval_hamilton_func_discrete_path(q_d,lam_d,k,parameters,U_d_12=None):
    ''' Evaluation not generally correct, uses p_y=dy  identity, ok for cartesian low thrust, but not generally!
    If f(q,dq) != f(q) then calculates wrong. Also vector field evaluation is very explicit, not super general defined
    fast explicit evaluation of the hamilton function from the discrete paths at a certain step.
    It is assumed that the step-variables are used: however, also the control at nodal values U_d_12, 
    Is it possible to calculate the hamiltonfunction otherwise? only via nodal values?
    '''
    
    q_k = q_d[k]
    lam_k = lam_d[k]
    if U_d_12 == None:
        #Note here that for the example the momenta and velocities are trivially linked
        #not necessarily true if f(q,\dot q) \neq f(q)
        U_d_12 = [None,None]
        p_q_k = dlam_k(q_d,lam_d,*U_d_12,k,covector_field_eval,parameters)
        p_lam_k = dqk(q_d,lam_d,*U_d_12,k,vector_field_eval,parameters)
        return calculate_control_Hamilton_function(q_k,lam_k,p_q_k,p_lam_k,parameters)
    else:
        p_q_k = dlam_k(q_d,lam_d,*U_d_12,k,covector_field_u_eval,parameters)
        p_lam_k = dqk(q_d,lam_d,*U_d_12,k,vector_field_u_eval,parameters)
        
        u_d_from_calc = [np.array([get_u_from_lambda(lam,q_val,parameters)] ) for lam,q_val in zip(lam_d,q_d)]
        u_k = u_d_from_calc[k]
        return calculate_control_Hamilton_function_u(q_k,lam_k,p_q_k,p_lam_k,u_k,parameters)








#######################
#### The class handling the generation of the optimality conditions of the new augmented approach
#######################


class optimality_conditions_generator:
    '''This class will create the optimality conditions for the u-independent case 
    (potentially best to split it, but maybe it is also ok to have both at the same time and just work with a flag...)
    The idea is, that this is mostly generic, i.e. just supply the force fields and consistent discrete paths, then everything should be applicable to any other system'''
    def __init__(self,vector_field,covector_field,params):
        self.parameters = params
        # self.q_d_start = copy.deepcopy(q_d) 
        # self.lambda_d_start = copy.deepcopy(lambda_d)
        # self.U_d_start = copy.deepcopy(U_d)
        self.vector_field = vector_field
        self.covector_field = covector_field
        self.gamma = params["gamma"]
        self.beta =  params["beta"]
        self.alpha = params["alpha"]
    
    def create_q_DEL(self,k,q_d, lam_d, U_d_1,U_d_2):
        '''Function that creates the δq-DEL for a choice of k'''
        if k == 0:
            raise Exception("invalid k-choice for δq_DEL, needs to be larger than 0")
        elif k >= self.parameters["N"]:
            raise Exception("Invalid k-choice for δq_DEL, only smaller than N allowed")
        
        DDeltalam = (lam_d[k+1] - 2*lam_d[k] + lam_d[k-1])
        cofield1 = self.covector_field(q_d,lam_d,U_d_1,k  ,self.gamma  ,self.beta,False,self.parameters)
        cofield2 = self.covector_field(q_d,lam_d,U_d_1,k-1,self.gamma  ,self.beta,True,self.parameters)
        cofield3 = self.covector_field(q_d,lam_d,U_d_2,k  ,1-self.gamma,1-self.beta,False,self.parameters)
        cofield4 = self.covector_field(q_d,lam_d,U_d_2,k-1,1-self.gamma,1-self.beta,True,self.parameters)
        sol = DDeltalam - self.parameters["h"]**2*(self.alpha * (cofield1 + cofield2) + (1-self.alpha)*(cofield3 + cofield4))
        return sol
    
    def create_lambda_DEL(self,k,q_d, lam_d, U_d_1,U_d_2):
        '''Function that creates the δλ-DEL for a choice of k'''
        if k == 0:
            raise Exception("invalid k-choice for δλ-DEL, needs to be larger than 0")
        elif k >= self.parameters["N"]:
            raise Exception("Invalid k-choice for δλ-DEL, only smaller than N allowed")
        
        DDeltaq = (q_d[k+1] - 2*q_d[k] + q_d[k-1])
        field1 = self.vector_field(q_d,lam_d,U_d_1, k, self.gamma,self.beta, False,self.parameters)
        field2 = self.vector_field(q_d,lam_d,U_d_1, k-1, self.gamma,self.beta, True,self.parameters)
        field3 = self.vector_field(q_d,lam_d,U_d_2, k, 1-self.gamma,1-self.beta, False,self.parameters)
        field4 = self.vector_field(q_d,lam_d,U_d_2, k-1, 1-self.gamma,1-self.beta, True,self.parameters)
        sol = DDeltaq - self.parameters["h"]**2*(self.alpha*(field1+field2) +(1-self.alpha)*(field3+field4))
        return sol
    
    # def __ulam_rel_eval(self,k,gamma,beta,alpha,lam_d,U_d,q_d,invert):
    #     lam_wht = weighted_avg(lam_d[k],lam_d[k+1],gamma)
    #     q_wht = weighted_avg(q_d[k], q_d[k+1],gamma)
    #     U_wht = weighted_avg(U_d[k],U_d[k+1],beta)
    #     if invert:
    #         return alpha*(1-self.beta)*lambda_u_relation(q_wht,lam_wht,U_wht,self.parameters)
    #     else:
    #         return alpha*self.beta*lambda_u_relation(q_wht,lam_wht,U_wht,self.parameters)
        
    def create_lambda_u_relation_DEL_1(self,k,q_d, lam_d, U_d_1):
        '''create the the bulk lambda-u relation for a choice of k
        αβ(λρ-Ug)+ etc, no h multiplied'''     
        if k >= self.parameters["N"]:
            return 0* U_d_1[k]  #just return 0 vector for the 
            # raise Exception("Invalid k-choice for λ-u-DEL, only smaller than N allowed")

        lam_wht = weighted_avg(lam_d[k],lam_d[k+1],self.gamma)
        q_wht = weighted_avg(q_d[k], q_d[k+1],self.gamma)
        U_wht = U_d_1[k]
        return self.parameters['alpha']*lambda_u_relation(q_wht,lam_wht,U_wht,self.parameters)
       
    def create_lambda_u_relation_DEL_2(self,k,q_d, lam_d, U_d_2):
        '''create the the bulk lambda-u relation for a choice of k
        αβ(λρ-Ug)+ etc, no h multiplied'''     
        if k >= self.parameters["N"]:
            return 0* U_d_2[k]
            # raise Exception("Invalid k-choice for λ-u-DEL, only smaller than N allowed")
        lam_wht = weighted_avg(lam_d[k],lam_d[k+1],1-self.gamma)
        q_wht = weighted_avg(q_d[k], q_d[k+1],1-self.gamma)
        U_wht = U_d_2[k]
        return (1-self.parameters['alpha'])*lambda_u_relation(q_wht,lam_wht,U_wht,self.parameters)
      

    def initial_position_condition(self,q_d, lam_d, U_d,mu,nu):
        return q_d[0] - self.parameters["q_0"]

    def initial_velocity_condition(self,q_d, lam_d, U_d_1,U_d_2,mu,nu):
        return dq0(q_d,lam_d,U_d_1,U_d_2,self.vector_field,self.parameters) - self.parameters["dq_0"]
    
    def final_lambda_identity(self,q_d, lam_d, U_d_1,U_d_2):
        '''The postulated final lambda relation informed by the continuous setting'''
        N_val = self.parameters["N"]
        return D2Phi(q_d[N_val], dqT(q_d,lam_d,U_d_1,U_d_2,self.vector_field,self.parameters),self.parameters) + lam_d[N_val]    


    def final_q_condition(self,q_d, lam_d,U_d_1,U_d_2):
        N_val = self.parameters["N"]
        h = self.parameters["h"]
        Deltalam = (lam_d[N_val] - lam_d[N_val-1] )
        cofield2 = h*self.covector_field(q_d,lam_d,U_d_1,N_val-1,self.gamma,self.beta,True,self.parameters)
        cofield4=  h*self.covector_field(q_d,lam_d,U_d_2,N_val-1,1-self.gamma,1-self.beta,True,self.parameters)
        sol = h*D1Phi(q_d[N_val], dqT(q_d,lam_d,U_d_1,U_d_2,self.vector_field,self.parameters),self.parameters) - Deltalam - self.alpha *h* cofield2 - (1-self.alpha)*h* cofield4
        return sol

    def initial_q_condition(self,q_d, lam_d, U_d_1,U_d_2,mu,nu):
        h = self.parameters["h"]
        Deltalam = (lam_d[1] - lam_d[0] )
        cofield2 = h*self.covector_field(q_d,lam_d,U_d_1,0,self.gamma,self.beta,False,self.parameters)
        cofield4=  h*self.covector_field(q_d,lam_d,U_d_2,0,1-self.gamma,1-self.beta,False,self.parameters)
        sol = h*mu.transpose() + Deltalam - self.alpha *h* cofield2 - (1-self.alpha)*h* cofield4
        return sol
        
    def initial_lambda_identity(self,q_d, lam_d, U_d,mu,nu):
        '''The postulated initial lambda relation informed by the continuous setting'''
        return nu.transpose() - lam_d[0]

    def create_all_optimality_conditions(self,q_d, lam_d,mu,nu, U_d_1,U_d_2):
        ''' This function takes the values for the variables q_d, lam_d, U_d, mu,nu and  outputs the full set of optimality conditions as a list, useable for optimization'''
        list_of_optimality_conditions = []
        #boundary optimality conditions
        list_of_optimality_conditions.append(self.final_lambda_identity(q_d,lam_d,U_d_1,U_d_2).flatten())             #creates probs
        list_of_optimality_conditions.append(self.final_q_condition(q_d,lam_d,U_d_1,U_d_2).flatten())
        list_of_optimality_conditions.append(self.initial_lambda_identity(q_d,lam_d,U_d_1,mu,nu).flatten())
        list_of_optimality_conditions.append(self.initial_q_condition(q_d,lam_d,U_d_1,U_d_2,mu,nu).flatten())
        list_of_optimality_conditions.append(self.initial_position_condition(q_d,lam_d,U_d_1,mu,nu).flatten())
        list_of_optimality_conditions.append(self.initial_velocity_condition(q_d,lam_d,U_d_1,U_d_2,mu,nu).flatten())    #creates probs
        # #discrete Euler Lagrange equations
        for k in range(1,self.parameters["N"]):
            list_of_optimality_conditions.append(self.create_q_DEL(k,q_d,lam_d,U_d_1,U_d_2).flatten())
            list_of_optimality_conditions.append(self.create_lambda_DEL(k,q_d,lam_d,U_d_1,U_d_2).flatten())

        for k in range(0,self.parameters["N"]+1):
            list_of_optimality_conditions.append(self.create_lambda_u_relation_DEL_1(k,q_d,lam_d,U_d_1).flatten())
            list_of_optimality_conditions.append(self.create_lambda_u_relation_DEL_2(k,q_d,lam_d,U_d_2).flatten())    
        return list_of_optimality_conditions
    
    def create_all_optimality_conditions_debugging_many_arguments(self,q_d, lam_d,mu,nu, U_d_1,U_d_2):
        ''' problem that u_d_1,u_d_2 are often redundant, convergence bad. check here if it can be fixed just only working with one u, e.g. in the case of alpha=1=beta=gamma and the 1,0 case too
        Schould then be completely equivalent in these cases'''
        list_of_optimality_conditions = []
        #boundary optimality conditions
        list_of_optimality_conditions.append(self.final_lambda_identity(q_d,lam_d,U_d_1,U_d_2).flatten())             #creates probs
        list_of_optimality_conditions.append(self.final_q_condition(q_d,lam_d,U_d_1,U_d_2).flatten())
        list_of_optimality_conditions.append(self.initial_lambda_identity(q_d,lam_d,U_d_1,mu,nu).flatten())
        list_of_optimality_conditions.append(self.initial_q_condition(q_d,lam_d,U_d_1,U_d_2,mu,nu).flatten())
        list_of_optimality_conditions.append(self.initial_position_condition(q_d,lam_d,U_d_1,mu,nu).flatten())
        list_of_optimality_conditions.append(self.initial_velocity_condition(q_d,lam_d,U_d_1,U_d_2,mu,nu).flatten())    #creates probs
        # #discrete Euler Lagrange equations
        for k in range(1,self.parameters["N"]):
            list_of_optimality_conditions.append(self.create_q_DEL(k,q_d,lam_d,U_d_1,U_d_2).flatten())
            list_of_optimality_conditions.append(self.create_lambda_DEL(k,q_d,lam_d,U_d_1,U_d_2).flatten())

        for k in range(0,self.parameters["N"]+1):
            list_of_optimality_conditions.append(self.create_lambda_u_relation_DEL_1(k,q_d,lam_d,U_d_1).flatten())
            # list_of_optimality_conditions.append(self.create_lambda_u_relation_DEL_2(k,q_d,lam_d,U_d_2).flatten())    
        return list_of_optimality_conditions

    def create_all_optimality_conditions_no_u(self,q_d, lam_d,mu,nu,U_d_1,U_d_2):
        ''' This function takes the values for the variables q_d, lam_d, U_d, mu,nu and  outputs the full set of optimality conditions as a list, useable for optimization'''
        list_of_optimality_conditions = []
        #boundary optimality conditions
        list_of_optimality_conditions.append(self.final_lambda_identity(q_d,lam_d,U_d_1,U_d_2).flatten())
        list_of_optimality_conditions.append(self.final_q_condition(q_d,lam_d,U_d_1,U_d_2).flatten())
        # list_of_optimality_conditions.append(simplifier*self.final_lambda_u_condition(q_d,lam_d,U_d).flatten())
        list_of_optimality_conditions.append(self.initial_lambda_identity(q_d,lam_d,U_d_1,mu,nu).flatten())
        list_of_optimality_conditions.append(self.initial_q_condition(q_d,lam_d,U_d_1,U_d_2,mu,nu).flatten())
        # list_of_optimality_conditions.append(simplifier*self.initial_lambda_u_condition(q_d,lam_d,U_d,mu,nu).flatten())
        list_of_optimality_conditions.append(self.initial_position_condition(q_d,lam_d,U_d_1,mu,nu).flatten())
        list_of_optimality_conditions.append(self.initial_velocity_condition(q_d,lam_d,U_d_1,U_d_2,mu,nu).flatten())
        #discrete Euler Lagrange equations
        for k in range(1,self.parameters["N"]):
            list_of_optimality_conditions.append(self.create_q_DEL(k,q_d,lam_d,U_d_1,U_d_2).flatten())
            list_of_optimality_conditions.append(self.create_lambda_DEL(k,q_d,lam_d,U_d_1,U_d_2).flatten())
            # list_of_optimality_conditions.append(simplifier*self.create_lambda_u_relation_DEL(k,q_d,lam_d,U_d).flatten())
        return list_of_optimality_conditions
    
    def standard_calculate_terminal_cost(self,q_d,v_d,U_d):
        Deltaq = (self.parameters['q_T'] - q_d[-1]).flatten()
        Deltadq = (self.parameters['dq_T']-v_d[-1]).flatten()
        return self.parameters["q_T_weight"]* Deltaq@Deltaq+ self.parameters['dq_T_weight']*Deltadq@Deltadq
    
    def standard_calculate_q_v(self,U_d):
        def midpoint_eval_traj(qk,vk,qk1,vk1,Uk,Uk1):
            h= self.parameters["h"]
            midpt_q = (qk+qk1)/2
            midpt_v = (vk+vk1)/2
            midpt_U = (Uk+Uk1)/2
            qk_eval = qk1-qk-h*midpt_v 
            vk_eval = vk1-vk - h*(f(midpt_q,midpt_v,self.parameters)+ rho(midpt_q,self.parameters)@midpt_U)
            return np.concatenate([qk_eval.flatten(),vk_eval.flatten()])
        def simple_stack(q,v):
            stacked_data = q.flatten()
            stacked_data = np.append(stacked_data,v.flatten())
            return stacked_data
        def simple_unstack(stacked_variables):
            '''unstack qk,qk1,vk,kvk1,uk,uk1'''
            return  np.reshape(stacked_variables,[2,2,1])

        solution_q = [self.parameters["q_0"]]
        solution_v = [self.parameters["dq_0"]]
        for i in range(self.parameters["N"]):
            U_k_val = U_d[i]
            U_k1_val = U_d[i+1] 
            qk_val = solution_q[-1]
            vk_val = solution_v[-1]
            step_evolve = lambda X : midpoint_eval_traj(qk_val,vk_val,*simple_unstack(X),U_k_val,U_k1_val)
            solution = opt.root(step_evolve,np.zeros(4))
            q_k1,v_k1= simple_unstack(solution.x)
            solution_q.append(q_k1)
            solution_v.append(v_k1)
        return solution_q,solution_v
    
    
    def standard_calculate_running_cost(self,q_d,v_d,U_d):
        '''calculate the running cost ugu via consistent method. easiest just implicit euler
        (lets do implicit always)'''
        running_cost_total = np.zeros(self.parameters["N"])
        for i in range(len(running_cost_total)):
            x,y = ((q_d[i]+q_d[i+1])/2).flatten()
            U_i_mean = (U_d[i+1][0,0]+U_d[i][0,0])/2
            running_cost_total[i] = self.parameters['h'] * U_i_mean**2/2
        return running_cost_total

    def standard_create_running_termina_cost(self, U_d):
        '''create here the cost functions for the direct single shooting method. all depends on the control and this will be optimized
        running cost and terminal cost will be outputted'''
        sol_q, sol_v =  self.standard_calculate_q_v(U_d)
        running_cost = self.standard_calculate_running_cost(sol_q,sol_v,U_d)
        terminal_cost = self.standard_calculate_terminal_cost(sol_q,sol_v,U_d)
        return  np.concatenate([running_cost,[terminal_cost]]),sol_q,sol_v
    
    def standard_calculate_running_terminal_cost(self,q_d,v_d,U_d):
        running_cost = self.standard_calculate_running_cost(q_d,v_d,U_d)
        terminal_cost = self.standard_calculate_terminal_cost(q_d,v_d,U_d)
        return sum([running_cost.flatten(),terminal_cost])
