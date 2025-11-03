import sympy

############################################################################
################### Low thrust functions
############################################################################
#Cartesian functions
def f_vec_cart(q,v,params):
    g = params["G"]
    M = params["M"]
    x,y = q
    r = sympy.sqrt(x**2+y**2)
    return -sympy.Matrix([x,y])*g*M/r**3

def rho_vec_cart(q,v,params):
    x,y=q
    r= sympy.sqrt(x**2+y**2)
    return sympy.Matrix([-y,x])/r

def g_mat_cart(q,params):
    return sympy.Matrix([1])

def mayer_term_cart(q,v,params):
    x,y = q
    vx,vy = v
    xT,yT = params["qT"]
    vxT,vyT = params["dqT"]
    phi = params["Aq"]*((x-xT)**2 + (y-yT)**2)
    phi+= params["Adq"]*((vx-vxT)**2 + (vy-vyT)**2)
    return sympy.Matrix([phi])

def running_cost_cart(q,u,params):
    return u.transpose()@u/2


def f_vec_polar(q,v,params):
    r,phi = q
    vr,vphi = v
    f_vec = sympy.Matrix([r * vphi**2 - params["G"]*params["M"] /r**2, - 2* vr*vphi/r])
    return f_vec

def rho_vec_polar(q,v,params):
    r,phi = q
    vr,vphi = v
    rho_vec = sympy.Matrix([0,1/r])
    return rho_vec

def g_mat_polar(q,params):
    return sympy.Matrix([1])

def mayer_term_polar(q,vq,params): #polar coordinates
    # return params["Aq"]*q.transpose()@q + params["Adq"]*vq.transpose()@vq #define terminal cost
    r,phi = q
    vr,vphi = vq
    rT,phiT = params["qT"]
    drT, dphiT = params["dqT"]
    q_term = params["Aq"]* ((r-rT)**2 + 2*r*rT*(1+sympy.cos(phi) ) ) # specific choice of (xT,yT) = (-r,0)
    dq_term = params["Adq"]* (2*rT*dphiT*vr*sympy.sin(phi)+( r*vphi -  rT*dphiT)**2 + 2*r*rT*vphi*dphiT*(1+sympy.cos(phi))+vr**2)
    return sympy.Matrix([q_term+ dq_term])

def running_cost_polar(q,u,params):
    return u.transpose()@u/2


############################################################################
################### Pendulum on a cart functions
############################################################################



# def pendulum_cart_lagrangian(q,vq,params):
#     m1,m2 = params["m1"],params["m2"]
#     l,g,I_param = params["l"],params["G"],params["I"]
#     x,theta   = q
#     vx,vtheta = vq
#     Lfunc = (m1+m2)*vx**2/2 + l*m2*vx*vtheta*sympy.cos(theta)/2 + l**2*m2*vtheta**2/8 + I_param*vtheta**2/2 + m2*g*l*sympy.cos(theta)/2
#     return Lfunc

# def f_L(q,vq,u,params):
#     uvec,=u 
#     return sympy.Matrix([uvec,0])

# def running_cost_pendulum(q,u,params):
#     A_u = params["A_u"]
#     return A_u *u.transpose()@u/2

# def mayer_term_pendulum(q,v,params):
#     x,phi = q
#     vx,vphi = v
#     xT,phiT = params["qT"]
#     vxT,vphiT = params["dqT"]
#     Phi = params["Aq"]*((x-xT)**2 + (phi-phiT)**2)
#     Phi+= params["Adq"]*((vx-vxT)**2 + (vphi-vphiT)**2)
#     return sympy.Matrix([Phi])

# def conserved_I_control_pendulum(control_L, q,lam,vq,vlam,u,params):
#     '''give here the conserved quantity for the control Lagrangian
#     for pendulum on a cart that is p_x = dL_c/dv_x'''
#     I_ocp = sympy.derive_by_array(control_L,vq[0])
#     return I_ocp

# def g_mat_pendulum(q,params):
#     return sympy.Matrix([1])


