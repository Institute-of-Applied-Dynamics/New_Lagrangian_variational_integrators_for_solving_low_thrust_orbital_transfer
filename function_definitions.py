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

