import numpy as np

N = 2

# Energy in embedding space
def E(x):
    return x[0]**3 + x[1]**2

# Gradient in embedding space
def G(x):
    result = np.zeros(len(x))
    result[0] = 3*x[0]
    result[1] = -x[1]
    return result

# Implements dx_i / dphi_j
def dx_dphi(x, i, phi, j):
    if i==j:
        return x[i] * -np.tan(phi[j])
    if i>j:
        return x[i] * 1/np.tan(phi[j])
    if i<j:
        return 0

def G_phi(G, x, phi):
    dim     = len(G)
    G_phi   = np.zeros(dim)
    tan_phi = np.tan(phi)

    for j in range(dim):
        for i in range(dim):
            G_phi[j] += G[i] * dx_dphi(x, i, phi, j)

    return G_phi

def x_to_phi(x):
    dim = len(x)
    phi = np.zeros( dim)

    phi[0] = np.arccos( x[0] )
    phi[1] = np.arcsin( x[1] / np.sin(phi[0]) )

    return phi

def phi_to_x(phi):
    dim = len(phi)
    x = np.cos(phi)

    # Looks something like this:
    # x[0] = np.cos( phi[0] )
    # x[1] = np.sin( sin[0] ) * np.cos( phi[1] )
    # x[2] = np.sin( sin[0] ) * np.sin( phi[1] ) * np.cos( phi[2] )
    # x[3] = np.sin( sin[0] ) * np.sin( phi[1] ) * np.sin( phi[1] ) * np.cos( phi[3] ) # NOTE: per definition we always set phi[dim-1]=0

    mult = 1
    for i in range(dim):
        x[i] *= mult
        mult *= np.sin( phi[i] )
    return x

def x_to_phi(x):
    dim = len(x)
    phi = np.zeros(dim)

    # Looks something like this:
    # phi[0] = np.arccos( x[0] )
    # phi[1] = np.arccos( x[1] /   np.sin( phi[0] ) )
    # phi[2] = np.arccos( x[2] / ( np.sin( phi[0] ) * np.sin( phi[1] ) ) )
    # phi[3] = np.arccos( x[3] / ( np.sin( phi[0] ) * np.sin( phi[1] ) * np.sin( phi[2] ) ) )

    div = 1
    for i in range(dim):
        phi[i] = np.arccos( x[i] / div )
        div *= np.sin(phi[i])
    phi[dim-1] = 0
    return phi

x   = np.ones(10)
phi = x_to_phi(x)
x2  = phi_to_x(x)

# phi  = np.array([0.5, 0.5, 0.5, 0])
# phi  = np.ones(120)
# x    = phi_to_x(phi)
# phi2 = x_to_phi(x)

# print("x ", x)
# print("np.linalg.norm(x) ", np.linalg.norm(x))
# print("x2 ", x2)
# print("np.linalg.norm(x2) ", np.linalg.norm(x2))
# print("phi ", phi)
# print("phi2 ", phi2)
# print("G(x) ", G(x))
# print("G_phi(G(x), x, phi) ", G_phi(G(x), x, phi))