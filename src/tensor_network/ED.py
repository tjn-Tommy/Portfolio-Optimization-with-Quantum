import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg as LAs

# for exact diagonalization
def KronList(A_list):
    """Kronecker product of a list of matrices."""
    result = A_list[0]
    for A in A_list[1:]:
        result = np.kron(result, A)
    return result

def getHamiltonian(J, h):
    S0 = np.array([[1.,0.],[0.,1.]])
    Sz = np.array([[1.,0.],[0.,-1.]])

    Ns = J.shape[0]
    dim = 2**Ns
    H = np.zeros((dim,dim))
    for i in range(Ns):
        # local field term
        ops = [S0]*Ns
        ops[i] = h[i] * Sz
        H += KronList(ops)
        for j in range(Ns):
            if J[i,j] != 0:
                ops = [S0]*Ns
                ops[i] = J[i,j] * Sz
                ops[j] = Sz
                H += KronList(ops)
    # print("Hamiltonian matrix:\n", H)
    return H

def getGroundState(H):
    """Get ground state by exact diagonalization"""
    # use scipy for the first eigenvalue/eigenvector
    Eig, Evec = LAs.eigsh(H, k=1, which='SA')
    return Eig[0], Evec[:,0]

def getObservable(Ns,psi0,op):
    """Get expectation value of observable op at each site"""
    exp_values = []
    for i in range(Ns):
        ops = [np.eye(2)]*Ns
        ops[i] = op
        O_i = KronList(ops)
        exp_val = np.vdot(psi0, O_i @ psi0)
        exp_values.append(exp_val.real)
    return exp_values

def ExactDiagonalization(J, h):
    H = getHamiltonian(J, h)
    E0, psi0 = getGroundState(H)
    print(f"Exact diagonalization ground state energy: {E0}")
    # print("Ground state wavefunction:", psi0)
    max_idx = np.argmax(np.abs(psi0))
    # convert to binary representation
    state_bin = bin(max_idx)[2:].zfill(J.shape[0])
    state = [-int(bit)*2+1 for bit in state_bin]
    print("State:", state)
    return E0, state
