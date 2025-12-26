import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg as LAs
from tensor_network import Sub180221 as Sub
# import Sub180221 as Sub
import copy

def GetApproxMPO(J, h, R):
    assert R <= min(J.shape), "R must be less than or equal to min(L,N)"
    # Low-rank decomposition of J
    U, s, Vh = np.linalg.svd(J, full_matrices=False)
    s = s[:R]
    U = U[:, :R]
    V = Vh[:R, :].T              # R x L

    # Build MPO tensors W[p]
    L = J.shape[0]
    D = R + 2

    W = []
    for p in range(L):
        # physical ops
        Id = np.eye(2)
        Sz = np.array([[1.,0.],[0.,-1.]])

        # W[p] has shape (D_left, D_right, d, d)
        Wp = np.zeros((D, 2, D, 2), dtype=float)

        # diagonal flow: carries no interaction
        for i in range(D):
            Wp[i,:,i,:] = Id
        # R "channels"
        for r in range(R):
            Wp[0,:,1+r,:] = U[p, r] * Sz        # output Sz for channel r
            Wp[1+r,:,D-1,:] = s[r] * V[p, r] * Sz  # channel r ending
        # local field term h_i S_i^z
        Wp[0,:,D-1,:] += h[p] * Sz
        
        W.append(Wp)
    return W

def InitMps(Ns,Dp,Ds,seed=None):
    if seed is not None:
        np.random.seed(seed)
    T = [None]*Ns
    for i in range(Ns):
        Dl = min(Dp**i,Dp**(Ns-i),Ds)
        Dr = min(Dp**(i+1),Dp**(Ns-1-i),Ds)
        T[i] = np.random.rand(Dl,Dp,Dr)
    
    U = np.eye(np.shape(T[-1])[-1])
    for i in range(Ns-1,0,-1):
        U,T[i] = Sub.Mps_LQP(T[i],U)
    
    return T

def InitH(Mpo_list,T):
    Ns = len(T)
    Dmpo = np.shape(Mpo_list[0])[0]
    
    HL = [None]*Ns
    HR = [None]*Ns
    
    HL[0] = np.zeros((1,Dmpo,1))
    HL[0][0,0,0] = 1.0
    HR[-1] = np.zeros((1,Dmpo,1))
    HR[-1][0,-1,0] = 1.0
    
    for i in range(Ns-1,0,-1):
        Mpo = Mpo_list[i]
        HR[i-1] = Sub.NCon([HR[i],T[i],Mpo,np.conj(T[i])],[[1,3,5],[-1,2,1],[-2,2,3,4],[-3,4,5]])
    
    return HL,HR

def OptTSite1(Mpo,HL,HR,T):
    DT = np.shape(T)
    Dl = np.prod(DT)

    A = Sub.NCon([HL,Mpo,HR],[[-1,1,-4],[1,-5,2,-2],[-6,2,-3]])
    A = Sub.Group(A,[[0,1,2],[3,4,5]])
    A += 1e-8 * np.eye(A.shape[0])  # regularization to avoid singular matrix
    Eig, V = LAs.eigsh(-A, k=1, which='LA')
    Eig = -Eig
    T = np.reshape(V,DT)
    # print('Eig',Eig)
        
    return T,Eig

def OptT1(Mpo_list,HL,HR,T):
    Ns = len(T)
    Eng0 = np.zeros(Ns)
    Eng1 = np.zeros(Ns)
    
    converge = False
    for r in range(100):
        # print(r)
    
        for i in range(Ns-1):
            Mpo = Mpo_list[i]
            T[i],Eng1[i] = OptTSite1(Mpo,HL[i],HR[i],T[i])
            # print(i,Eng1[i])
            T[i],U = Sub.Mps_QR0P(T[i])
            HL[i+1] = Sub.NCon([HL[i],np.conj(T[i]),Mpo,T[i]],[[1,3,5],[1,2,-1],[3,4,-2,2],[5,4,-3]])
            T[i+1] = np.tensordot(U,T[i+1],(1,0))
            # print("1",T[i+1])
        
        for i in range(Ns-1,0,-1):
            Mpo = Mpo_list[i]
            T[i],Eng1[i] = OptTSite1(Mpo,HL[i],HR[i],T[i])
            # print(i,Eng1[i])
            U,T[i] = Sub.Mps_LQ0P(T[i])
            HR[i-1] = Sub.NCon([HR[i],T[i],Mpo,np.conj(T[i])],[[1,3,5],[-1,2,1],[-2,2,3,4],[-3,4,5]])
            T[i-1] = np.tensordot(T[i-1],U,(2,0))
        
        # print(Eng1)
        if abs(Eng1[1]-Eng0[1]) < 1.0e-10:
            # print(f'Converged after {r} sweeps.')
            converge = True
            break
        Eng0 = copy.copy(Eng1)
    
    # print("Energy per site:", Eng1 / float(Ns))
    # print("Ground Energy:", np.mean(Eng1))
    
    return T, converge

def OptSite2(Mpo,HL,HR,T1,T2):
    DT = np.shape(T1)
    Dl = np.prod(DT)
    DT2 = np.shape(T2)
    Dr = np.prod(DT2)

    # Combine two tensors
    T_combined = np.tensordot(T1, T2, axes=(2, 0))
    DT_combined = np.shape(T_combined)
    Dl_combined = DT_combined[0]
    Dp1 = DT_combined[1]
    Dp2 = DT_combined[2]
    Dr_combined = DT_combined[3]
    T_combined = np.reshape(T_combined, (Dl_combined * Dp1, Dp2 * Dr_combined))

    A = Sub.NCon([HL, Mpo, Mpo, HR],
                 [[-1, 1, -5], [1, -6, 2, -2], [2, -7, 3, -3], [-8, 3, -4]])
    A = Sub.Group(A, [[0, 1, 2, 3], [4, 5, 6, 7]])

    A += 1e-8 * np.eye(A.shape[0])  # regularization to avoid singular matrix
    Eig, V = LAs.eigsh(-A, k=1, which='LA')
    Eig = -Eig
    T_combined = np.reshape(V, DT_combined)

    # Split back into two tensors, with truncation
    T1_new, S, T2_new = Sub.Mps_SVD0P(T_combined, chi_max=DT[-1])
    # if no truncation, same as exact diagonalization
    # T1_new, S, T2_new = Sub.Mps_SVD0P(T_combined)
    return T1_new, S, T2_new, Eig

def OptT2(Mpo_list, HL, HR, T):
    """
    Perform 2-site variational optimization for the MPS.

    Parameters:
        Mpo: The MPO representation of the Hamiltonian.
        HL: Left environment tensors.
        HR: Right environment tensors.
        T: The MPS tensors.

    Returns:
        T: Updated MPS tensors after 2-site optimization.
    """
    Ns = len(T)
    Eng0 = np.zeros(Ns)
    Eng1 = np.zeros(Ns)

    converge = False
    for r in range(100):
        # print(r)
        for i in range(Ns - 1):
            Mpo = Mpo_list[i]
            T[i], S, T2_new, Eng1[i] = OptSite2(Mpo, HL[i], HR[i + 1], T[i], T[i + 1])
            T[i+1] = np.tensordot(S, T2_new, axes=(1, 0))
            # Update the environments
            HL[i + 1] = Sub.NCon([HL[i], np.conj(T[i]), Mpo, T[i]],
                                 [[1, 3, 5], [1, 2, -1], [3, 4, -2, 2], [5, 4, -3]])
            
        for i in range(Ns - 1, 0, -1):
            Mpo = Mpo_list[i]
            T1_new, S, T[i], Eng1[i] = OptSite2(Mpo, HL[i-1], HR[i], T[i-1], T[i])
            T[i-1] = np.tensordot(T1_new, S, axes=(2, 0))
            # Update the environments
            HR[i-1] = Sub.NCon([HR[i], T[i], Mpo, np.conj(T[i])],
                              [[1, 3, 5], [-1, 2, 1], [-2, 2, 3, 4], [-3, 4, 5]])

        # print(Eng1)
        # Check convergence
        if abs(Eng1[0] - Eng0[0]) < 1.0e-10:
            print(f'Converged after {r} sweeps.')
            converge = True
            break
        Eng0 = copy.copy(Eng1)

    print("Energy per site:", Eng1 / float(Ns))
    print("Energy average:", np.mean(Eng1) / float(Ns))

    return T, converge

def ExpectationValue(op, pos, Ns, T):
    '''
    Expectation value of a single-site operator 'op' at site 'pos' in an MPS of length Ns.
    '''
    H = [None]*Ns
    Dmpo = op.shape[0]
    for i in range(Ns):
        H[i] = np.zeros((Dmpo,2,Dmpo,2))
        if i == pos:
            H[i][0,:,1,:] = op
        else:
            H[i][0,:,0,:] = np.eye(Dmpo)
            H[i][1,:,1,:] = np.eye(Dmpo)
    L = np.zeros((1,Dmpo,1))
    L[0,0,0] = 1.0
    R = np.zeros((1,Dmpo,1))
    R[0,-1,0] = 1.0

    assert len(T) == Ns
    for i in range(Ns):
        L = Sub.NCon([L,T[i],H[i],np.conj(T[i])],[[1,3,5],[1,2,-1],[3,2,-2,4],[5,4,-3]])
    # contract L and R
    ans = Sub.NCon([L,R],[[1,2,3],[1,2,3]])
    # print(L)
    return ans

def VariationalMPS(J, h, R=10, Dp=2, Ds=6, opt=1, seed=None, max_trial=100):
    Ns = J.shape[0]
    assert len(h) == J.shape[1] == Ns

    converged = False
    for trial in range(max_trial):
        # print(f'Trial {trial+1}/{max_trial}')
        Mpo_list = GetApproxMPO(J, h, R)
        T = InitMps(Ns, Dp, Ds, seed=seed)
        HL, HR = InitH(Mpo_list, T)
        if opt == 1:
            T, converged = OptT1(Mpo_list, HL, HR, T)
        elif opt == 2:
            T, converged = OptT2(Mpo_list, HL, HR, T)
        if converged:
            break
        if seed is not None:
            seed += 1  # change seed for next trial

    state = []
    for i in range(Ns):
        state.append(ExpectationValue(np.array([[1.,0.],[0.,-1.]]), i, Ns, T))
    state_int = [int(np.round(s)) for s in state]
    # print("State:", state_int)
    
    return T, state_int
