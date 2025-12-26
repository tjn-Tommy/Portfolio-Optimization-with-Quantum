import numpy as np
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS
from tenpy.models.lattice import TrivialLattice
from tenpy.models.model import CouplingMPOModel
from tenpy.algorithms.exact_diag import get_numpy_Hamiltonian
from tenpy.algorithms.dmrg import SingleSiteDMRGEngine, TwoSiteDMRGEngine
import warnings
warnings.filterwarnings(
    "ignore",
    message="unit_cell_width is a new argument for MPS"
)

class MyModel(CouplingMPOModel):
    def __init__(self, model_params):
        super().__init__(model_params)

    def init_lattice(self, model_params):
        L = model_params.get('L', None)
        return TrivialLattice([SpinHalfSite(None)] * L)
    
    ##  定义系统的作用量，这一部分需要根据需求自己实现，model_params控制可变参数
    def init_terms(self, model_params):
        J = model_params.get('J', None)
        h = model_params.get('h', None)
        L = model_params.get('L', None)
        for i in range(L):
            # local field term
            self.add_onsite_term(2*h[i], i, 'Sz') # s_z = 1/2 sigma_z
            for j in range(L):
                if J[i,j] != 0:
                    self.add_coupling_term(4*J[i,j], i, j, 'Sz', 'Sz') # s_z s_z = 1/4 sigma_z sigma_z
         
def tenpy_dmrg(J,h):
    L = J.shape[0]
    # psi = MPS.from_product_state([site] * L, ["up"] * L)
    model_params = {
        "L": L,
        "J": J,
        "h": h
    }
    dmrg_params = {
        "mixer": True,
        'max_E_err': 1.e-10,
        "max_N_sites_per_ring": 50,
        "mixer_params": {"amplitude": 1e-2, "decay": 1.5, "disable_after": 10},
        "trunc_params": {"chi_max": 50, "svd_min": 1e-10},
        "max_sweeps": 10,
    }
    model = MyModel(model_params)
    psi = MPS.from_random_unitary_evolution(sites=model.lat.mps_sites(), chi=50, p_state=["up"] * L)

    # result = dmrg(psi, model, dmrg_params)
    engine = SingleSiteDMRGEngine(psi, model, dmrg_params)
    E, psi = engine.run()
    state = np.round(psi.expectation_value('Sz') * 2).astype(int).tolist()  # convert to -1,1
    # print("Ground state energy =", E)
    # print("Ground state Sz values =", state)
    return state

if __name__ == "__main__":
    # L = 6                       # system size
    # np.random.seed(1)
    # J = np.random.randn(L, L)    # full random couplings
    # J = (J + J.T) / 2            # symmetrize
    # h = np.random.randn(L)       # random fields

    L = 2
    J = np.array([[0., 1.0],
                  [0., 0.]])
    h = np.array([0.5, 0.5])

    # Only use i<j terms
    for i in range(L):
        for j in range(L):
            if i >= j:
                J[i, j] = 0.0

    model = MyModel({'L': L, 'J': J, 'h': h})
    # print(model.H_MPO)
    H_numpy = get_numpy_Hamiltonian(model)
    print("Hamiltonian matrix:\n", H_numpy)
    # get the exact ground state energy for comparison
    Eig, Evec = np.linalg.eigh(H_numpy)
    print(f"Exact diagonalization ground state energy: {Eig[0]}, {Evec[:,0]}")
