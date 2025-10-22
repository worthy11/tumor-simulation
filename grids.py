import numpy as np
from config import *

time_step = 0

def set_initial_conditions():
    O2[:] = ch_O2
    G[:] = ch_g
    CO2[:] = ch_CO2
    ECM[:] = ch_ECM
    VEGF[:] = MMP[:] = P_INS[:] = P_LUM[:] = 0.
    U_INS[:] = U_BLOOD[:] = 0.

# -- MOLECULAR SCALE --
def update_o2():
    advection = diffusion = consumption = perfusion = np.zeros((ROWS, COLS), dtype=np.float64)

    Fx = U_INS[0] * O2
    Fy = U_INS[1] * O2

    dFx_dx = (np.roll(Fx, -1, axis=1) - np.roll(Fx, 1, axis=1))
    dFy_dy = (np.roll(Fy, -1, axis=0) - np.roll(Fy, 1, axis=0))

    divergence = dFx_dx + dFy_dy
    divergence[:, 0] = divergence[:, 1]
    divergence[:, -1] = divergence[:, -2]
    divergence[0, :] = divergence[1, :]
    divergence[-1, :] = divergence[-2, :]
    advection = divergence

    laplacian = (np.roll(O2, 1, axis=0) + np.roll(O2, -1, axis=0) + np.roll(O2, 1, axis=1) + np.roll(O2, -1, axis=1) - 4*O2) / DELTA**2
    laplacian[:, 0] = laplacian[:, 1]
    laplacian[:, -1] = laplacian[:, -2]
    laplacian[0, :] = laplacian[1, :]
    laplacian[-1, :] = laplacian[-2, :]
    diffusion = O2 * laplacian

    C_O2 = gamma_0 * V
    consumption = C_O2 * CELLS[1]

    # TODO: o co chodzi z d_v w tym równaniu? na razie d_0
    vessels = P_LUM > 0 # * CELLS[0]?
    perfusion = f_O2 * d_0/d_c * (P_LUM[vessels]-P_INS[vessels])/P_LUM[vessels]

    O2 += DT * (-advection + diffusion - consumption + perfusion)

def update_glucose():
    advection = diffusion = consumption = perfusion = np.zeros((ROWS, COLS), dtype=np.float64)

    Fx = U_INS[0] * G
    Fy = U_INS[1] * G

    dFx_dx = (np.roll(Fx, -1, axis=1) - np.roll(Fx, 1, axis=1))
    dFy_dy = (np.roll(Fy, -1, axis=0) - np.roll(Fy, 1, axis=0))

    divergence = dFx_dx + dFy_dy
    divergence[:, 0] = divergence[:, 1]
    divergence[:, -1] = divergence[:, -2]
    divergence[0, :] = divergence[1, :]
    divergence[-1, :] = divergence[-2, :]
    advection = divergence

    laplacian = (np.roll(G, 1, axis=0) + np.roll(G, -1, axis=0) + np.roll(G, 1, axis=1) + np.roll(G, -1, axis=1) - 4*G) / DELTA**2
    laplacian[:, 0] = laplacian[:, 1]
    laplacian[:, -1] = laplacian[:, -2]
    laplacian[0, :] = laplacian[1, :]
    laplacian[-1, :] = laplacian[-2, :]
    diffusion = G * laplacian

    C_g = 1/6 * gamma_0 * V
    consumption = C_g * CELLS[1]

    # TODO: o co chodzi z d_v w tym równaniu? na razie d_0
    vessels = P_LUM > 0 # * CELLS[0]?
    perfusion = (f_g * G / (G + km_g)) * d_0/d_c * (P_LUM[vessels]-P_INS[vessels])/P_LUM[vessels]

    G += DT * (-advection + diffusion - consumption + perfusion)

def update_co2():
    advection = diffusion = consumption = perfusion = np.zeros((ROWS, COLS), dtype=np.float64)

    Fx = U_INS[0] * CO2
    Fy = U_INS[1] * CO2

    dFx_dx = (np.roll(Fx, -1, axis=1) - np.roll(Fx, 1, axis=1))
    dFy_dy = (np.roll(Fy, -1, axis=0) - np.roll(Fy, 1, axis=0))

    divergence = dFx_dx + dFy_dy
    divergence[:, 0] = divergence[:, 1]
    divergence[:, -1] = divergence[:, -2]
    divergence[0, :] = divergence[1, :]
    divergence[-1, :] = divergence[-2, :]
    advection = divergence

    laplacian = (np.roll(CO2, 1, axis=0) + np.roll(CO2, -1, axis=0) + np.roll(CO2, 1, axis=1) + np.roll(CO2, -1, axis=1) - 4*CO2) / DELTA**2
    laplacian[:, 0] = laplacian[:, 1]
    laplacian[:, -1] = laplacian[:, -2]
    laplacian[0, :] = laplacian[1, :]
    laplacian[-1, :] = laplacian[-2, :]
    diffusion = CO2 * laplacian

    C_CO2 = -gamma_0 * V
    consumption = C_CO2 * CELLS[1]

    # TODO: o co chodzi z d_v w tym równaniu? na razie d_0
    vessels = P_LUM > 0 # * CELLS[0]?
    perfusion = f_CO2 * d_0/d_c * (P_LUM[vessels]-P_INS[vessels])/P_LUM[vessels]

    CO2 += DT * (-advection + diffusion - consumption - perfusion)

def update_ecm():
    pass

def update_mmp():
    pass

def update_vegf():
    pass

def update_vegfr2():
    pass

def update_treatment():
    pass

# -- CELLULAR SCALE --
def update_vitality():
    o2_component = phi * O2 / (O2 + ch_O2)
    glucose_component = G / (G + ch_g)
    co2_component = np.exp(-5 * (CO2 / ch_CO2 - 1)**4 * (CO2 - ch_CO2 > 0))

    V[:] = o2_component * glucose_component * co2_component

def update_energy():
    # TODO: drug_impact = k_ac * c_ac * V_saturation
    V_saturation = V / (V + 1)
    drug_impact = 0
    active_cells = ((kp_a * V - kc_a * V_saturation - drug_impact) * (V > v_ch))
    quiescent_cells = -kc_q * (V <= v_ch)

    E[:] = active_cells + quiescent_cells

def update_tumor_phenotypes():
    for i in range(ROWS):
        for j in range(COLS):
            vitality = V[i, j]
            energy = E[i, j]

            # active
            if CELLS[2, i, j]:
                # quiescence
                if vitality < v_ch:
                    CELLS[2, i, j] = 0
                    CELLS[3, i, j] = 1
                
                # division
                elif energy > psi_ch:
                    i_min, i_max = max(0, i-1), min(ROWS, i+2)
                    j_min, j_max = max(0, j-1), min(COLS, j+2)

                    block = V[i_min:i_max, j_min:j_max].copy()
                    block_center = (i - i_min, j - j_min)
                    neighbors = np.delete(block.flatten(), block_center[0] * (j_max - j_min) + block_center[1])
                    neighbor_sum = neighbors.sum()
                    weights = neighbors / neighbor_sum if neighbor_sum > 0 else np.full(8, 1/8)
                    choice = np.random.choice(len(neighbors), p=weights)
                    indices = [(i_offset, j_offset) for i_offset in range(i_min, i_max) 
                             for j_offset in range(j_min, j_max)
                             if (i_offset, j_offset) != (i, j)]

                    chosen_i, chosen_j = indices[choice]
                    CELLS[2, chosen_i, chosen_j] = 1
            
            # quiescent
            if CELLS[3, i, j]:
                # wake up
                if vitality > v_ch:
                    CELLS[2, i, j] = 1
                    CELLS[3, i, j] = 0
                # necrosis
                elif energy < 1e-30:
                    CELLS[5, i, j] = 1
                    CELLS[3, i, j] = 0

def update_endothelial_phenotypes():
    pass


# -- TISSUE SCALE -- 
def update_vessels():
    # 1. lumenogenesis
    # 2. vessel adaptation
    # 3. vessel deformation
    # 4. vessel disruption
    pass

def update_hemodynamics():
    # 1. P_LUM
    # 2. U_BLOOD
    pass

# TODO NEXT
def update_iff():
    # 1. P_INS
    # 2. U_INS
    pass

def update_hemorheology():
    pass

def update_tumor_growth():
    pass

def update_angiogenesis():
    pass


def update_molecular_scale():
    update_o2()
    update_glucose()
    update_co2()

    update_ecm()
    update_mmp()

    update_vegf()
    update_vegfr2()

    update_treatment()

def update_cellular_scale():
    update_vitality()
    update_energy()
    update_tumor_phenotypes()
    update_endothelial_phenotypes()

def update_tissue_scale():
    update_vessels()
    update_hemodynamics()
    update_iff()
    update_hemorheology()

    update_tumor_growth()
    update_angiogenesis()


def update_tme():
    update_molecular_scale()
    update_tissue_scale()
    update_cellular_scale() # is this order correct?