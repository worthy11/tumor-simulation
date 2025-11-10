import numpy as np
from config import *
from scipy.linalg import solve_banded

# Buffer for cells removed at entry points in the current step
REMOVED_ENTRY_CELLS = []  # list of (i, j)

def laplacian(Z, dx=DELTA, dy=DELTA):
    Z_top    = np.roll(Z,  1, axis=0)
    Z_bottom = np.roll(Z, -1, axis=0)
    Z_left   = np.roll(Z,  1, axis=1)
    Z_right  = np.roll(Z, -1, axis=1)
    lap = (Z_top + Z_bottom + Z_left + Z_right - 4*Z) / (dx*dy)
    
    lap[0, :] = lap[1, :]
    lap[-1, :] = lap[-2, :]
    lap[:, 0] = lap[:, 1]
    lap[:, -1] = lap[:, -2]
    return lap

def gradient(Z, dx=DELTA, dy=DELTA):
    gx = (np.roll(Z, -1, axis=0) - np.roll(Z, 1, axis=0)) / (2*dx)
    gy = (np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1)) / (2*dy)
    return gx, gy

def divergence(px, py, dx=DELTA, dy=DELTA):
    div_x = (np.roll(px, -1, axis=0) - np.roll(px, 1, axis=0)) / (2*dx)
    div_y = (np.roll(py, -1, axis=1) - np.roll(py, 1, axis=1)) / (2*dy)
    div = div_x + div_y
    
    div[0, :] = div[1, :]
    div[-1, :] = div[-2, :]
    div[:, 0] = div[:, 1]
    div[:, -1] = div[:, -2]
    return div

def set_initial_conditions():
    global O2, G, CO2, ECM, VEGF, MMP, P_INS, P_LUM, U_INS, U_BLOOD, RHO_EC, RHO_TC
    O2[:] = ch_O2
    G[:] = ch_g
    CO2[:] = ch_CO2

    ECM[:] = ch_ECM
    MMP[:] = 0.

    P_INS[:] = 1.
    P_LUM[:] = 60.
    U_INS[:] = 1e-8
    U_BLOOD[:] = 1e-2

    RHO_TC[:] = CELLS[1].astype(np.float64)

# -- MOLECULAR SCALE --
def update_o2():
    global O2, V, CELLS, P_LUM, P_INS, U_INS
    advection = np.zeros((ROWS, COLS), dtype=np.float64)
    diffusion = np.zeros((ROWS, COLS), dtype=np.float64)
    consumption = np.zeros((ROWS, COLS), dtype=np.float64)
    perfusion = np.zeros((ROWS, COLS), dtype=np.float64)

    Fx = U_INS[0] * O2
    Fy = U_INS[1] * O2
    advection = divergence(Fx, Fy) * (CELLS[0] == 0)

    diffusion = D_O2 * laplacian(O2)

    C_O2 = gamma_0 * V
    consumption = C_O2 * CELLS[1]

    vessels = (P_LUM > 0) * CELLS[0]
    perfusion[vessels] = f_O2 * d_0/d_c * (P_LUM[vessels]-P_INS[vessels])/P_LUM[vessels]

    O2[:] = O2 + DT * (advection + diffusion - consumption + perfusion)
    O2[O2 < 0] = 0

def update_glucose():
    global O2, G, CO2, V, CELLS, P_LUM, P_INS, U_INS
    advection = np.zeros((ROWS, COLS), dtype=np.float64)
    diffusion = np.zeros((ROWS, COLS), dtype=np.float64)
    consumption = np.zeros((ROWS, COLS), dtype=np.float64)
    perfusion = np.zeros((ROWS, COLS), dtype=np.float64)

    Fx = U_INS[0] * G
    Fy = U_INS[1] * G
    advection = divergence(Fx, Fy) * (CELLS[0] == 0)

    diffusion = D_g * laplacian(G)

    C_g = 1/6 * gamma_0 * V
    consumption = C_g * CELLS[1]

    perfusion[CELLS[0]] = ((f_g * G / (G + km_g)) * d_0/d_c)[CELLS[0]]
    perfusion[CELLS[0]] *= (P_LUM[CELLS[0]]-P_INS[CELLS[0]])/P_LUM[CELLS[0]]

    G = G[:] + DT * (-advection + diffusion - consumption + perfusion)
    G[G < 0] = 0

def update_co2():
    global O2, G, CO2, V, CELLS, P_LUM, P_INS, U_INS
    advection = np.zeros((ROWS, COLS), dtype=np.float64)
    diffusion = np.zeros((ROWS, COLS), dtype=np.float64)
    consumption = np.zeros((ROWS, COLS), dtype=np.float64)
    perfusion = np.zeros((ROWS, COLS), dtype=np.float64)

    Fx = U_INS[0] * CO2
    Fy = U_INS[1] * CO2
    advection = divergence(Fx, Fy) * (CELLS[0] == 0)

    diffusion = D_CO2 * laplacian(CO2)

    C_CO2 = -gamma_0 * V
    consumption = C_CO2 * CELLS[1]

    vessels = (P_LUM > 0) * CELLS[0]
    perfusion[vessels] = f_CO2 * d_0/d_c * (P_LUM[vessels]-P_INS[vessels])/P_LUM[vessels]

    CO2[:] = CO2 + DT * (-advection + diffusion - consumption - perfusion)
    CO2[CO2 < 0] = 0

def update_mmp():
    global MMP
    advection = np.zeros((ROWS, COLS), dtype=np.float64)
    diffusion = np.zeros((ROWS, COLS), dtype=np.float64)
    productionTC = np.zeros((ROWS, COLS), dtype=np.float64)
    productionEC = np.zeros((ROWS, COLS), dtype=np.float64)
    excretion = np.zeros((ROWS, COLS), dtype=np.float64)

    Fx = U_INS[0] * MMP
    Fy = U_INS[1] * MMP
    advection = divergence(Fx, Fy)

    diffusion = D_MMP * laplacian(MMP)

    productionTC = r_MMPTC * V * CELLS[1]

    productionEC = r_MMPEC * CELLS[0]

    excretion = eps_MMP * MMP

    MMP[:] = MMP + DT * (-advection + diffusion + productionTC + productionEC - excretion)
    MMP[MMP < 0] = 0

def update_ecm():
    global ECM
    excretion = np.zeros((ROWS, COLS), dtype=np.float64)

    excretion = eps_ECM * ECM * MMP

    ECM[:] = ECM - DT * excretion
    ECM[ECM < 0] = 0

# def update_vegf():
#     global VEGF
#     advection = np.zeros((ROWS, COLS), dtype=np.float64)
#     diffusion = np.zeros((ROWS, COLS), dtype=np.float64)
#     production = np.zeros((ROWS, COLS), dtype=np.float64)
#     perfusion = np.zeros((ROWS, COLS), dtype=np.float64)
#     binding = np.zeros((ROWS, COLS), dtype=np.float64)
#     unbinding = np.zeros((ROWS, COLS), dtype=np.float64)
#     excretion = np.zeros((ROWS, COLS), dtype=np.float64)

#     Fx = U_INS[0] * VEGF
#     Fy = U_INS[1] * VEGF
#     advection = divergence(Fx, Fy)

#     diffusion = D_VEGF * laplacian(VEGF)

#     production = (ch_CO2 > CO2) * (1 - CO2 / ch_CO2) * r_VEGF * CELLS[1]

#     perfusion = f_VEGF * d_0 / d_c * CELLS[0]

#     binding = k_plus_VEGF * VEGFR2[0] * VEGF

#     unbinding = k_minus_VEGF * VEGFR2[1]

#     excretion = eps_VEGF * VEGF

#     VEGF[:] = VEGF + DT * (-advection + diffusion + production - perfusion - binding + unbinding - excretion)
#     VEGF[VEGF < 0] = 0

# def update_vegfr2():
#     global VEGFR2
#     loss_free = np.zeros((ROWS, COLS), dtype=np.float64)
#     gain_free = np.zeros((ROWS, COLS), dtype=np.float64)
#     production_active = np.zeros((ROWS, COLS), dtype=np.float64)
#     loss_active = np.zeros((ROWS, COLS), dtype=np.float64)

#     loss_free = k_plus_VEGF * VEGFR2[0] * VEGF
#     gain_free = k_minus_VEGF * VEGFR2[1]

#     production_active = k_plus_VEGF * VEGFR2[0] * VEGF
#     loss_active = k_minus_VEGF * VEGFR2[1]

#     VEGFR2[0][:] = VEGFR2[0] + DT * (-loss_free + gain_free)
#     VEGFR2[1][:] = VEGFR2[1] + DT * (production_active - loss_active)
    
#     VEGFR2[0][VEGFR2[0] < 0] = 0
#     VEGFR2[1][VEGFR2[1] < 0] = 0


# -- CELLULAR SCALE --
def update_vitality():
    global O2, G, CO2, V
    o2_component = phi * O2 / (O2 + ch_O2)
    glucose_component = G / (G + ch_g)
    co2_component = np.exp(-5 * (CO2 / ch_CO2 - 1)**4 * (CO2 - ch_CO2 > 0))

    V[:] = np.abs(o2_component * glucose_component * co2_component)

def update_energy():
    global V, E
    V_saturation = V / (V + 1)
    active_cells = ((kp_a * V - kc_a * V_saturation) * CELLS[1])
    quiescent_cells = -kc_q * CELLS[2]

    E[:] = E + DT * (active_cells + quiescent_cells)

def update_tumor_phenotypes():
    global V, E, CELLS
    for i in range(ROWS):
        for j in range(COLS):
            vitality = V[i, j]
            energy = E[i, j]

            # active
            if CELLS[1, i, j]:
                # quiescence
                if vitality < v_ch:
                    CELLS[1, i, j] = 0
                    CELLS[2, i, j] = 1
                
                # division
                elif energy > psi_ch:
                    i_min, i_max = max(0, i - 1), min(ROWS - 1, i + 1)
                    j_min, j_max = max(0, j - 1), min(COLS - 1, j + 1)

                    indices = [(ii, jj)
                            for ii in range(i_min, i_max + 1)
                            for jj in range(j_min, j_max + 1)
                            if not (ii == i and jj == j)]

                    can_divide = not all([CELLS[1, ii, jj] for (ii, jj) in indices])
                    if not can_divide:
                        continue
                    
                    densities = np.array([RHO_TC[ii, jj] for (ii, jj) in indices])
                    densities_inv = 1 / np.where(densities != 0, densities, np.finfo(float).eps)
                    probs = densities_inv / densities_inv.sum()
                    choice = np.random.choice(len(indices), p=probs)
                    chosen_i, chosen_j = indices[choice]

                    CELLS[1, chosen_i, chosen_j] = 1

            # quiescent
            if CELLS[2, i, j]:
                # wake up
                if vitality > v_ch:
                    CELLS[1, i, j] = 1
                    CELLS[2, i, j] = 0
                # necrosis
                elif energy < 1e-30:
                    CELLS[4, i, j] = 1
                    CELLS[2, i, j] = 0

# -- TISSUE SCALE -- 
def update_tumor_growth():
    global RHO_TC, MMP
    diff_term = D_TC * laplacian(RHO_TC)
    
    grad_ce_x, grad_ce_y = gradient(V)
    
    haptotaxis_x = beta_h * grad_ce_x
    haptotaxis_y = beta_h * grad_ce_y
    
    taxis_div = divergence(haptotaxis_x, haptotaxis_y)

    RHO_TC[:] = RHO_TC + DT * (diff_term - taxis_div)
    RHO_TC[RHO_TC < 0] = 0

# Migration update functions
def vessel_entry():
    global CELLS, RHO_TC, REMOVED_ENTRY_CELLS
    REMOVED_ENTRY_CELLS = []
    overlap = CELLS[1] & CELLS[5]
    if not overlap.any():
        return

    coords = np.argwhere(overlap)
    for (i, j) in coords:
        if np.random.random() < 0.7:
            neighbors = [(i, j), (i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            for ii, jj in neighbors:
                if 0 <= ii < ROWS and 0 <= jj < COLS:
                    CELLS[1, ii, jj] = False
                    RHO_TC[ii, jj] = 0
            REMOVED_ENTRY_CELLS.append((i, j))


def tissue_entry():
    global CELLS, RHO_TC, REMOVED_ENTRY_CELLS
    if not REMOVED_ENTRY_CELLS:
        return

    entry_points = np.argwhere(CELLS[5])
    if entry_points.size == 0:
        REMOVED_ENTRY_CELLS = []
        return

    for (i, j) in REMOVED_ENTRY_CELLS:
        if np.random.random() < 0.4:
            candidates = [(ii, jj) for (ii, jj) in entry_points
                          if (ii != i or jj != j) and not CELLS[1, ii, jj]]
            if not candidates:
                candidates = [(ii, jj) for (ii, jj) in entry_points if (ii != i or jj != j)]

            if candidates:
                ni, nj = candidates[np.random.randint(len(candidates))]
                CELLS[1, ni, nj] = True
                RHO_TC[ni, nj] = 1.0

    REMOVED_ENTRY_CELLS = []

def update_molecular_scale():
    update_o2()
    update_glucose()
    update_co2()
    update_ecm()
    update_mmp()

def update_cellular_scale():
    update_vitality()
    update_energy()
    update_tumor_phenotypes()

def update_tissue_scale():
    update_tumor_growth()
    vessel_entry()
    tissue_entry()

def update_tme():
    update_molecular_scale()
    update_tissue_scale()
    update_cellular_scale()