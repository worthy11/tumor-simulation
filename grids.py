import numpy as np
from config import *
from scipy.linalg import solve_banded

def laplacian(Z, dx=1.0, dy=1.0):
    """Dyskretna Laplasja w 2D z warunkami brzegowymi zerowymi."""
    Z_top    = np.roll(Z,  1, axis=0)
    Z_bottom = np.roll(Z, -1, axis=0)
    Z_left   = np.roll(Z,  1, axis=1)
    Z_right  = np.roll(Z, -1, axis=1)
    lap = (Z_top + Z_bottom + Z_left + Z_right - 4*Z) / (dx*dy)
    
    lap[0,:] = lap[-1,:] = lap[:,0] = lap[:,-1] = 0
    return lap

def gradient(Z, dx=1.0, dy=1.0):
    """Oblicza gradient 2D w osiach x i y."""
    gx = (np.roll(Z, -1, axis=0) - np.roll(Z, 1, axis=0)) / (2*dx)
    gy = (np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1)) / (2*dy)
    return gx, gy

def divergence(px, py, dx=1.0, dy=1.0):
    """Oblicza dywergencję wektora 2D (px, py)."""
    div_x = (np.roll(px, -1, axis=0) - np.roll(px, 1, axis=0)) / (2*dx)
    div_y = (np.roll(py, -1, axis=1) - np.roll(py, 1, axis=1)) / (2*dy)
    div = div_x + div_y
    div[0,:] = div[-1,:] = div[:,0] = div[:,-1] = 0
    return div

def set_initial_conditions():
    global O2, G, CO2, ECM, VEGF, MMP, P_INS, P_LUM, U_INS, U_BLOOD
    O2[:] = ch_O2
    G[:] = ch_g
    CO2[:] = ch_CO2

    ECM[:] = ch_ECM
    VEGF[:] = 0.
    MMP[:] = 0.

    P_INS[:] = 0.
    P_LUM[:] = p_0
    U_INS[:] = 0.
    U_BLOOD[:] = 0.

# -- MOLECULAR SCALE --
def update_o2():
    global O2, V, CELLS, P_LUM, P_INS, U_INS
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
    diffusion = D_O2 * laplacian

    C_O2 = gamma_0 * V
    consumption = C_O2 * CELLS[1]

    vessels = P_LUM > 0 # * CELLS[0]?
    perfusion[vessels] = f_O2 * d_0/d_c * (P_LUM[vessels]-P_INS[vessels])/P_LUM[vessels]
    
    O2[:] = O2 + DT * (-advection + diffusion - consumption + perfusion)
    O2[O2 < 0] = 0

def update_glucose():
    global O2, G, CO2, V, CELLS, P_LUM, P_INS, U_INS
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
    diffusion = D_g * laplacian

    C_g = 1/6 * gamma_0 * V
    consumption = C_g * CELLS[1]

    # TODO: o co chodzi z d_v w tym równaniu? na razie d_0
    vessels = P_LUM > 0 # * CELLS[0]?
    perfusion[vessels] = ((f_g * G / (G + km_g)) * d_0/d_c)[vessels]
    perfusion[vessels] *= (P_LUM[vessels]-P_INS[vessels])/P_LUM[vessels]

    G = G[:] + DT * (-advection + diffusion - consumption + perfusion)
    G[G < 0] = 0

def update_co2():
    global O2, G, CO2, V, CELLS, P_LUM, P_INS, U_INS
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
    diffusion = D_CO2 * laplacian

    C_CO2 = -gamma_0 * V
    consumption = C_CO2 * CELLS[1]

    # TODO: o co chodzi z d_v w tym równaniu? na razie d_0
    vessels = P_LUM > 0 # * CELLS[0]?
    perfusion[vessels] = f_CO2 * d_0/d_c * (P_LUM[vessels]-P_INS[vessels])/P_LUM[vessels]

    CO2[:] = CO2 + DT * (-advection + diffusion - consumption - perfusion)
    CO2[CO2 < 0] = 0

def update_mmp():
    global MMP
    advection = diffusion = productionTC = productionEC = excretion = np.zeros((ROWS, COLS), dtype=np.float64)

    Fx = U_INS[0] * MMP
    Fy = U_INS[1] * MMP

    dFx_dx = (np.roll(Fx, -1, axis=1) - np.roll(Fx, 1, axis=1))
    dFy_dy = (np.roll(Fy, -1, axis=0) - np.roll(Fy, 1, axis=0))

    divergence = dFx_dx + dFy_dy
    divergence[:, 0] = divergence[:, 1]
    divergence[:, -1] = divergence[:, -2]
    divergence[0, :] = divergence[1, :]
    divergence[-1, :] = divergence[-2, :]
    advection = divergence

    laplacian = (np.roll(MMP, 1, axis=0) + np.roll(MMP, -1, axis=0) + np.roll(MMP, 1, axis=1) + np.roll(MMP, -1, axis=1) - 4*MMP) / DELTA**2
    laplacian[:, 0] = laplacian[:, 1]
    laplacian[:, -1] = laplacian[:, -2]
    laplacian[0, :] = laplacian[1, :]
    laplacian[-1, :] = laplacian[-2, :]
    diffusion = D_MMP * laplacian

    productionTC = r_MMPTC * V * CELLS[1]

    productionEC = r_MMPEC * CELLS[0]

    excretion = eps_MMP * MMP

    MMP[:] = MMP + DT * (-advection + diffusion + productionTC + productionEC - excretion)
    MMP[MMP < 0] = 0

def update_ecm():
    global ECM
    excretion = np.zeros((ROWS, COLS), dtype=np.float64)

    excretion = eps_ECM * MMP * ECM

    ECM[:] = ECM - DT * excretion
    ECM[ECM < 0] = 0

def update_vegf():
    pass

def update_vegfr2():
    pass  # Not implemented yet

def update_treatment():
    pass

# -- CELLULAR SCALE --
def update_vitality():
    global O2, G, CO2, V
    o2_component = phi * O2 / (O2 + ch_O2)
    glucose_component = G / (G + ch_g)
    co2_component = np.exp(-5 * (CO2 / ch_CO2 - 1)**4 * (CO2 - ch_CO2 > 0))

    V[:] = o2_component * glucose_component * co2_component

def update_energy():
    global V, E
    # TODO: drug_impact = k_ac * c_ac * V_saturation
    V_saturation = V / (V + 1)
    drug_impact = 0
    active_cells = ((kp_a * V - kc_a * V_saturation - drug_impact) * CELLS[1])
    quiescent_cells = -kc_q * CELLS[2]

    E[:] = active_cells + quiescent_cells

def update_tumor_phenotypes():
    global V, E, CELLS
    for i in range(ROWS):
        for j in range(COLS):
            vitality = V[i, j]
            energy = E[i, j]
            print(f"{i} {j}: {vitality} {energy}")

            # active
            if CELLS[1, i, j]:
                # quiescence
                if vitality < v_ch:
                    CELLS[1, i, j] = 0
                    CELLS[2, i, j] = 1
                
                # division
                elif energy > psi_ch:
                    i_min, i_max = max(0, i-1), min(ROWS, i+2)
                    j_min, j_max = max(0, j-1), min(COLS, j+2)

                    block = V[i_min:i_max, j_min:j_max].copy()
                    block_center = (i - i_min, j - j_min)
                    neighbors = np.delete(block.flatten(), block_center[0] * (j_max - j_min) + block_center[1])
                    choice = np.argmin(neighbors)
                    indices = [(i_offset, j_offset) for i_offset in range(i_min, i_max) 
                             for j_offset in range(j_min, j_max)
                             if (i_offset, j_offset) != (i, j)]

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

def update_endothelial_phenotypes():
    pass

# -- TISSUE SCALE -- 
def update_vessels():
    # 1. lumenogenesis
    # 2. vessel adaptation
    # 3. vessel deformation
    # 4. vessel disruption
    pass

def update_hemodynamics(alpha=0.7, tol=1e-6, max_iter=1000):
    global P_LUM, U_BLOOD, CELLS
    # TODO: L = DELTA bo liczymy po kawalku
    # TODO: Add mu_blood to params
    G = np.pi * d_0**4 / (128 * DELTA * update_hemorheology())
    S_EC = np.pi * d_0 * DELTA * L_pEC * CELLS[0]
    S_TC = np.pi * d_0 * DELTA * L_pTC * CELLS[1]
    S = S_EC + S_TC

    P_LUM_new = P_LUM.copy()

    for iteration in range(max_iter):
        Q_TFF_EC = S * ((P_LUM - (pi_lum - pi_insEC) * delta_EC)) * CELLS[0]
        Q_TFF_TC = S * ((P_LUM - (pi_lum - pi_insTC) * delta_TC)) * CELLS[1]
        Q_TFF = Q_TFF_EC + Q_TFF_TC

        for j in range(COLS):
            bvec = Q_TFF[:, j].copy()      # 1D vector per column

            a = -G * np.ones(ROWS-1)
            b = 2*G * np.ones(ROWS)
            c = -G * np.ones(ROWS-1)

            # simple Dirichlet BC at last row
            b[-1] = 1.0
            c[-1] = 0.0

            ab = np.zeros((3, ROWS))
            ab[0, 1:] = c
            ab[1, :] = b
            ab[2, :-1] = a

            p_new = solve_banded((1,1), ab, bvec)
            P_LUM_new[:, j] = alpha * p_new + (1 - alpha) * P_LUM[:, j]

        if np.max(np.abs(P_LUM_new - P_LUM)) < tol:
            break

        P_LUM[:] = P_LUM_new

    A_cross = 0.25 * np.pi * d_0**2

    Q_IBF_x = G * np.diff(P_LUM, axis=1)  # length ROWS-1
    Q_TFF_mid_x = 0.5 * (Q_TFF[:, :-1] + Q_TFF[:, 1:])
    Q_lum_x = Q_IBF_x - Q_TFF_mid_x
    U_BLOOD[0, :, :-1] = Q_lum_x / A_cross

    Q_IBF_y = G * np.diff(P_LUM, axis=0)  # length ROWS-1
    Q_TFF_mid_y = 0.5 * (Q_TFF[:-1, :] + Q_TFF[1:, :])
    Q_lum_y = Q_IBF_y - Q_TFF_mid_y
    U_BLOOD[1, :-1, :] = Q_lum_y / A_cross


def update_iff(alpha=0.25, tol=1e-6, max_iter=100):
    global P_INS, U_INS, P_LUM, CELLS
    new_p_ins = P_INS.copy()
    for _ in range(max_iter):
        laplacian = (
            np.roll(P_INS, 1, axis=0) + np.roll(P_INS, -1, axis=0) +
            np.roll(P_INS, 1, axis=1) + np.roll(P_INS, -1, axis=1) - 4 * P_INS
        ) / DELTA**2

        rhs_EC = L_pEC * (S_VEC / K_insEC) * ((P_LUM - P_INS) - (pi_lum - pi_insEC) * delta_EC) * CELLS[0]
        rhs_TC = L_pTC * (S_VTC / K_insTC) * ((P_LUM - P_INS) - (pi_lum - pi_insTC) * delta_TC) * CELLS[1]
        update = rhs_EC + rhs_TC + laplacian

        new_p_ins = P_INS + alpha * update

        if np.max(np.abs(new_p_ins - P_INS)) < tol:
            break

        P_INS[:] = new_p_ins.copy()

    dp_dy, dp_dx = np.gradient(P_INS, DELTA)
    rhs_EC = -K_insEC * np.stack((dp_dx, dp_dy), axis=0) * CELLS[0]
    rhs_TC = -K_insTC * np.stack((dp_dx, dp_dy), axis=0) * CELLS[1]
    U_INS[:] = rhs_EC + rhs_TC

def update_hemorheology():
    # TODO: d_0 użyte jako placeholder; d_v powinno być aktualizowane w każdym kroku czasowym
    # właściwie ta funkcja nic nie robi jeżeli używamy d_0
    mu_n = 3.2 + 6*np.exp(-0.085 * d_0) - 2.44 * np.exp(-0.06 * d_0**0.645)
    
    # TODO: zakładamy, że H_D=1 (nie do końca rozumiem jak to działa)
    mu_blood = (1 + (mu_n-1) * 1 * (d_0/(d_0-1.1))**2) * ((d_0/(d_0-1.1))**2) * mu_plasma
    return mu_blood

def update_angiogenesis():
    global RHO_EC, VEGF, ECM
    diff_term = D_tEC * laplacian(RHO_EC)
    
    grad_cv_x, grad_cv_y = gradient(VEGF)
    grad_ce_x, grad_ce_y = gradient(ECM)
    
    chemotaxis_x = (beta_c / (1 + VEGF)) * RHO_EC * grad_cv_x
    chemotaxis_y = (beta_c / (1 + VEGF)) * RHO_EC * grad_cv_y
    haptotaxis_x = beta_h * RHO_EC * grad_ce_x
    haptotaxis_y = beta_h * RHO_EC * grad_ce_y
    
    taxis_div = divergence(chemotaxis_x + haptotaxis_x,
                           chemotaxis_y + haptotaxis_y)
    
    RHO_EC[:] = RHO_EC + DT * (diff_term - taxis_div)
    RHO_EC[RHO_EC < 0] = 0

def update_tumor_growth():
    global RHO_TC, ECM
    diff_term = D_TC * laplacian(RHO_TC)
    
    grad_ce_x, grad_ce_y = gradient(ECM)
    
    haptotaxis_x = beta_h * RHO_TC * grad_ce_x
    haptotaxis_y = beta_h * RHO_TC * grad_ce_y
    
    taxis_div = divergence(haptotaxis_x, haptotaxis_y)
    
    RHO_TC[:] = RHO_TC + DT * (diff_term - taxis_div)
    RHO_TC[RHO_TC < 0] = 0


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
    # update_iff()
    update_hemorheology()

    update_tumor_growth()
    update_angiogenesis()


def update_tme():
    update_molecular_scale()
    update_tissue_scale()
    update_cellular_scale() # is this order correct?