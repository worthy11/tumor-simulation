import numpy as np
from config import *

O2 = np.full((ROWS, COLS), 0.5 * 1e-7) #mM
GLUCOSE = np.full((ROWS, COLS), 4 * 1e-3)
CO2 = np.full((ROWS, COLS), 2 * 1e-3)
VITALITY = np.full((ROWS, COLS), 2 * 1e-3)
ENERGY = np.full((ROWS, COLS), 2 * 1e-3)

time_step = 0

def update_O2_concentration():
    global O2, time_step
    delta = 2 * 1e-5
    assumed_p_ins = 5
    assumed_p_lum = 40
    assumed_diameter_ratio = 0.3  # Pulse strength
    
    reaction = -CELLULAR_RESPIRATION["gamma_0"]*get_vitality() - CELLULAR_RESPIRATION["eps_O2"]*O2
    convection = np.zeros_like(O2)
    diffusion = np.zeros_like(O2)
    
    for i in range(ROWS):
        for j in range(COLS):
            self = O2[i][j]
            
            left = O2[i][j-1] if j > 0 else O2[i][j]
            right = O2[i][j+1] if j < COLS-1 else O2[i][j]
            up = O2[i-1][j] if i > 0 else O2[i][j]
            down = O2[i+1][j] if i < ROWS-1 else O2[i][j]

            convection[i][j] = V * (left + up) / delta

            laplacian = (left + right + up + down - 4*self) / delta**2
            diffusion[i][j] = CELLULAR_RESPIRATION["D_O2"] * laplacian

    delta_di = (1-(1-CELLULAR_RESPIRATION["d_O2"]/HEMODYNAMICS_TME["dt_p"])**2)**2
    P_i = (1-delta_di) * (HEMODYNAMICS_TME["L0_p"] / HEMODYNAMICS_TME["k_iEC"] * (assumed_p_lum - assumed_p_ins - HEMODYNAMICS_TME["delta_vEC"] * (HEMODYNAMICS_TME["pi_lum"]-HEMODYNAMICS_TME["pi_insEC"])))
    vascular = HEMODYNAMICS_TME["k_iEC"] * (P_i * HEMODYNAMICS_TME["S_VEC"] * assumed_diameter_ratio * CELLULAR_RESPIRATION["c_pO2"] + HEMODYNAMICS_TME["S_VEC"] * assumed_diameter_ratio * (CELLULAR_RESPIRATION["c_pO2"] - O2)*P_i/(np.exp(P_i)-1))
    
    pulse = (time_step * DT * 60) % BPM == 0
    vascular = vascular * pulse
    time_step += 1

    vascular = (DT*vascular - DT*convection) * VESSEL_MAP
    O2 = O2 + DT*reaction + DT*vascular + DT*diffusion
    O2 = np.maximum(O2, 0)

    return O2


def get_glucose_concentration():
    pass
def get_CO2_concentration():
    pass
def get_VEGF_concentration():
    pass
def get_ECM_concentration():
    pass
def get_MMP_concentration():
    pass

def get_vitality():
    o2_component = VITALITY_ENERGY["phi"] * O2/(O2+CELLULAR_RESPIRATION["ch_O2"])+VITALITY_ENERGY["k_W"]
    glucose_component = GLUCOSE/(GLUCOSE+CELLULAR_RESPIRATION["ch_g"])
    co2_component = np.exp(-5 * (CO2/CELLULAR_RESPIRATION["ch_CO2"]-1)**4 * (CO2-CELLULAR_RESPIRATION["ch_CO2"]>0))

    return o2_component * glucose_component * co2_component

def get_energy():
    # TODO: drug_impact = k_ac * c_ac * V_saturation
    V = get_vitality()
    V_saturation = V / (V + 1)
    drug_impact = 0
    active_cells = ((VITALITY_ENERGY["kp_a"] * V - VITALITY_ENERGY["kc_a"] * V_saturation - drug_impact) *
                    (V > VITALITY_ENERGY["v_ch"]))
    quiescent_cells = (-VITALITY_ENERGY["kc_q"] * V_saturation) * (V <= VITALITY_ENERGY["v_ch"])

    return active_cells + quiescent_cells