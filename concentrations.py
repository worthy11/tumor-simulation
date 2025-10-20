import numpy as np
from config import *

O2 = np.full((ROWS, COLS), 0.5 * 10e-3) #mM
GLUCOSE = np.full((ROWS, COLS), 4 * 10e-3)
CO2 = np.full((ROWS, COLS), 2 * 10e-3)
VITALITY = np.full((ROWS, COLS), 2 * 10e-3)
ENERGY = np.full((ROWS, COLS), 2 * 10e-3)

vessel_map = np.zeros((ROWS, COLS))
vessel_map[0:3, :] = 1.0  # Top edge vessels
vessel_map[:, 0:3] = 1.0  # Left edge vessels
for i in range(5, ROWS, 50):
    vessel_map[i:i+2, :] = 0.5  # Smaller vessels
for j in range(5, COLS, 50):
    vessel_map[:, j:j+2] = 0.5  # Smaller vessels

time_step = 0

def update_O2_concentration(dt=1e-2, use_pulsatile=False):
    global O2, time_step
    
    reaction = - CELLULAR_RESPIRATION["gamma_0"]*get_vitality() - CELLULAR_RESPIRATION["eps_O2"]*O2

    delta_di = (1-(1-CELLULAR_RESPIRATION["d_O2"]/HEMODYNAMICS_TME["dt_p"])**2)**2
    P_i = (1-delta_di)*(HEMODYNAMICS_TME["L0_p"]/HEMODYNAMICS_TME["k_iEC"]*(40-5-HEMODYNAMICS_TME["delta_vEC"]*(HEMODYNAMICS_TME["pi_lum"]-HEMODYNAMICS_TME["pi_insEC"])))
    vascular_base = HEMODYNAMICS_TME["k_iEC"] * (P_i * HEMODYNAMICS_TME["S_VEC"] * 0.5 * CELLULAR_RESPIRATION["c_pO2"] + HEMODYNAMICS_TME["S_VEC"] * 0.5 * (CELLULAR_RESPIRATION["c_pO2"] - O2)*P_i/(np.exp(P_i)-1))
    
    VASCULAR_SCALE = 1e-10  # Tune this: higher = more O2 supply, lower = less supply
    vascular = vessel_map * vascular_base * VASCULAR_SCALE
    
    if use_pulsatile:
        heartbeat_freq = 1  # Hz
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * heartbeat_freq * time_step * dt)
        vascular = vascular * pulse
        time_step += 1
    
    delta = 2 * 1e-7
    
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

    O2 = O2 - dt * convection + dt * diffusion + dt * reaction + dt * vascular
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