import numpy as np
from config import *

time_step = 0

def update_environment():
    global ENV, time_step
    assumed_p_ins = 5
    assumed_p_lum = 40
    assumed_diameter_ratio = 0.9  # Pulse strength
    
    reaction_O2 = -CELL_RES["gamma_0"]*update_vitality()*CELLS[2] - CELL_RES["eps_O2"]*ENV[0]
    reaction_g = -1/6*CELL_RES["gamma_0"]*update_vitality()*CELLS[2] - CELL_RES["eps_g"]*ENV[1]
    reaction_CO2 = CELL_RES["gamma_0"]*update_vitality()*CELLS[2] - CELL_RES["eps_CO2"]*ENV[2]
    reaction = np.stack([reaction_O2, reaction_g, reaction_CO2])

    diffusion = np.zeros_like(ENV)
    laplacian = np.array([np.gradient(np.gradient(ENV[i], DELTA)[0], DELTA)[0] + np.gradient(np.gradient(ENV[i], DELTA)[1], DELTA)[1] for i in range(3)])
    diffusion = DIFFUSION * laplacian
    
    convection = np.zeros_like(ENV)
    left = np.roll(ENV, 1, axis=2)
    left[:, :, 0] = ENV[:, :, 0]
    up = np.roll(ENV, 1, axis=1)
    up[:, 0, :] = ENV[:, 0, :]
    convection = V * (left + up) / DELTA * VESSEL_MAP

    delta_di = (1-(1-DIAMETER/HEMODYNAMICS_TME["dt_p"])**2)**2
    P_i = ((1-delta_di) * (HEMODYNAMICS_TME["L0_p"] / HEMODYNAMICS_TME["k_iEC"] * (assumed_p_lum - assumed_p_ins - HEMODYNAMICS_TME["delta_vEC"] * (HEMODYNAMICS_TME["pi_lum"]-HEMODYNAMICS_TME["pi_insEC"])))).reshape(3, 1, 1)
    vascular = HEMODYNAMICS_TME["k_iEC"] * (P_i * HEMODYNAMICS_TME["S_VEC"] * assumed_diameter_ratio * PLASMA + HEMODYNAMICS_TME["S_VEC"] * assumed_diameter_ratio * (PLASMA - ENV)*P_i/(np.exp(P_i)-1))
    pulse = (time_step * DT * 60) % BPM == 0
    vascular = vascular * pulse * VESSEL_MAP - DT*convection

    ENV[:] = ENV + DT*reaction + DT*vascular + DT*diffusion
    ENV[:] = np.maximum(ENV, 0)

    time_step += 1

    return ENV

def update_vitality():
    o2_component = VIT_EN["phi"] * ENV[0]/(ENV[0]+CELL_RES["ch_O2"])+VIT_EN["k_W"]
    glucose_component = ENV[1]/(ENV[1]+CELL_RES["ch_g"])
    co2_component = np.exp(-5 * (ENV[2]/CELL_RES["ch_CO2"]-1)**4 * (ENV[2]-CELL_RES["ch_CO2"]>0))

    TUMOR[0, :, :] = o2_component * glucose_component * co2_component
    return o2_component * glucose_component * co2_component

def update_energy():
    # TODO: drug_impact = k_ac * c_ac * V_saturation
    V = TUMOR[0]
    V_saturation = V / (V + 1)
    drug_impact = 0
    active_cells = ((VIT_EN["kp_a"] * V - VIT_EN["kc_a"] * V_saturation - drug_impact) *
                    (V > VIT_EN["v_ch"]))
    quiescent_cells = (-VIT_EN["kc_q"] * V_saturation) * (V <= VIT_EN["v_ch"])

    TUMOR[1, :, :] = active_cells + quiescent_cells
    return active_cells + quiescent_cells

def update_tumor_density():
    laplacian = np.gradient(np.gradient(TUMOR[2], DELTA)[0], DELTA)[0] + np.gradient(np.gradient(TUMOR[2], DELTA)[1], DELTA)[1]
    randomwalk = GROWTH["D_TC"] * laplacian

    # TODO: haptotaxis (ECM density)
    haptotaxis = 0

    dV_dy, dV_dx = np.gradient(TUMOR[0], DELTA)
    cooption_y = GROWTH["beta_COP"] * TUMOR[2] * dV_dy
    cooption_x = GROWTH["beta_COP"] * TUMOR[2] * dV_dx

    divergence = np.gradient(cooption_y, DELTA, axis=0)[0] + np.gradient(cooption_x, DELTA, axis=1)[1]

    TUMOR[2, :, :] = TUMOR[2] + randomwalk - divergence

def grow_tumor():
    update_vitality()
    update_energy()
    update_tumor_density()
    update_environment()
    print(f"V: {np.mean(TUMOR[0])}")
    print(f"E: {np.mean(TUMOR[1])}")

    for i in range(ROWS):
        for j in range(COLS):
            v = TUMOR[0, i, j]
            e = TUMOR[1, i, j]

            # active
            if CELLS[2, i, j]:
                # quiescence
                if v < VIT_EN["v_ch"]:
                    CELLS[2, i, j] = 0
                    CELLS[3, i, j] = 1
                
                # division
                elif e > VIT_EN["psi_ch"]:
                    i_min, i_max = max(0, i-1), min(ROWS, i+2)
                    j_min, j_max = max(0, j-1), min(COLS, j+2)

                    block = TUMOR[2, i_min:i_max, j_min:j_max].copy()
                    block_center = (i - i_min, j - j_min)
                    neighbors = np.delete(block.flatten(), block_center[0] * (j_max - j_min) + block_center[1])
                    weights = neighbors / neighbors.sum()
                    choice = np.random.choice(8, p=weights)
                    indices = [(i_offset, j_offset) for i_offset in range(i_min, i_max) 
                             for j_offset in range(j_min, j_max)
                             if (i_offset, j_offset) != (i, j)]

                    chosen_i, chosen_j = indices[choice]
                    CELLS[2, chosen_i, chosen_j] = 1
            
            # quiescent
            if CELLS[3, i, j]:
                # wake up
                if v > VIT_EN["v_ch"]:
                    CELLS[2, i, j] = 1
                    CELLS[3, i, j] = 0
                # necrosis
                elif e < 1e-30:
                    CELLS[5, i, j] = 1
                    CELLS[3, i, j] = 0
