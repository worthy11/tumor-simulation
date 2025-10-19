ROWS = 200
COLS = 200
DIVISION_PROB = 0.2
DEATH_PROB = 0.05
STEPS = 100


CELLULAR_RESPIRATION = {
    "D_O2": 1 * 1e-9,      # diffusion coefficient of O2
    "D_g": 3.6 * 1e-10,    # diffusion coefficient of glucose
    "D_CO2": 7.4 * 1e-10,  # diffusion coefficient of CO2
    "gamma_0": 3 * 1e-8,   # max consumption/production rate of CR-agents
    "ch_O2": 8.4 * 1e-3,   # characteristic O2 concentration
    "ch_g": 5.5 * 1e-3,    # characteristic glucose concentration
    "ch_CO2": 10.5 * 1e-3, # characteristic CO2 concentration
    "c_pO2": 8.4 * 1e-3,   # plasma concentraction of O2
    "c_pg": 5.5 * 1e-3,    # plasma concentraction of glucose
    "c_pCO2": 1.2 * 1e-3,  # plasma concentraction of CO2
    "eps_O2": 1 * 1e-9,    # natural decay rate of O2
    "eps_g": 1.7 * 1e-10,  # natural decay rate of glucose
    "eps_CO2": 1 * 1e-9,   # natural decay rate of CO2
    "r_f": 1,               # retardation factor
    "d_O2": 0.3 # !!! nm !!!
}
V = CELLULAR_RESPIRATION["r_f"] * 0.7 * 1e-6

GROWTH_FACTORS = {
                            # -- Values concern VEGF --
    "D_VEGF": 2.9 * 1e-11,    # diffusion coefficient
    "r_VEGF": 2 * 1e-12,      # production rate by tumor cells
    "k+_VEGF": 1.3 * 1e-2,    # binding rate by VEGFR-2
    "k-_VEGF": 6.3 * 1e-5,    # ubinding rate from VEGFR-2
    "eps_VEGF": 2.78 * 1e-7,  # natural decay rate
    "ch_VEGF": 1.1 * 1e-8,    # characteristic concentration
    "c_pVEGF": 5.78            # plasma concentration
}

ECM = {
                                # -- Values concern MMP / ECM --
    "D_MMP": 1 * 1e-15,        # diffusion coefficient
    "r_MMPTC": 1.7 * 1e-13,    # production rate by tumor cells
    "r_MMPEC": 0.3 * 1e-13,    # production rate by epithelial cells
    "eps_MMP": 1.7 * 1e-8,     # natural decay rate
    "eps_ECM": 1.3 * 1e-7,     # natural decay rate
    "ch_ECM": 1.36 * 1e-9,     # characteristic concentration
    "ch_MMP": 1.36 * 1e-9,     # characteristic concentration
    "c_pMMP": 72                # plasma concentration
}

VITALITY_ENERGY = {
    "phi" : 3.67,       # proportionality coefficient of consumption/production rate of CR-agents
    "v_ch": .5,         # characteristic cellular vitality for active tumor cells
    "psi_ch": 30,       # characteristic cellular energy for proliferation
    "kc_q": .1,         # constant consumption rate of cellular energy by quiescent tumor cells
    "kp_a": 1,          # coefficient of production rate of cellular energy by active tumor cells
    "kc_a": 1,          # max consumption rate of cellular energy by active tumor cells
    "k_W": 1 * 1e-10,  # Warburg effect of tumor cells
}

TUMOR_GROWTH_ANGIO = {
    "alpha": 1,             # saturation coefficient of chemotaxis
    "beta_c": 0.26,         # weight coefficient of chemotaxis
    "beta_h": 0.1,          # weight coefficient of haptotaxis
    "beta_COP": 0.3,        # weight coefficient of cooption
    "D_tEC": 1 * 1e-13,    # diffusivity of tip epithelial cells
    "D_TC": 1 * 1e-13,     # diffusivity of tumor cells
}

HEMODYNAMICS_TME = {
                             # -- HC = hydraulic conductivity --
    "K_insEC": 8.53 * 1e-13, # interstitial HC of TME for normal tissue
    "K_insTC": 4.13 * 1e-12, # interstitial HC of TME for tumor tissue
    "k_ps": 1.,              # tumor-induced reduction coefficient of HC
    "L0_p": 3.6 * 1e-10,    # reference value for HC of neo-vessel wall
    "k_L": 1.1 * 1e-8,      # constant to control effect of VEGF on vessel wall HC
    "delta_vEC": 0.91,       # average oncotic reflection coefficient of plasma proteins for healthy tissue
    "delta_vTC": 0.82,       # average oncotic reflection coefficient of plasma proteins for tumor tissue
    "pi_lum": 20,            # collide osmotic pressures of intravascular plasma
    "pi_insEC": 10,          # collide osmotic pressure of interstitial fluid in healthy tissue
    "pi_insTC": 15,          # collide osmotic pressure of interstitial fluid in tumor tissue
    "S_VEC": 7 * 1e3,       # characteristic value of area/unit of neo-vessels in healthy tissue
    "S_VTC": 2 * 1e4,       # characteristic value of area/unit of neo-vessels in tumor tissue
    "p_0": 60,               # characteristic TME pressure and WSS
    "p_m": 60,               # max pressure in primary vessels
    "dt_p": 400, # !!!nm!!!  # intratumoral vessel wall pore size
    "k_iEC": 0.73 * 1e-11,   # permeability coefficient of healthy vessel wall
    "k_iTC": 5.73 * 1e-11    # permeability coefficient of healthy tumor wall
}

ANGIO = {
    "d_c": 5 * 1e-7
}