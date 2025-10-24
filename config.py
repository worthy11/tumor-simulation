import numpy as np

# Grid parameters
ROWS = 50
COLS = 50
STEPS = 1000

# Time parameters
DT = 1e-1
DELTA = 5 * 1e-6

O2 = G = CO2 = ECM = MMP = VEGF = V = E = P_INS = P_LUM = RHO_TC = RHO_EC = np.zeros((ROWS, COLS), dtype=np.float64)
U_INS = U_BLOOD = np.zeros((2, ROWS, COLS), dtype=np.float64)

# P_INS - interstitial fluid pressure
# P_LUM - intravascular blood pressure
# U_INS - interstitial fluid flow velocity
# U_BLOOD - intravascular blood flow velocity
# P = P_LUM - P_INS

# Cell types array
CELLS = np.zeros((5, ROWS, COLS))
# 0 - endothelial cell
# 1 - active tumor cell
# 2 - quiescent tumor cell
# 3 - migrating tumor cell
# 4 - necrotic tumor cell
start_x, start_y = ROWS // 2, COLS // 2
CELLS[2, start_x-1:start_x+2, start_y] = 1
CELLS[2, start_x, start_y-1:start_y+2] = 1

# Cell respiration parameters
D_O2 = 8 * 1e-14        # diffusion coefficient of O2
D_g = 8 * 1e-14         # diffusion coefficient of glucose
D_CO2 = 4 * 1e-14       # diffusion coefficient of CO2
gamma_0 = 3 * 1e-8      # max consumption/production rate of CR-agents
f_O2 = 6.8 * 1e-7       # max consumption rate of O2
f_g = 8 * 1e-6          # max consumption rate of glucose
km_g = 3.4 * 1e-3       # Michaelis constant for glucose
f_CO2 = 2.5 * 1e-5      # max production rate of CO2
ch_O2 = 8.4 * 1e-3      # characteristic O2 concentration
ch_g = 8.7 * 1e-3       # characteristic glucose concentration
ch_CO2 = 10.5 * 1e-3    # characteristic CO2 concentration

# VEGF parameters
D_VEGF = 2.9 * 1e-11    # diffusion coefficient
r_VEGF = 2 * 1e-12      # production rate by tumor cells
k_plus_VEGF = 1.3 * 1e-2   # binding rate by VEGFR-2
k_minus_VEGF = 6.3 * 1e-5  # unbinding rate from VEGFR-2
eps_VEGF = 2.78 * 1e-7  # natural decay rate
ch_VEGF = 1.1 * 1e-8    # characteristic concentration
c_pVEGF = 5.78          # plasma concentration

# MMP/ECM parameters
D_MMP = 1 * 1e-15       # diffusion coefficient
r_MMPTC = 1.7 * 1e-13   # production rate by tumor cells
r_MMPEC = 0.3 * 1e-13   # production rate by epithelial cells
eps_MMP = 1.7 * 1e-8    # natural decay rate
eps_ECM = 1.3 * 1e-7    # natural decay rate
ch_ECM = 1.36 * 1e-9    # characteristic concentration
ch_MMP = 1.36 * 1e-9    # characteristic concentration
c_pMMP = 72             # plasma concentration

# Vitality and energy parameters
phi = 3.67              # proportionality coefficient of consumption/production rate of CR-agents
v_ch = 0.5              # characteristic cellular vitality for active tumor cells
psi_ch = 30             # characteristic cellular energy for proliferation
kc_q = 0.1              # constant consumption rate of cellular energy by quiescent tumor cells
kp_a = 1                # coefficient of production rate of cellular energy by active tumor cells
kc_a = 1                # max consumption rate of cellular energy by active tumor cells

# Tumor growth parameters
alpha = 1               # saturation coefficient of chemotaxis
beta_c = 0.26           # weight coefficient of chemotaxis
beta_h = 0.1            # weight coefficient of haptotaxis
beta_COP = 0.3          # weight coefficient of cooption
D_tEC = 1 * 1e-13       # diffusivity of tip epithelial cells
D_TC = 1 * 1e-13        # diffusivity of tumor cells

# Hemodynamics/TME parameters (HC = hydraulic conductivity)
K_insEC = 8.53 * 1e-13  # interstitial HC of TME for normal tissue
K_insTC = 4.13 * 1e-12  # interstitial HC of TME for tumor tissue
k_ps = 1.0              # tumor-induced reduction coefficient of HC
L0_p = 3.6 * 1e-10      # reference value for HC of neo-vessel wall
k_L = 1.1 * 1e-8        # constant to control effect of VEGF on vessel wall HC
delta_vEC = 0.91        # average oncotic reflection coefficient of plasma proteins for healthy tissue
delta_vTC = 0.82        # average oncotic reflection coefficient of plasma proteins for tumor tissue
pi_lum = 20             # collide osmotic pressures of intravascular plasma
pi_insEC = 10           # collide osmotic pressure of interstitial fluid in healthy tissue
pi_insTC = 15           # collide osmotic pressure of interstitial fluid in tumor tissue
S_VEC = 7 * 1e3         # characteristic value of area/unit of neo-vessels in healthy tissue
S_VTC = 2 * 1e4         # characteristic value of area/unit of neo-vessels in tumor tissue
p_0 = 60                # characteristic TME pressure and WSS
p_m = 60                # max pressure in primary vessels
dt_p = 400              # intratumoral vessel wall pore size (nm)
k_iEC = 0.73 * 1e-11    # permeability coefficient of healthy vessel wall
k_iTC = 5.73 * 1e-11    # permeability coefficient of healthy tumor wall

# Angiogenesis parameters
d_c = 5 * 1e-5  # characteristic vessel diameter
d_0 = 3 * 1e-5  # stress-free vessel diameter