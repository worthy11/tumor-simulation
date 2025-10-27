import numpy as np

# Grid parameters
ROWS = 500
COLS = 500
STEPS = 1000

# Time parameters
DT = 60
DELTA = 2e-5

O2 = np.zeros((ROWS, COLS), dtype=np.float64)
G = np.zeros((ROWS, COLS), dtype=np.float64)
CO2 = np.zeros((ROWS, COLS), dtype=np.float64)
ECM = np.zeros((ROWS, COLS), dtype=np.float64)
MMP = np.zeros((ROWS, COLS), dtype=np.float64)
VEGF = np.zeros((ROWS, COLS), dtype=np.float64)
VEGFR2 = np.zeros((2, ROWS, COLS), dtype=np.float64)
# 0 - free (unbound) VEGFR-2 receptors
# 1 - active (bound) VEGFR-2 receptors
V = np.zeros((ROWS, COLS), dtype=np.float64)
E = np.zeros((ROWS, COLS), dtype=np.float64)
P_INS = np.zeros((ROWS, COLS), dtype=np.float64)
P_LUM = np.zeros((ROWS, COLS), dtype=np.float64)
RHO_TC = np.zeros((ROWS, COLS), dtype=np.float64)
RHO_EC = np.zeros((ROWS, COLS), dtype=np.float64)
U_INS = np.zeros((2, ROWS, COLS), dtype=np.float64)
U_BLOOD = np.zeros((2, ROWS, COLS), dtype=np.float64)

# P_INS - interstitial fluid pressure
# P_LUM - intravascular blood pressure (czemu nie moze byc P_BLOOD??)
# U_INS - interstitial fluid flow velocity
# U_BLOOD - intravascular blood flow velocity
# P = P_LUM-P_INS - transvascular pressure

# Cell types array
CELLS = np.zeros((5, ROWS, COLS), dtype=np.bool)
# 0 - endothelial cell
# 1 - active tumor cell
# 2 - quiescent tumor cell
# 3 - migrating tumor cell
# 4 - necrotic tumor cell
start_x, start_y = ROWS // 2, COLS // 2
# CELLS[0, :, start_y+start_y//2:start_y+start_y//2+5] = 1
CELLS[0, :, start_y+start_y//2:] = 1
CELLS[1, start_x-1:start_x+2, start_y] = 1
CELLS[1, start_x, start_y-1:start_y+2] = 1

# Cell respiration parameters
D_O2 = 8e-14        # diffusion coefficient of O2
D_g = 8e-14         # diffusion coefficient of glucose
D_CO2 = 4e-14       # diffusion coefficient of CO2
gamma_0 = 3e-8      # max consumption/production rate of CR-agents
f_O2 = 6.8e-7       # max consumption rate of O2
f_g = 8e-6          # max consumption rate of glucose
km_g = 3.4e-3       # Michaelis constant for glucose
f_CO2 = 2.5e-5      # max production rate of CO2
ch_O2 = 8.4e-3      # characteristic O2 concentration
ch_g = 8.7e-3       # characteristic glucose concentration
ch_CO2 = 10.5e-3    # characteristic CO2 concentration

# VEGF parameters
D_VEGF = 2.9e-11    # diffusion coefficient
r_VEGF = 2e-12      # production rate by tumor cells
k_plus_VEGF = 1.3e-2   # binding rate by VEGFR-2
k_minus_VEGF = 6.3e-5  # unbinding rate from VEGFR-2
eps_VEGF = 2.78e-7  # natural decay rate
ch_VEGF = 1.1e-8    # characteristic concentration
c_pVEGF = 5.78          # plasma concentration
f_VEGF = 1e-20     # flux of VEGF from blood to TME

# MMP/ECM parameters
D_MMP = 1e-15       # diffusion coefficient
r_MMPTC = 1.7e-13   # production rate by tumor cells
r_MMPEC = 0.3e-13   # production rate by epithelial cells
eps_MMP = 1.7e-8    # natural decay rate
eps_ECM = 1.3e-7    # natural decay rate
ch_ECM = 1.36e-9    # characteristic concentration
ch_MMP = 1.36e-9    # characteristic concentration

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
D_tEC = 1e-13       # diffusivity of tip epithelial cells
D_TC = 1e-13        # diffusivity of tumor cells

# Hemodynamics/TME parameters (HC = hydraulic conductivity)
K_insEC = 9e-15  # interstitial HC of TME for normal tissue
K_insTC = 4.5e-15  # interstitial HC of TME for tumor tissue
L_pEC = 3.6e-10      # reference value for HC of neo-vessel wall
L_pTC = 2.8e-9      # reference value for HC of neo-vessel wall
delta_EC = 0.91        # average oncotic reflection coefficient of plasma proteins for healthy tissue
delta_TC = 0.82        # average oncotic reflection coefficient of plasma proteins for tumor tissue
pi_lum = 20             # collide osmotic pressures of intravascular plasma
pi_insEC = 10           # collide osmotic pressure of interstitial fluid in healthy tissue
pi_insTC = 15           # collide osmotic pressure of interstitial fluid in tumor tissue
S_VEC = 7e3         # characteristic value of area/unit of neo-vessels in healthy tissue
S_VTC = 2e4         # characteristic value of area/unit of neo-vessels in tumor tissue
u_0 = 5e-6                # characteristic TME pressure and WSS
u_0f = 1.3e-2
p_0 = 60

# Angiogenesis parameters
d_c = 5e-5  # characteristic vessel diameter
d_0 = 3e-5  # stress-free vessel diameter

# Hemorgeology parameters
mu_plasma = 9e-6