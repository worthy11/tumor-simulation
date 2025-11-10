import numpy as np

def _draw_segment(mask, start, end, radius):
    r0, c0 = start
    r1, c1 = end

    min_r = max(0, int(np.floor(min(r0, r1) - radius - 1)))
    max_r = min(mask.shape[0] - 1, int(np.ceil(max(r0, r1) + radius + 1)))
    min_c = max(0, int(np.floor(min(c0, c1) - radius - 1)))
    max_c = min(mask.shape[1] - 1, int(np.ceil(max(c0, c1) + radius + 1)))

    if min_r > max_r or min_c > max_c:
        return

    rr = np.arange(min_r, max_r + 1, dtype=np.float64)
    cc = np.arange(min_c, max_c + 1, dtype=np.float64)
    R, C = np.meshgrid(rr, cc, indexing="ij")

    ab = np.array([r1 - r0, c1 - c0], dtype=np.float64)
    ab_norm_sq = np.dot(ab, ab)

    if ab_norm_sq == 0:
        dist = np.sqrt((R - r0) ** 2 + (C - c0) ** 2)
    else:
        ap_row = R - r0
        ap_col = C - c0
        t = (ap_row * ab[0] + ap_col * ab[1]) / ab_norm_sq
        t = np.clip(t, 0.0, 1.0)

        proj_row = r0 + t * ab[0]
        proj_col = c0 + t * ab[1]
        dist = np.sqrt((R - proj_row) ** 2 + (C - proj_col) ** 2)

    mask[min_r : max_r + 1, min_c : max_c + 1] |= dist <= radius


def _draw_polyline(mask, points, radius):
    if len(points) < 2:
        return
    for start, end in zip(points[:-1], points[1:]):
        _draw_segment(mask, start, end, radius)


def generate_vessel_pattern(rows, cols, seed=7):
    rng = np.random.default_rng(seed)
    vessels = np.zeros((rows, cols), dtype=bool)
    major_paths = []

    def add_major_trunk(amplitude, phase, thickness, orientation="horizontal"):
        num_points = 320
        t = np.linspace(0.0, 1.0, num_points)
        if orientation == "horizontal":
            r = rows * 0.2 + t * rows * 0.6 + amplitude * np.sin(2 * np.pi * (t + phase))
            c = t * (cols - 1)
        else:
            r = t * (rows - 1)
            c = cols * 0.25 + t * cols * 0.5 + amplitude * np.sin(2 * np.pi * (t + phase))
        points = np.stack([np.clip(r, 0, rows - 1), np.clip(c, 0, cols - 1)], axis=1)
        major_paths.append(points)
        _draw_polyline(vessels, points, thickness)

    add_major_trunk(amplitude=60.0, phase=0.1, thickness=5.5, orientation="horizontal")
    add_major_trunk(amplitude=45.0, phase=0.45, thickness=4.8, orientation="vertical")

    if major_paths:
        samples = []
        for path in major_paths:
            idx = np.linspace(0, len(path) - 1, num=90, dtype=int)
            samples.append(path[idx])
        anchor_points = np.concatenate(samples, axis=0)

        branch_count = 8
        for _ in range(branch_count):
            anchor = anchor_points[rng.integers(0, len(anchor_points))]
            length = rng.uniform(110.0, 180.0)
            angle = rng.uniform(-1.3, 1.3)
            curvature = rng.uniform(0.6, 1.6)
            wobble = rng.uniform(12.0, 28.0)
            base_t = np.linspace(0.0, 1.0, int(max(10, length // 8)))

            dr = length * np.sin(angle) * base_t
            dc = length * np.cos(angle) * base_t

            modulation_phase = rng.uniform(0.0, 2.0 * np.pi)
            modulation = np.sin(curvature * np.pi * base_t + modulation_phase)
            jitter_r = wobble * modulation
            jitter_c = wobble * np.cos(curvature * np.pi * base_t + modulation_phase)

            branch_r = np.clip(anchor[0] + dr + jitter_r, 0, rows - 1)
            branch_c = np.clip(anchor[1] + dc + 0.4 * jitter_c, 0, cols - 1)
            branch_points = np.stack([branch_r, branch_c], axis=1)
            _draw_polyline(vessels, branch_points, radius=rng.uniform(1.6, 2.6))

    loop_count = 3
    for _ in range(loop_count):
        center_r = rng.uniform(rows * 0.2, rows * 0.8)
        center_c = rng.uniform(cols * 0.2, cols * 0.8)
        radius_major = rng.uniform(35.0, 55.0)
        radius_minor = radius_major * rng.uniform(0.55, 0.8)
        thickness = rng.uniform(1.8, 2.8)
        angles = np.linspace(0.0, 2.0 * np.pi, 120)
        loop_r = np.clip(center_r + radius_major * np.sin(angles), 0, rows - 1)
        loop_c = np.clip(center_c + radius_minor * np.cos(angles), 0, cols - 1)
        loop_points = np.stack([loop_r, loop_c], axis=1)
        _draw_polyline(vessels, loop_points, thickness)

    return vessels

# Grid parameters
ROWS = 500
COLS = 500
DELTA = 2e-5

# Time parameters
DT = 60*12
# DT = 1
STEPS = 1000

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

import random
# Cell types array
CELLS = np.zeros((6, ROWS, COLS), dtype=bool)
# 0 - endothelial cell
# 1 - active tumor cell
# 2 - quiescent tumor cell
# 3 - migrating tumor cell
# 4 - necrotic tumor cell
# 5 - vein entry points

# Create a thin vertical vein at ~2/3 of the grid width.
# Grid points occupied by the vein are marked in the endothelial layer
# (CELLS[0] == True). Tumor cells (CELLS[1]) are still allowed to
# appear on the same grid points (layers are independent boolean masks).
VEIN_COL = int(COLS * 3 / 5)
VEIN_WIDTH = 3  # thin vein (number of columns)
col_start = max(0, VEIN_COL - VEIN_WIDTH // 2)
col_end = min(COLS, VEIN_COL + VEIN_WIDTH // 2 + 1)
CELLS[0, :, col_start:col_end] = True
CELLS[0, col_start:col_end, :] = True
# Vein entry points: cover full vein thickness at intersection + 3 other spots
vein_width = col_end - col_start
half_w = max(1, vein_width // 2)
vein_mid = (col_start + col_end) // 2

# Helper to clamp ranges
def _clamp(a, lo, hi):
	return max(lo, min(hi, a))

# 1) Intersection of the two veins (square patch thickness x thickness)
CELLS[5, col_start:col_end, col_start:col_end] = True

# 2) On the vertical vein near the top (square patch centered at (r_top, vein_mid))
r_top_center = ROWS // 4
r0 = _clamp(r_top_center - half_w, 0, ROWS)
r1 = _clamp(r_top_center + half_w + (vein_width % 2), 0, ROWS)
CELLS[5, r0:r1, col_start:col_end] = True

# 3) On the vertical vein near the bottom (square patch centered at (r_bot, vein_mid))
r_bot_center = (3 * ROWS) // 4
r0 = _clamp(r_bot_center - half_w, 0, ROWS)
r1 = _clamp(r_bot_center + half_w + (vein_width % 2), 0, ROWS)
CELLS[5, r0:r1, col_start:col_end] = True

# 4) On the horizontal vein towards the left (square patch centered at (vein_mid, c_left))
c_left_center = COLS // 4
c0 = _clamp(c_left_center - half_w, 0, COLS)
c1 = _clamp(c_left_center + half_w + (vein_width % 2), 0, COLS)
CELLS[5, col_start:col_end, c0:c1] = True

start_x, start_y = ROWS // 2, COLS // 2
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
D_EC = 1e-13       # diffusivity of tip epithelial cells
D_TC = 1e-13        # diffusivity of tumor cells
beta_c = 0.26           # weight coefficient of chemotaxis
beta_h = 0.1            # weight coefficient of haptotaxis
alpha = 1               # saturation coefficient of chemotaxis

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