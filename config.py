import numpy as np
import cv2

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


# --- Vein entry point utilities ---
def _skeletonize_binary(mask_bool: np.ndarray) -> np.ndarray:
    """Return a 1-pixel wide skeleton of the binary mask as a boolean array.

    Uses a standard morphological skeletonization (erode->open->subtract) loop
    with a 3x3 cross structuring element, available in base OpenCV.
    """
    img = (mask_bool.astype(np.uint8) * 255)
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Iteratively erode and accumulate the residuals until fully eroded
    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break

    return skel > 0


def _junctions_and_endpoints(skel_bool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect junction (degree >=3) and endpoint (degree==1) pixels on a skeleton.

    Returns two boolean masks (junctions, endpoints).
    """
    sk = skel_bool.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    # neighbor_count includes the center pixel itself
    neighbor_count = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    deg = neighbor_count - sk
    junctions = (sk == 1) & (deg >= 3)
    endpoints = (sk == 1) & (deg == 1)
    return junctions, endpoints


def _select_spaced_points(coords: np.ndarray, min_dist: float, max_points: int | None, seed: int = 7) -> list[tuple[int, int]]:
    """Greedy selection of points from coords (Nx2 [r,c]) keeping >= min_dist spacing.

    Uses a coarse grid to make checks O(1) on average.
    """
    if coords.size == 0:
        return []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(coords))
    selected: list[tuple[int, int]] = []
    cell_size = max(1, int(min_dist))
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def cell_idx(r: int, c: int) -> tuple[int, int]:
        return (r // cell_size, c // cell_size)

    min_dist_sq = float(min_dist) * float(min_dist)
    for i in order:
        r, c = int(coords[i, 0]), int(coords[i, 1])
        ci_r, ci_c = cell_idx(r, c)
        ok = True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nb = buckets.get((ci_r + dr, ci_c + dc))
                if not nb:
                    continue
                for pr, pc in nb:
                    if (r - pr) * (r - pr) + (c - pc) * (c - pc) < min_dist_sq:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            selected.append((r, c))
            buckets.setdefault((ci_r, ci_c), []).append((r, c))
            if max_points is not None and len(selected) >= max_points:
                break
    return selected


def make_vein_entry_points(
    vein_mask_bool: np.ndarray,
    seed: int = 7,
    min_dist: float = 30.0,
    max_intersections: int = 200,
    max_additional: int = 150,
    include_endpoints: bool = False,
) -> list[tuple[int, int]]:
    """Generate vein entry points predominantly at crossings, plus some along vessels.

    Parameters
    - vein_mask_bool: True for vessel pixels
    - min_dist: minimum spacing between chosen entry points (pixels)
    - max_intersections: cap on number of intersection-based points
    - max_additional: additional points sampled along skeleton away from intersections
    - include_endpoints: also allow endpoints with spacing
    """
    skel = _skeletonize_binary(vein_mask_bool)
    junctions, endpoints = _junctions_and_endpoints(skel)

    # 1) Prefer junctions (crossings)
    j_coords = np.column_stack(np.where(junctions))
    chosen = _select_spaced_points(j_coords, min_dist=min_dist, max_points=max_intersections, seed=seed)

    # 2) Optionally add endpoints (still spaced from chosen)
    if include_endpoints and max_additional > 0:
        # Mask out vicinity of already chosen points using a distance transform
        mask_far = np.ones(vein_mask_bool.shape, dtype=np.uint8)
        for r, c in chosen:
            mask_far[r, c] = 0
        dist = cv2.distanceTransform(mask_far, distanceType=cv2.DIST_L2, maskSize=3)
        ep_coords = np.column_stack(np.where(endpoints & (dist >= min_dist)))
        add_ep = _select_spaced_points(ep_coords, min_dist=min_dist, max_points=max_additional, seed=seed + 1)
        chosen.extend(add_ep)
        max_additional = max(0, max_additional - len(add_ep))

    # 3) Add some interior points along skeleton, away from intersections
    if max_additional > 0:
        mask_far = np.ones(vein_mask_bool.shape, dtype=np.uint8)
        for r, c in chosen:
            mask_far[r, c] = 0
        dist = cv2.distanceTransform(mask_far, distanceType=cv2.DIST_L2, maskSize=3)
        sk_coords = np.column_stack(np.where(skel & (dist >= min_dist)))
        add_sk = _select_spaced_points(sk_coords, min_dist=min_dist, max_points=max_additional, seed=seed + 2)
        chosen.extend(add_sk)

    return chosen

# Grid parameters
ROWS = 1000
COLS = 1000
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


VEINS = cv2.imread('vein_pattern.jpg', cv2.IMREAD_GRAYSCALE)
if VEINS is None:
    vessels_bool = generate_vessel_pattern(ROWS, COLS, seed=7)
    VEINS = np.where(vessels_bool, 0, 255).astype(np.uint8)

# Boolean mask: True where vessel lumen/wall is present
VEIN_MASK = VEINS < 128
CELLS[0] = VEIN_MASK

# Create vein entry points (CELLS[5]) at crossings and selected interior spots
ENTRY_SEED = 42
# Increase spacing and lower caps to reduce total number of entry points
ENTRY_MIN_DIST = 45.0
MAX_INTERSECTIONS = 150
MAX_ADDITIONAL = 60
INCLUDE_ENDPOINTS = False

entry_points = make_vein_entry_points(
    VEIN_MASK,
    seed=ENTRY_SEED,
    min_dist=ENTRY_MIN_DIST,
    max_intersections=MAX_INTERSECTIONS,
    max_additional=MAX_ADDITIONAL,
    include_endpoints=INCLUDE_ENDPOINTS,
)

if entry_points:
    rr, cc = zip(*entry_points)
    CELLS[5, np.array(rr, dtype=int), np.array(cc, dtype=int)] = True
    # Slightly enlarge (dilate) the entry point regions to make them more accessible
    ENTRY_POINT_RADIUS = 3  # slightly larger regions
    ep_mask = CELLS[5].astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ENTRY_POINT_RADIUS + 1, 2 * ENTRY_POINT_RADIUS + 1)
    )
    ep_mask = cv2.dilate(ep_mask, kernel)
    # Ensure enlarged regions remain within vein lumen
    ep_mask &= VEIN_MASK.astype(np.uint8)
    CELLS[5] = ep_mask.astype(bool)



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