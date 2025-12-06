import numpy as np
import cv2

def _skeletonize_binary(mask_bool: np.ndarray) -> np.ndarray:
    img = (mask_bool.astype(np.uint8) * 255)
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

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
    sk = skel_bool.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    deg = neighbor_count - sk
    junctions = (sk == 1) & (deg >= 3)
    endpoints = (sk == 1) & (deg == 1)
    return junctions, endpoints


def _select_spaced_points(coords: np.ndarray, min_dist: float, max_points: int | None, seed: int = 7) -> list[tuple[int, int]]:
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
    skel = _skeletonize_binary(vein_mask_bool)
    junctions, endpoints = _junctions_and_endpoints(skel)

    j_coords = np.column_stack(np.where(junctions))
    chosen = _select_spaced_points(j_coords, min_dist=min_dist, max_points=max_intersections, seed=seed)

    if include_endpoints and max_additional > 0:
        mask_far = np.ones(vein_mask_bool.shape, dtype=np.uint8)
        for r, c in chosen:
            mask_far[r, c] = 0
        dist = cv2.distanceTransform(mask_far, distanceType=cv2.DIST_L2, maskSize=3)
        ep_coords = np.column_stack(np.where(endpoints & (dist >= min_dist)))
        add_ep = _select_spaced_points(ep_coords, min_dist=min_dist, max_points=max_additional, seed=seed + 1)
        chosen.extend(add_ep)
        max_additional = max(0, max_additional - len(add_ep))

    if max_additional > 0:
        mask_far = np.ones(vein_mask_bool.shape, dtype=np.uint8)
        for r, c in chosen:
            mask_far[r, c] = 0
        dist = cv2.distanceTransform(mask_far, distanceType=cv2.DIST_L2, maskSize=3)
        sk_coords = np.column_stack(np.where(skel & (dist >= min_dist)))
        add_sk = _select_spaced_points(sk_coords, min_dist=min_dist, max_points=max_additional, seed=seed + 2)
        chosen.extend(add_sk)

    return chosen

ROWS = 500
COLS = 500
DELTA = 2e-5

DT = 60*12
STEPS = 320

O2 = np.zeros((ROWS, COLS), dtype=np.float64)
G = np.zeros((ROWS, COLS), dtype=np.float64)
CO2 = np.zeros((ROWS, COLS), dtype=np.float64)
ECM = np.zeros((ROWS, COLS), dtype=np.float64)
MMP = np.zeros((ROWS, COLS), dtype=np.float64)
V = np.zeros((ROWS, COLS), dtype=np.float64)
E = np.zeros((ROWS, COLS), dtype=np.float64)
P_INS = np.zeros((ROWS, COLS), dtype=np.float64)
P_LUM = np.zeros((ROWS, COLS), dtype=np.float64)
RHO_TC = np.zeros((ROWS, COLS), dtype=np.float64)
U_INS = np.zeros((2, ROWS, COLS), dtype=np.float64)
TUMOR_SUBTYPE = np.zeros((ROWS, COLS), dtype=np.uint8)

CELLS = np.zeros((6, ROWS, COLS), dtype=bool)
# 0 - endothelial cell
# 1 - active tumor cell
# 2 - quiescent tumor cell
# 3 - migrating tumor cell
# 4 - necrotic tumor cell
# 5 - vein entry points

pattern_filepath = "vein_patterns/zyla1meta.png"

VEINS = np.zeros((ROWS, COLS))
VEINS = cv2.imread(pattern_filepath, cv2.IMREAD_GRAYSCALE)
VEINS = cv2.resize(VEINS, (ROWS, COLS))
VEINS = cv2.bitwise_not(VEINS)
METASTATIC = True
MEZ = True

CELLS[0] = VEINS

ENTRY_SEED = np.random.randint(1, 100000)
ENTRY_MIN_DIST = 100
MAX_INTERSECTIONS = 80
MAX_ADDITIONAL = 20
INCLUDE_ENDPOINTS = False

if METASTATIC:
    entry_points = make_vein_entry_points(
        VEINS,
        seed=ENTRY_SEED,
        min_dist=ENTRY_MIN_DIST,
        max_intersections=MAX_INTERSECTIONS,
        max_additional=MAX_ADDITIONAL,
        include_endpoints=INCLUDE_ENDPOINTS,
    )

    if entry_points:
        rr, cc = zip(*entry_points)
        CELLS[5, np.array(rr, dtype=int), np.array(cc, dtype=int)] = True
        ENTRY_POINT_RADIUS = 3
        ep_mask = CELLS[5].astype(np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * ENTRY_POINT_RADIUS + 1, 2 * ENTRY_POINT_RADIUS + 1)
        )
        ep_mask = cv2.dilate(ep_mask, kernel)
        ep_mask &= VEINS.astype(np.uint8)
        CELLS[5] = ep_mask.astype(bool)

start_x, start_y = ROWS // 2, COLS // 2
CELLS[1, start_x-1:start_x+2, start_y] = 1
CELLS[1, start_x, start_y-1:start_y+2] = 1

# Initialize tumor subtypes for the seeded cells: 60% mez, 40% nie_mez
if MEZ:
    init_coords = np.column_stack(np.where(CELLS[1]))
    if init_coords.size > 0:
        rng = np.random.default_rng(1)
        n = len(init_coords)
        mez_count = int(np.round(0.6 * n))
        if mez_count > 0:
            chosen = rng.choice(n, size=mez_count, replace=False)
            for idx in chosen:
                r, c = init_coords[idx]
                TUMOR_SUBTYPE[r, c] = 1

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

# MMP/ECM parameters
D_MMP = 1e-15       # diffusion coefficient
r_MMPTC = 1.7e-13   # production rate by tumor cells
r_MMPEC = 0.3e-13   # production rate by epithelial cells
eps_MMP = 1.7e-8    # natural decay rate
eps_ECM = 1.3e-7    # natural decay rate
ch_ECM = 1.36e-9    # characteristic concentration

# Vitality and energy parameters
phi = 3.67              # proportionality coefficient of consumption/production rate of CR-agents
v_ch = 0.5              # characteristic cellular vitality for active tumor cells
psi_ch = 50             # characteristic cellular energy for proliferation
kc_q = 0.1              # constant consumption rate of cellular energy by quiescent tumor cells
kp_a = 1                # coefficient of production rate of cellular energy by active tumor cells
kc_a = 1                # max consumption rate of cellular energy by active tumor cells

# Tumor growth parameters
D_TC = 1e-13        # diffusivity of tumor cells
beta_h = 0.1            # weight coefficient of haptotaxis

# Angiogenesis parameters
d_c = 5e-5  # characteristic vessel diameter
d_0 = 3e-5  # stress-free vessel diameter