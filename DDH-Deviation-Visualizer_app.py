import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# --------------- ANGLES AND BASIC VECTORS ---------------

def deg2rad(a):
    return np.deg2rad(a)

def step_vector(ds, az_deg, dip_deg):
    """
    ds meters, azimuth clockwise from North, dip from horizontal positive down.
    Returns dE, dN, dDepth (depth positive down).
    """
    az = deg2rad(az_deg)
    dip = deg2rad(dip_deg)
    horiz = ds * np.cos(dip)
    dE = horiz * np.sin(az)
    dN = horiz * np.cos(az)
    dDepth = ds * np.sin(dip)  # positive down
    return dE, dN, dDepth

def unit_from_azimuth(az_deg):
    az = deg2rad(az_deg)
    return np.array([np.sin(az), np.cos(az)])

# --------------- SECTION PROJECTIONS ---------------

def project_to_vertical_plane(E, N, D, E0, N0, az_deg):
    """
    Vertical plane through (E0,N0) oriented along az_deg.
    Returns x_along, y_depth, offset_out_of_plane.
    """
    az = deg2rad(az_deg)
    ux, uy = np.sin(az), np.cos(az)        # in-plane x direction
    nx, ny = np.cos(az), -np.sin(az)       # plane normal (horizontal)
    dE = E - E0
    dN = N - N0
    x = dE * ux + dN * uy
    off = dE * nx + dN * ny
    y = D
    return x, y, off

# --------------- TRAJECTORY BUILDERS ---------------

def trajectory_planned(E0, N0, dip0_deg, az0_deg, length_m, lift_per_100=0.0, drift_per_100=0.0, step=5.0):
    num = int(np.ceil(length_m / step))
    E = [E0]; N = [N0]; D = [0.0]
    az = az0_deg; dip = dip0_deg
    d_dip_per_m = -lift_per_100 / 100.0
    d_az_per_m = drift_per_100 / 100.0
    for i in range(num):
        ds = step if (i < num - 1) else (length_m - step * (num - 1))
        dE, dN, dDep = step_vector(ds, az, dip)
        E.append(E[-1] + dE)
        N.append(N[-1] + dN)
        D.append(D[-1] + dDep)
        az += d_az_per_m * ds
        dip += d_dip_per_m * ds
    return np.array(E), np.array(N), np.array(D)

def trajectory_from_surveys(E0, N0, surveys, total_length, rem_lift_per_100=0.0, rem_drift_per_100=0.0, step=5.0):
    if len(surveys) == 0:
        raise ValueError("Provide at least one survey row.")
    surveys = sorted(surveys, key=lambda s: s["depth"])
    if surveys[0]["depth"] > 0:
        surveys = [{"depth": 0.0, "azimuth": surveys[0]["azimuth"], "dip": surveys[0]["dip"]}] + surveys

    E = [E0]; N = [N0]; D = [0.0]
    # Integrate between survey stations with linear change of az and dip
    for i in range(len(surveys) - 1):
        d0 = surveys[i]["depth"]; d1 = surveys[i + 1]["depth"]
        if d1 <= d0:
            continue
        az0, az1 = surveys[i]["azimuth"], surveys[i + 1]["azimuth"]
        dip0, dip1 = surveys[i]["dip"], surveys[i + 1]["dip"]
        d_az_per_m = (az1 - az0) / (d1 - d0)
        d_dip_per_m = (dip1 - dip0) / (d1 - d0)
        s = d0; az = az0; dip = dip0
        while s < d1 - 1e-6:
            ds = min(step, d1 - s)
            dE, dN, dDep = step_vector(ds, az, dip)
            E.append(E[-1] + dE); N.append(N[-1] + dN); D.append(D[-1] + dDep)
            s += ds; az += d_az_per_m * ds; dip += d_dip_per_m * ds

    # Extrapolate to total length with remaining lift and drift
    last = surveys[-1]
    curr_depth = last["depth"]; az = last["azimuth"]; dip = last["dip"]
    if total_length > curr_depth + 1e-6:
        d_az_per_m = rem_drift_per_100 / 100.0
        d_dip_per_m = -rem_lift_per_100 / 100.0
        s = curr_depth
        while s < total_length - 1e-6:
            ds = min(step, total_length - s)
            dE, dN, dDep = step_vector(ds, az, dip)
            E.append(E[-1] + dE); N.append(N[-1] + dN); D.append(D[-1] + dDep)
            s += ds; az += d_az_per_m * ds; dip += d_dip_per_m * ds

    return np.array(E), np.array(N), np.array(D)

# --------------- TARGET PLANE ---------------

def plane_normal_from_strike_dip(strike_deg, dip_deg):
    """
    Right-hand convention, strike clockwise from North, dip from horizontal positive down.
    Returns normal vector n = [nx, ny, nz] in E, N, Depth coords (Depth positive down).
    """
    a = deg2rad(strike_deg)
    d = deg2rad(dip_deg)
    nx = np.cos(a) * np.sin(d)
    ny = -np.sin(a) * np.sin(d)
    nz = -np.cos(d)
    return np.array([nx, ny, nz], dtype=float)

def plane_signed_distance(E, N, D, plane):
    # plane dict: {"E0","N0","D0","n":np.array([nx,ny,nz])}
    dE = E - plane["E0"]; dN = N - plane["N0"]; dD = D - plane["D0"]
    return plane["n"][0] * dE + plane["n"][1] * dN + plane["n"][2] * dD

def hole_plane_pierce_point(E, N, D, plane):
    """
    Piecewise linear intersection. Returns (Ep, Np, Dp) or None if no crossing.
    Also returns measured distance from collar along the polyline.
    """
    # cumulative 3D distances to approximate along-hole MD for display
    seg_len = np.sqrt(np.diff(E)**2 + np.diff(N)**2 + np.diff(D)**2)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    svals = plane_signed_distance(E, N, D, plane)
    for i in range(len(E) - 1):
        s0, s1 = svals[i], svals[i + 1]
        if s0 == 0:
            return (E[i], N[i], D[i], cum[i])
        if s0 * s1 < 0 or s1 == 0:
            denom = (s0 - s1)
            if abs(denom) < 1e-12:
                continue
            t = s0 / (s0 - s1)
            Ep = E[i] + t * (E[i + 1] - E[i])
            Np = N[i] + t * (N[i + 1] - N[i])
            Dp = D[i] + t * (D[i + 1] - D[i])
            md = cum[i] + t * seg_len[i]
            return (Ep, Np, Dp, md)
    return None

# --------------- STREAMLIT UI ---------------

st.set_page_config(page_title="Drill Tracking - Plan, Section, Long Section", layout="wide")
st.title("Drill Tracking - plan, cross section, long section with target plane")

with st.sidebar:
    st.subheader("Planned hole")
    collar_E = st.number_input("Collar Easting m", value=500000.0, step=1.0, format="%.3f")
    collar_N = st.number_input("Collar Northing m", value=5500000.0, step=1.0, format="%.3f")
    planned_az = st.number_input("Planned azimuth deg (clockwise from North)", value=355.0, step=0.1)
    planned_dip = st.number_input("Planned dip deg (positive down from horizontal)", value=68.0, step=0.1)
    eoh = st.number_input("Planned end of hole length m", value=730.0, step=1.0)
    plan_lift = st.number_input("Planned lift deg per 100 m", value=4.0, step=0.1)
    plan_drift = st.number_input("Planned drift deg per 100 m", value=2.0, step=0.1)

    st.subheader("Current hole surveys")
    st.caption("Depth m, azimuth clockwise from North, dip positive down from horizontal.")
    default_rows = [
        {"depth": 0.0, "azimuth": 355.0, "dip": 68.0},
        {"depth": 100.0, "azimuth": 355.5, "dip": 66.0},
        {"depth": 200.0, "azimuth": 356.0, "dip": 64.0},
        {"depth": 300.0, "azimuth": 357.0, "dip": 62.0},
    ]
    df = st.data_editor(
        default_rows,
        num_rows="dynamic",
        key="survey_editor",
        column_config={
            "depth": st.column_config.NumberColumn("depth", step=1.0),
            "azimuth": st.column_config.NumberColumn("azimuth", step=0.1),
            "dip": st.column_config.NumberColumn("dip", step=0.1),
        },
    )
    rem_lift = st.number_input("Remaining lift deg per 100 m after last survey", value=0.0, step=0.1)
    rem_drift = st.number_input("Remaining drift deg per 100 m after last survey", value=0.0, step=0.1)
    step_len = st.number_input("Integration step m", value=5.0, min_value=0.5, step=0.5)

    st.subheader("Target plane")
    plane_strike = st.number_input("Strike azimuth deg", value=110.0, step=0.1)
    plane_dip = st.number_input("Dip deg (from horizontal, positive down)", value=60.0, step=0.1)
    plane_E0 = st.number_input("Reference point Easting m", value=collar_E, step=1.0, format="%.3f")
    plane_N0 = st.number_input("Reference point Northing m", value=collar_N, step=1.0, format="%.3f")
    plane_D0 = st.number_input("Reference point depth m (positive down)", value=200.0, step=1.0)

# Parse surveys
surveys = []
for r in df:
    try:
        surveys.append({"depth": float(r["depth"]), "azimuth": float(r["azimuth"]), "dip": float(r["dip"])})
    except Exception:
        pass

# Build holes
try:
    E_plan, N_plan, D_plan = trajectory_planned(
        collar_E, collar_N, planned_dip, planned_az, eoh, lift_per_100=plan_lift, drift_per_100=plan_drift, step=step_len
    )
    E_curr, N_curr, D_curr = trajectory_from_surveys(
        collar_E, collar_N, surveys, eoh, rem_lift_per_100=rem_lift, rem_drift_per_100=rem_drift, step=step_len
    )
except Exception as e:
    st.error(f"Could not build trajectories: {e}")
    st.stop()

# Plane object
plane = {"E0": plane_E0, "N0": plane_N0, "D0": plane_D0, "n": plane_normal_from_strike_dip(plane_strike, plane_dip)}

# Pierce points
pp_planned = hole_plane_pierce_point(E_plan, N_plan, D_plan, plane)
pp_current = hole_plane_pierce_point(E_curr, N_curr, D_curr, plane)

if pp_planned is None or pp_current is None:
    st.warning("One or both holes do not intersect the target plane within the modeled length.")
else:
    (EpP, NpP, DpP, mdP) = pp_planned
    (EpC, NpC, DpC, mdC) = pp_current

# --------------- VIEW HELPERS ---------------

def plane_trace_in_plan(extent_half, center_E, center_N, strike_deg):
    """Return two endpoints of the strike line for plotting in plan view."""
    u = unit_from_azimuth(strike_deg)
    p1 = np.array([center_E, center_N]) - extent_half * u
    p2 = np.array([center_E, center_N]) + extent_half * u
    return p1, p2

def plane_trace_in_section(section_az_deg, section_E0, section_N0, plane, x_min, x_max, num=200):
    """
    Compute the line of the plane within a vertical section whose horizontal axis is along section_az_deg.
    For each x along section, solve for depth D that satisfies the plane equation.
    """
    az = deg2rad(section_az_deg)
    ux, uy = np.sin(az), np.cos(az)
    xs = np.linspace(x_min, x_max, num)
    Es = section_E0 + xs * ux
    Ns = section_N0 + xs * uy
    nx, ny, nz = plane["n"]
    # D = D0 - (nx*(E - E0) + ny*(N - N0))/nz
    with np.errstate(divide="ignore", invalid="ignore"):
        Ds = plane["D0"] - (nx * (Es - plane["E0"]) + ny * (Ns - plane["N0"])) / nz
    return xs, Ds

# --------------- PROJECTIONS FOR THE THREE VIEWS ---------------

# Cross section - vertical plane through planned collar, oriented along planned azimuth
xs_P, ys_P, off_P = project_to_vertical_plane(E_plan, N_plan, D_plan, collar_E, collar_N, planned_az)
xs_C, ys_C, off_C = project_to_vertical_plane(E_curr, N_curr, D_curr, collar_E, collar_N, planned_az)

# Long section - vertical plane through plane reference point, oriented along strike
xsL_P, ysL_P, offL_P = project_to_vertical_plane(E_plan, N_plan, D_plan, plane_E0, plane_N0, plane_strike)
xsL_C, ysL_C, offL_C = project_to_vertical_plane(E_curr, N_curr, D_curr, plane_E0, plane_N0, plane_strike)

# For plotting extents
all_E = np.concatenate([E_plan, E_curr])
all_N = np.concatenate([N_plan, N_curr])
all_D = np.concatenate([D_plan, D_curr])
E_min, E_max = float(np.min(all_E)), float(np.max(all_E))
N_min, N_max = float(np.min(all_N)), float(np.max(all_N))
D_max = float(np.max(all_D))
padE = max(10.0, 0.05 * (E_max - E_min + 1.0))
padN = max(10.0, 0.05 * (N_max - N_min + 1.0))
padD = max(10.0, 0.05 * D_max)

# Section ranges
xP_min, xP_max = float(np.min(np.concatenate([xs_P, xs_C]))), float(np.max(np.concatenate([xs_P, xs_C])))
xL_min, xL_max = float(np.min(np.concatenate([xsL_P, xsL_C]))), float(np.max(np.concatenate([xsL_P, xsL_C])))

# Plane traces in views
sec_x, sec_y = plane_trace_in_section(planned_az, collar_E, collar_N, plane, xP_min - 20.0, xP_max + 20.0)
long_x, long_y = plane_trace_in_section(plane_strike, plane_E0, plane_N0, plane, xL_min - 20.0, xL_max + 20.0)
plan_center_E = 0.5 * (E_min + E_max); plan_center_N = 0.5 * (N_min + N_max)
extent_half = 0.75 * max(E_max - E_min, N_max - N_min, 100.0)
(plan_p1, plan_p2) = plane_trace_in_plan(extent_half, plane_E0, plane_N0, plane_strike)

# Pierce points projected to each view
if pp_planned is not None and pp_current is not None:
    # Plan
    EpP2, NpP2 = EpP, NpP
    EpC2, NpC2 = EpC, NpC
    # Cross section
    ppx_P, ppy_P, _ = project_to_vertical_plane(np.array([EpP]), np.array([NpP]), np.array([DpP]), collar_E, collar_N, planned_az)
    ppx_C, ppy_C, _ = project_to_vertical_plane(np.array([EpC]), np.array([NpC]), np.array([DpC]), collar_E, collar_N, planned_az)
    # Long section
    pplx_P, pply_P, _ = project_to_vertical_plane(np.array([EpP]), np.array([NpP]), np.array([DpP]), plane_E0, plane_N0, plane_strike)
    pplx_C, pply_C, _ = project_to_vertical_plane(np.array([EpC]), np.array([NpC]), np.array([DpC]), plane_E0, plane_N0, plane_strike)

# --------------- METRICS - DEVIATION ---------------

def fmt_m(v): return f"{v:.2f} m"

if pp_planned is not None and pp_current is not None:
    dE = EpC - EpP; dN = NpC - NpP; dD = DpC - DpP
    dist_3d = float(np.sqrt(dE**2 + dN**2 + dD**2))
    dist_plan = float(np.sqrt(dE**2 + dN**2))
    # cross section components
    dx_sec = float(ppx_C[0] - ppx_P[0]); dy_sec = float(ppy_C[0] - ppy_P[0])
    dist_sec = float(np.sqrt(dx_sec**2 + dy_sec**2))
    # long section components
    dx_long = float(pplx_C[0] - pplx_P[0]); dy_long = float(pply_C[0] - pply_P[0])
    dist_long = float(np.sqrt(dx_long**2 + dy_long**2))

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Deviation 3D", fmt_m(dist_3d))
    colB.metric("Plan distance", fmt_m(dist_plan))
    colC.metric("Cross section distance", fmt_m(dist_sec))
    colD.metric("Long section distance", fmt_m(dist_long))

    st.caption(
        f"Pierce points along hole: planned MD ~ {mdP:.1f} m, current MD ~ {mdC:.1f} m."
    )

# --------------- PLOTS ---------------

tab_plan, tab_xsec, tab_lsec = st.tabs(["Plan view", "Cross section", "Long section"])

with tab_plan:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.plot(E_plan, N_plan, label="Planned hole")
    ax.plot(E_curr, N_curr, label="Current hole")
    # Plane strike trace
    ax.plot([plan_p1[0], plan_p2[0]], [plan_p1[1], plan_p2[1]], linestyle="--", label="Target plane trace")
    # Pierce points and distance
    if pp_planned is not None and pp_current is not None:
        ax.scatter([EpP2, EpC2], [NpP2, NpC2], s=60, zorder=5, label="Pierce points")
        ax.plot([EpP2, EpC2], [NpP2, NpC2], linestyle=":", linewidth=1.5)
        ax.annotate(f"{np.sqrt((EpC2-EpP2)**2 + (NpC2-NpP2)**2):.2f} m",
                    xy=((EpP2+EpC2)/2, (NpP2+NpC2)/2), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Easting m"); ax.set_ylabel("Northing m")
    ax.set_xlim(E_min - padE, E_max + padE); ax.set_ylim(N_min - padN, N_max + padN)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend()
    st.pyplot(fig)

with tab_xsec:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(xs_P, ys_P, label="Planned hole", linewidth=2)
    ax.plot(xs_C, ys_C, label="Current hole", linewidth=2)
    # Plane line in section
    ax.plot(sec_x, sec_y, linestyle="--", label="Target plane")
    # Collar on top
    ax.scatter([0.0], [0.0], s=60, zorder=5, label="Collar")
    # Pierce points and distance
    if pp_planned is not None and pp_current is not None:
        ax.scatter([ppx_P[0], ppx_C[0]], [ppy_P[0], ppy_C[0]], s=60, zorder=6, label="Pierce points")
        ax.plot([ppx_P[0], ppx_C[0]], [ppy_P[0], ppy_C[0]], linestyle=":", linewidth=1.5)
        ax.annotate(f"{np.sqrt((ppx_C[0]-ppx_P[0])**2 + (ppy_C[0]-ppy_P[0])**2):.2f} m",
                    xy=((ppx_P[0]+ppx_C[0])/2, (ppy_P[0]+ppy_C[0])/2), xytext=(5, -10), textcoords="offset points")
    ax.set_xlabel("Section distance along planned azimuth m")
    ax.set_ylabel("Depth m, positive down")
    max_depth = max(float(np.nanmax(ys_P)), float(np.nanmax(ys_C)))
    max_x = max(abs(float(np.nanmin(xs_P))), abs(float(np.nanmin(xs_C))), abs(float(np.nanmax(xs_P))), abs(float(np.nanmax(xs_C))))
    pad_x = max(10.0, 0.05 * max_x); pad_y = max(10.0, 0.05 * max_depth)
    ax.set_xlim(-pad_x, max_x + pad_x)
    ax.set_ylim(max_depth + pad_y, -pad_y)
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend()
    st.pyplot(fig)

with tab_lsec:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(xsL_P, ysL_P, label="Planned hole", linewidth=2)
    ax.plot(xsL_C, ysL_C, label="Current hole", linewidth=2)
    # Plane line in long section
    ax.plot(long_x, long_y, linestyle="--", label="Target plane")
    # Pierce points and distance
    if pp_planned is not None and pp_current is not None:
        ax.scatter([pplx_P[0], pplx_C[0]], [pply_P[0], pply_C[0]], s=60, zorder=6, label="Pierce points")
        ax.plot([pplx_P[0], pplx_C[0]], [pply_P[0], pply_C[0]], linestyle=":", linewidth=1.5)
        ax.annotate(f"{np.sqrt((pplx_C[0]-pplx_P[0])**2 + (pply_C[0]-pply_P[0])**2):.2f} m",
                    xy=((pplx_P[0]+pplx_C[0])/2, (pply_P[0]+pply_C[0])/2), xytext=(5, -10), textcoords="offset points")
    ax.set_xlabel("Long section distance along strike m")
    ax.set_ylabel("Depth m, positive down")
    max_depth_L = max(float(np.nanmax(ysL_P)), float(np.nanmax(ysL_C)))
    max_x_L = max(abs(float(np.nanmin(xsL_P))), abs(float(np.nanmin(xsL_C))), abs(float(np.nanmax(xsL_P))), abs(float(np.nanmax(xsL_C))))
    pad_xL = max(10.0, 0.05 * max_x_L); pad_yL = max(10.0, 0.05 * max_depth_L)
    ax.set_xlim(-pad_xL, max_x_L + pad_xL)
    ax.set_ylim(max_depth_L + pad_yL, -pad_yL)
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend()
    st.pyplot(fig)

# --------------- OFF-PLANE INFO ---------------

mean_off_sec = float(np.nanmean(np.abs(off_C))) if off_C.size else 0.0
max_off_sec = float(np.nanmax(np.abs(off_C))) if off_C.size else 0.0
mean_off_long = float(np.nanmean(np.abs(offL_C))) if offL_C.size else 0.0
max_off_long = float(np.nanmax(np.abs(offL_C))) if offL_C.size else 0.0

st.caption(
    f"Cross section offsets - mean {mean_off_sec:.1f} m, max {max_off_sec:.1f} m. "
    f"Long section offsets - mean {mean_off_long:.1f} m, max {max_off_long:.1f} m. "
    f"Section plane: vertical through planned collar, azimuth {planned_az:.1f} deg. "
    f"Long section plane: vertical through plane reference point, along strike {plane_strike:.1f} deg."
)

