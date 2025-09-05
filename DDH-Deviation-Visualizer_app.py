import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ----------------- GEOMETRY HELPERS -----------------

def generate_segment(xyz0, dip_deg, az_deg, seg_len):
    """
    Single straight segment from a start point with given dip and az over seg_len.

    Coordinates:
      x east, y north, z is elevation (up). Positive dip is down.
      Drilling down must decrease elevation -> dz negative.
    """
    dip_rad = np.radians(dip_deg)
    az_rad = np.radians(az_deg)
    dx = seg_len * np.sin(az_rad) * np.cos(dip_rad)
    dy = seg_len * np.cos(az_rad) * np.cos(dip_rad)
    dz = -seg_len * np.sin(dip_rad)  # negative because elevation goes down when drilling
    return xyz0 + np.array([dx, dy, dz])

def plane_normal_from_strike_dip(strike, dip):
    """
    Plane normal from strike and dip (dip positive down). z is elevation (up).
    strike measured clockwise from north.
    """
    dipdir = (strike + 90.0) % 360.0
    dip_rad = np.radians(dip)
    dipdir_rad = np.radians(dipdir)
    nx = np.sin(dip_rad) * np.sin(dipdir_rad)
    ny = np.sin(dip_rad) * np.cos(dipdir_rad)
    nz = -np.cos(dip_rad)
    return np.array([nx, ny, nz])

def segment_plane_intersect(p1, p2, n, p0):
    """
    Intersect segment p1->p2 with plane n•(r - p0) = 0.
    Return intersection point or None if no intersection on the segment.
    """
    u = p2 - p1
    denom = np.dot(n, u)
    num = np.dot(n, p0 - p1)
    if np.isclose(denom, 0.0):
        return None
    t = num / denom
    if 0.0 <= t <= 1.0:
        return p1 + t * u
    return None

def polyline_plane_pierce(pts, n, p0):
    """Find first intersection of a polyline with a plane."""
    for i in range(len(pts) - 1):
        hit = segment_plane_intersect(pts[i], pts[i+1], n, p0)
        if hit is not None:
            return hit
    return None

def project_long_section(x, y, z, strike, dip, origin):
    """
    Project to plane coordinates: along strike (s) and down-dip (d) axes.
    z is elevation (up). d increases in the dip direction.
    """
    strike_rad = np.radians(strike)
    dip_rad = np.radians(dip)
    dipdir_rad = np.radians((strike + 90.0) % 360.0)

    u_strike = np.array([np.sin(strike_rad), np.cos(strike_rad), 0.0])
    u_dip = np.array([
        np.sin(dipdir_rad) * np.cos(dip_rad),
        np.cos(dipdir_rad) * np.cos(dip_rad),
        -np.sin(dip_rad)
    ])
    coords = np.vstack([x, y, z]).T - origin
    s = coords @ u_strike
    d = coords @ u_dip
    return s, d

def project_cross_section(x, y, z, azim, origin):
    """Project to vertical section aligned with azimuth azim. Returns horizontal section distance."""
    azim_rad = np.radians(azim)
    u_sec = np.array([np.sin(azim_rad), np.cos(azim_rad), 0.0])
    coords = np.vstack([x, y, z]).T - origin
    sec_x = coords @ u_sec
    return sec_x

def add_end_arrow(ax, x, y, color='black'):
    """Draw a single arrow at the end of a line, if there are at least 2 points."""
    if len(x) < 2:
        return
    ax.quiver(x[-2], y[-2], x[-1] - x[-2], y[-1] - y[-2],
              angles='xy', scale_units='xy', scale=1.0, width=0.01, color=color)

# ----------------- STREAMLIT UI -----------------

st.title("DDH Deviation Visualizer")

st.sidebar.header("Inputs")

# Collar and global controls
collar_x = st.sidebar.number_input("Collar X (m)", value=0.0, step=0.1)
collar_y = st.sidebar.number_input("Collar Y (m)", value=0.0, step=0.1)
surface_elev = st.sidebar.number_input("Surface Elevation Z (m)", value=260.0, step=0.1)

length = st.sidebar.number_input("Planned/EOH length MD (m)", value=730.0, step=1.0, min_value=0.0)
target_depth = st.sidebar.number_input("Target depth MD anchor (m)", value=690.0, step=0.1, min_value=0.0)
step = st.sidebar.number_input("Computation step (m)", value=10.0, step=1.0, min_value=1.0)

# Structure plane defaults per your request
strike = st.sidebar.number_input("Structure strike (deg)", value=114.0, step=0.1)
dip_struct = st.sidebar.number_input("Structure dip (deg, +down)", value=58.0, step=0.1)

# Planned hole controls
st.sidebar.subheader("Planned hole")
plan_dip_in = st.sidebar.number_input("Planned dip at collar (deg, +down)", value=68.0, step=0.1)
plan_azim = st.sidebar.number_input("Planned azimuth at collar (deg)", value=355.0, step=0.1)
plan_lift = st.sidebar.number_input("Planned lift (deg per 100 m)", value=4.0, step=0.1)
plan_drift = st.sidebar.number_input("Planned drift (deg per 100 m)", value=2.0, step=0.1)

# Actual hole controls - manual survey table
st.sidebar.subheader("Actual hole - survey table")
st.markdown("Enter surveys with MD, dip (+down), azimuth. You can add or remove rows.")

default_rows = [
    {"MD": 0.0, "Dip": plan_dip_in, "Azimuth": plan_azim},
    {"MD": min(100.0, length), "Dip": plan_dip_in, "Azimuth": plan_azim},
]
survey_rows = st.data_editor(
    default_rows,
    num_rows="dynamic",
    key="survey_table",
    column_config={
        "MD": st.column_config.NumberColumn("MD", step=1.0, help="Measured depth"),
        "Dip": st.column_config.NumberColumn("Dip (+down)", step=0.1),
        "Azimuth": st.column_config.NumberColumn("Azimuth", step=0.1),
    }
)

st.sidebar.subheader("After last survey - remaining trend")
rem_lift = st.sidebar.number_input("Remaining lift (deg per 100 m)", value=5.0, step=0.1)
rem_drift = st.sidebar.number_input("Remaining drift (deg per 100 m)", value=2.0, step=0.1)

# ----------------- BUILD TRAJECTORIES -----------------

# Collar at real surface elevation. z is elevation (up).
collar = np.array([collar_x, collar_y, surface_elev])

def build_planned(collar_xyz, eoh_len, dip0, az0, lift_per_100, drift_per_100, step_m):
    """Incremental change per step for the planned hole."""
    pts = [collar_xyz.copy()]
    dip = dip0
    az = az0
    n_steps = int(max(1, np.ceil(eoh_len / step_m)))
    seg = eoh_len / n_steps
    for _ in range(n_steps):
        nxt = generate_segment(pts[-1], dip, az, seg)
        pts.append(nxt)
        dip += (lift_per_100 / 100.0) * seg
        az += (drift_per_100 / 100.0) * seg
    return np.array(pts)

def build_actual_from_surveys(collar_xyz, eoh_len, survey_list, step_m, rem_lift_per_100, rem_drift_per_100):
    """
    Build trajectory using user survey stations. Between stations linear interp dip and az.
    After last station, apply constant remaining lift and drift until EOH.
    survey_list is a list of dicts with keys MD, Dip, Azimuth.
    """
    # Validate and sort surveys
    clean = []
    for r in survey_list:
        try:
            md = float(r["MD"])
            dp = float(r["Dip"])
            az = float(r["Azimuth"])
            clean.append([md, dp, az])
        except Exception:
            continue
    if not clean:
        return np.array([collar_xyz])

    arr = np.array(clean, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]

    # Ensure first MD is 0 at collar
    if arr[0, 0] > 0.0:
        arr = np.vstack([[0.0, arr[0, 1], arr[0, 2]], arr])

    # Clip or extend to EOH
    if arr[-1, 0] < eoh_len:
        arr = np.vstack([arr, [eoh_len, arr[-1, 1], arr[-1, 2]]])
    elif arr[-1, 0] > eoh_len:
        arr[-1, 0] = eoh_len

    pts = [collar_xyz.copy()]
    md_curr = 0.0
    dip_curr = arr[0, 1]
    az_curr = arr[0, 2]

    # Walk through survey intervals
    for i in range(len(arr) - 1):
        md1, dip1, az1 = arr[i]
        md2, dip2, az2 = arr[i + 1]
        span = md2 - md1
        if span <= 0:
            continue
        n_steps = max(1, int(np.ceil(span / step_m)))
        seg = span / n_steps
        for k in range(n_steps):
            t0 = (k * seg) / span
            t1 = ((k + 1) * seg) / span
            dip_k = dip1 * (1 - t0) + dip2 * t0
            az_k = az1 * (1 - t0) + az2 * t0
            nxt = generate_segment(pts[-1], dip_k, az_k, seg)
            pts.append(nxt)
            md_curr += seg
            dip_curr = dip1 * (1 - t1) + dip2 * t1
            az_curr = az1 * (1 - t1) + az2 * t1

    # If EOH beyond last survey MD, continue with remaining lift/drift
    if md_curr < eoh_len:
        remaining = eoh_len - md_curr
        n_steps = max(1, int(np.ceil(remaining / step_m)))
        seg = remaining / n_steps
        for _ in range(n_steps):
            nxt = generate_segment(pts[-1], dip_curr, az_curr, seg)
            pts.append(nxt)
            dip_curr += (rem_lift_per_100 / 100.0) * seg
            az_curr += (rem_drift_per_100 / 100.0) * seg

    return np.array(pts)

# Build both traces
plan_pts = build_planned(collar, length, plan_dip_in, plan_azim, plan_lift, plan_drift, step)
act_pts = build_actual_from_surveys(collar, length, survey_rows, step, rem_lift, rem_drift)

x_plan, y_plan, z_plan = plan_pts[:, 0], plan_pts[:, 1], plan_pts[:, 2]
x_act,  y_act,  z_act  = act_pts[:, 0],  act_pts[:, 1],  act_pts[:, 2]

# Target plane anchor point from planned at target_depth
idx_target = int(np.clip(np.floor(target_depth / step), 0, len(plan_pts) - 1))
target_point = plan_pts[idx_target]
n_plane = plane_normal_from_strike_dip(strike, dip_struct)

# Pierce points
pp_plan = polyline_plane_pierce(plan_pts, n_plane, target_point)
pp_act  = polyline_plane_pierce(act_pts,  n_plane, target_point)

# ----------------- PLOTTING -----------------

fig, axs = plt.subplots(3, 1, figsize=(10, 18))

# Long section in the structure plane
ls_plan_s, ls_plan_d = project_long_section(x_plan, y_plan, z_plan, strike, dip_struct, origin=np.zeros(3))
ls_act_s,  ls_act_d  = project_long_section(x_act,  y_act,  z_act,  strike, dip_struct, origin=np.zeros(3))
axs[0].plot(ls_plan_s, ls_plan_d, label='Planned')
axs[0].plot(ls_act_s,  ls_act_d,  '--', label='Actual')

if pp_plan is not None:
    s_pp_plan, d_pp_plan = project_long_section(
        np.array([pp_plan[0]]), np.array([pp_plan[1]]), np.array([pp_plan[2]]),
        strike, dip_struct, origin=np.zeros(3)
    )
    axs[0].plot(s_pp_plan, d_pp_plan, 'x', color='red', markersize=10)
if pp_act is not None:
    s_pp_act, d_pp_act = project_long_section(
        np.array([pp_act[0]]), np.array([pp_act[1]]), np.array([pp_act[2]]),
        strike, dip_struct, origin=np.zeros(3)
    )
    axs[0].plot(s_pp_act, d_pp_act, 'x', color='purple', markersize=10)

if pp_plan is not None and pp_act is not None:
    axs[0].plot([s_pp_plan[0], s_pp_act[0]], [d_pp_plan[0], d_pp_act[0]], 'k--')
    dist_3d = np.linalg.norm(pp_plan - pp_act)
    midx = 0.5 * (s_pp_plan[0] + s_pp_act[0])
    midy = 0.5 * (d_pp_plan[0] + d_pp_act[0])
    axs[0].text(midx, midy, f"{dist_3d:.1f} m (3D)", ha='center', va='bottom', fontsize=12, fontweight='bold')
else:
    st.info("No long-section pierce point for one or both traces.")

add_end_arrow(axs[0], ls_plan_s, ls_plan_d, 'blue')
add_end_arrow(axs[0], ls_act_s,  ls_act_d,  'orange')
axs[0].set_xlabel('Along strike (m)')
axs[0].set_ylabel('Down dip (m)')
axs[0].invert_yaxis()  # draw down-dip downwards
axs[0].legend()
axs[0].set_title('Long section - structure plane')

# Cross section aligned with planned azimuth
sec_plan_x = project_cross_section(x_plan, y_plan, z_plan, plan_azim, np.zeros(3))
sec_act_x  = project_cross_section(x_act,  y_act,  z_act,  plan_azim, np.zeros(3))
axs[1].plot(sec_plan_x, z_plan, label='Planned')
axs[1].plot(sec_act_x,  z_act,  '--', label='Actual')

if pp_plan is not None:
    sec_pp_plan = project_cross_section(np.array([pp_plan[0]]), np.array([pp_plan[1]]), np.array([pp_plan[2]]), plan_azim, np.zeros(3))
    axs[1].plot(sec_pp_plan, pp_plan[2], 'x', color='red', markersize=10)
if pp_act is not None:
    sec_pp_act = project_cross_section(np.array([pp_act[0]]), np.array([pp_act[1]]), np.array([pp_act[2]]), plan_azim, np.zeros(3))
    axs[1].plot(sec_pp_act, pp_act[2], 'x', color='purple', markersize=10)

if pp_plan is not None and pp_act is not None:
    axs[1].plot([sec_pp_plan[0], sec_pp_act[0]], [pp_plan[2], pp_act[2]], 'k--')
else:
    st.info("No cross-section pierce line to draw because a pierce point is missing.")

# Target plane guide line through planned pierce point if available
if pp_plan is not None:
    dip_rad = np.radians(dip_struct)
    xs = np.linspace(sec_pp_plan[0] - 200.0, sec_pp_plan[0] + 200.0, 20)
    # For z as elevation, line slope vs section distance is -tan(dip)
    zs = pp_plan[2] - (xs - sec_pp_plan[0]) * np.tan(dip_rad)
    axs[1].plot(xs, zs, 'g-', label='Target plane')

axs[1].set_xlabel('Section distance (m)')
axs[1].set_ylabel('Elevation (m)')
axs[1].invert_yaxis()  # draw downwards with depth
axs[1].legend()
axs[1].set_title('Cross section - along drill azimuth')

# Plan view
axs[2].plot(x_plan, y_plan, label='Planned')
axs[2].plot(x_act,  y_act,  '--', label='Actual')
if pp_plan is not None:
    axs[2].plot(pp_plan[0], pp_plan[1], 'x', color='red', markersize=10)
if pp_act is not None:
    axs[2].plot(pp_act[0], pp_act[1], 'x', color='purple', markersize=10)
if pp_plan is not None and pp_act is not None:
    axs[2].plot([pp_plan[0], pp_act[0]], [pp_plan[1], pp_act[1]], 'k--')

# Draw a strike-line through planned pierce point as a plane cue
if pp_plan is not None:
    plane_extent = 300.0
    strike_rad = np.radians(strike)
    dx = np.sin(strike_rad) * plane_extent
    dy = np.cos(strike_rad) * plane_extent
    axs[2].plot([pp_plan[0] - dx, pp_plan[0] + dx],
                [pp_plan[1] - dy, pp_plan[1] + dy], 'g-', label='Target plane')

add_end_arrow(axs[2], x_plan, y_plan, 'blue')
add_end_arrow(axs[2], x_act,  y_act,  'orange')
axs[2].set_xlabel('X (m)')
axs[2].set_ylabel('Y (m)')
axs[2].legend()
axs[2].set_title('Plan view')

for ax in axs:
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True)

plt.tight_layout()
st.pyplot(fig)



