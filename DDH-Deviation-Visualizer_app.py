import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ----------------- GEOMETRY -----------------

def generate_segment(xyz0, dip_deg, az_deg, seg_len):
    # x east, y north, z elevation (up). Positive dip = down, so dz is negative.
    dip = np.radians(dip_deg)
    az  = np.radians(az_deg)
    dx = seg_len * np.sin(az) * np.cos(dip)
    dy = seg_len * np.cos(az) * np.cos(dip)
    dz = -seg_len * np.sin(dip)
    return xyz0 + np.array([dx, dy, dz])

def plane_normal_from_strike_dip(strike_deg, dip_deg):
    # strike clockwise from north, dip positive down
    dipdir = (strike_deg + 90.0) % 360.0
    dip = np.radians(dip_deg)
    dipdir = np.radians(dipdir)
    nx = np.sin(dip) * np.sin(dipdir)
    ny = np.sin(dip) * np.cos(dipdir)
    nz = -np.cos(dip)
    return np.array([nx, ny, nz])

def segment_plane_intersect(p1, p2, n, p0):
    # intersect segment p1->p2 with plane n·(r - p0)=0
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
    for i in range(len(pts) - 1):
        hit = segment_plane_intersect(pts[i], pts[i+1], n, p0)
        if hit is not None:
            return hit
    return None

# ----------------- SECTION MATH -----------------

def section_basis(azimuth_deg):
    # vertical section aligned with azimuth
    az = np.radians(azimuth_deg)
    u_sec = np.array([np.sin(az), np.cos(az), 0.0])  # along-section horizontal axis
    k_up  = np.array([0.0, 0.0, 1.0])                # vertical axis
    n_sec = np.cross(u_sec, k_up)                    # horizontal normal to the section plane
    n_sec /= np.linalg.norm(n_sec)
    return u_sec, k_up, n_sec

def project_to_section(points_xyz, azimuth_deg, origin_xyz):
    # orthogonal projection to section plane, then 2D coords (s, z) and off-section distance
    u_sec, k_up, n_sec = section_basis(azimuth_deg)
    d = points_xyz - origin_xyz
    s = d @ u_sec                    # along-section distance from origin
    z = points_xyz[:, 2]             # elevation stays unchanged for a vertical plane
    off = d @ n_sec                  # signed perpendicular distance from the section plane
    return s, z, off

def target_plane_line_in_section(strike_deg, dip_deg, section_az_deg, sec_origin_xyz, p0_on_plane, s_span=600.0):
    """
    Analytic intersection of target plane with the vertical section plane.
    Points in section plane: r = sec_origin + s*u_sec + z*k_up
    Condition to be on target plane: n·(r - p0_on_plane) = 0
    Solve for z(s):  (n·u_sec)*s + (n·k_up)*z + n·(sec_origin - p0) = 0
                     z(s) = -[a*s + c]/b
    """
    n = plane_normal_from_strike_dip(strike_deg, dip_deg)
    u_sec, k_up, _ = section_basis(section_az_deg)
    a = float(np.dot(n, u_sec))
    b = float(np.dot(n, k_up))      # equals n_z
    c = float(np.dot(n, sec_origin_xyz - p0_on_plane))

    s_vals = np.array([-s_span, s_span])
    if np.isclose(b, 0.0):  # plane nearly vertical - fallback to a gentle line
        z_vals = np.full_like(s_vals, p0_on_plane[2])
    else:
        z_vals = -(a * s_vals + c) / b
    return s_vals, z_vals

# ----------------- BUILD TRAJECTORIES -----------------

def build_planned(collar_xyz, eoh_len, dip0, az0, lift_per_100, drift_per_100, step_m):
    pts = [collar_xyz.copy()]
    dip, az = dip0, az0
    n_steps = int(max(1, np.ceil(eoh_len / step_m)))
    seg = eoh_len / n_steps
    for _ in range(n_steps):
        nxt = generate_segment(pts[-1], dip, az, seg)
        pts.append(nxt)
        dip += (lift_per_100 / 100.0) * seg
        az  += (drift_per_100 / 100.0) * seg
    return np.array(pts)

def build_actual_from_surveys(collar_xyz, eoh_len, survey_rows, step_m, rem_lift_per_100, rem_drift_per_100):
    clean = []
    for r in survey_rows:
        try:
            clean.append([float(r["MD"]), float(r["Dip"]), float(r["Azimuth"])])
        except Exception:
            pass
    if not clean:
        return np.array([collar_xyz])

    arr = np.array(clean, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]

    # ensure first at collar
    if arr[0, 0] > 0.0:
        arr = np.vstack([[0.0, arr[0, 1], arr[0, 2]], arr])
    # cap last at EOH if needed
    if arr[-1, 0] > eoh_len:
        arr[-1, 0] = eoh_len

    pts = [collar_xyz.copy()]
    md_curr = 0.0

    # interpolate between surveys
    for i in range(len(arr) - 1):
        md1, dip1, az1 = arr[i]
        md2, dip2, az2 = arr[i + 1]
        if md2 <= md1:
            continue
        span = md2 - md1
        n_steps = max(1, int(np.ceil(span / step_m)))
        seg = span / n_steps
        for k in range(n_steps):
            t0 = (k * seg) / span
            dip_k = dip1 * (1 - t0) + dip2 * t0
            az_k  = az1 * (1 - t0) + az2 * t0
            pts.append(generate_segment(pts[-1], dip_k, az_k, seg))
            md_curr += seg

    # continue with remaining lift/drift
    if md_curr < eoh_len:
        remaining = eoh_len - md_curr
        n_steps = max(1, int(np.ceil(remaining / step_m)))
        seg = remaining / n_steps
        dip_curr = arr[-1, 1]
        az_curr  = arr[-1, 2]
        for _ in range(n_steps):
            pts.append(generate_segment(pts[-1], dip_curr, az_curr, seg))
            dip_curr += (rem_lift_per_100 / 100.0) * seg
            az_curr  += (rem_drift_per_100 / 100.0) * seg

    return np.array(pts)

# ----------------- STREAMLIT UI -----------------

st.title("DDH Deviation Visualizer - true vertical section")

st.sidebar.header("Inputs")

# Collar and globals
collar_x = st.sidebar.number_input("Collar X (m)", value=0.0, step=0.1)
collar_y = st.sidebar.number_input("Collar Y (m)", value=0.0, step=0.1)
surface_elev = st.sidebar.number_input("Surface Elevation Z (m)", value=260.0, step=0.1)

length = st.sidebar.number_input("Planned - EOH length MD (m)", value=730.0, step=1.0, min_value=0.0)
target_depth = st.sidebar.number_input("Target MD anchor for plane (m)", value=690.0, step=0.1, min_value=0.0)
step = st.sidebar.number_input("Computation step (m)", value=10.0, step=1.0, min_value=1.0)

# Section controls
section_half_width = st.sidebar.number_input("Section half width for clipping (m)", value=200.0, step=10.0, min_value=0.0)

# Structure plane defaults
strike = st.sidebar.number_input("Structure strike (deg)", value=114.0, step=0.1)
dip_struct = st.sidebar.number_input("Structure dip (deg, +down)", value=58.0, step=0.1)

# Planned hole
st.sidebar.subheader("Planned hole")
plan_dip_in = st.sidebar.number_input("Planned dip at collar (deg, +down)", value=68.0, step=0.1)
plan_azim = st.sidebar.number_input("Planned azimuth at collar (deg)", value=355.0, step=0.1)
plan_lift = st.sidebar.number_input("Planned lift (deg per 100 m)", value=4.0, step=0.1)
plan_drift = st.sidebar.number_input("Planned drift (deg per 100 m)", value=2.0, step=0.1)

# Actual hole surveys
st.sidebar.subheader("Actual hole - survey table")
st.markdown("Enter surveys with MD, dip (+down), azimuth. Remaining lift/drift applies after the last row.")
default_rows = [
    {"MD": 0.0, "Dip": plan_dip_in, "Azimuth": plan_azim},
    {"MD": min(100.0, length), "Dip": plan_dip_in, "Azimuth": plan_azim},
]
survey_rows = st.data_editor(
    default_rows,
    num_rows="dynamic",
    key="survey_table",
    column_config={
        "MD": st.column_config.NumberColumn("MD", step=1.0),
        "Dip": st.column_config.NumberColumn("Dip (+down)", step=0.1),
        "Azimuth": st.column_config.NumberColumn("Azimuth", step=0.1),
    }
)

st.sidebar.subheader("After last survey - remaining trend")
rem_lift = st.sidebar.number_input("Remaining lift (deg per 100 m)", value=5.0, step=0.1)
rem_drift = st.sidebar.number_input("Remaining drift (deg per 100 m)", value=2.0, step=0.1)

# ----------------- BUILD AND PROJECT -----------------

collar = np.array([collar_x, collar_y, surface_elev])

plan_pts = build_planned(collar, length, plan_dip_in, plan_azim, plan_lift, plan_drift, step)
act_pts  = build_actual_from_surveys(collar, length, survey_rows, step, rem_lift, rem_drift)

# target plane anchored at planned point near target MD
idx_target = int(np.clip(np.floor(target_depth / step), 0, len(plan_pts) - 1))
target_point = plan_pts[idx_target]
n_plane = plane_normal_from_strike_dip(strike, dip_struct)

# pierce points
pp_plan = polyline_plane_pierce(plan_pts, n_plane, target_point)
pp_act  = polyline_plane_pierce(act_pts,  n_plane, target_point)

# section projection
sec_origin = collar.copy()  # section plane goes through the collar
s_plan, z_plan, off_plan = project_to_section(plan_pts, plan_azim, sec_origin)
s_act,  z_act,  off_act  = project_to_section(act_pts,  plan_azim, sec_origin)

# clip to section half width
mask_plan = np.abs(off_plan) <= section_half_width
mask_act  = np.abs(off_act)  <= section_half_width

# pierce points in section
def project_point_to_section(p):
    s, z, off = project_to_section(p.reshape(1,3), plan_azim, sec_origin)
    return float(s[0]), float(z[0]), float(off[0])

pp_plan_szo = project_point_to_section(pp_plan) if pp_plan is not None else None
pp_act_szo  = project_point_to_section(pp_act)  if pp_act  is not None else None

# target plane line inside the section
s_line, z_line = target_plane_line_in_section(
    strike, dip_struct, plan_azim, sec_origin, p0_on_plane=target_point, s_span=1000.0
)

# ----------------- PLOTTING -----------------

fig, axs = plt.subplots(3, 1, figsize=(10, 18))

# 1) Long section in the structure plane - keep as before for context
def project_long_section(x, y, z, strike, dip, origin):
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

ls_plan_s, ls_plan_d = project_long_section(plan_pts[:,0], plan_pts[:,1], plan_pts[:,2], strike, dip_struct, origin=np.zeros(3))
ls_act_s,  ls_act_d  = project_long_section(act_pts[:,0],  act_pts[:,1],  act_pts[:,2],  strike, dip_struct, origin=np.zeros(3))
axs[0].plot(ls_plan_s, ls_plan_d, label='Planned')
axs[0].plot(ls_act_s,  ls_act_d,  '--', label='Actual')

if pp_plan is not None:
    s_ppp, d_ppp = project_long_section(
        np.array([pp_plan[0]]), np.array([pp_plan[1]]), np.array([pp_plan[2]]),
        strike, dip_struct, origin=np.zeros(3)
    )
    axs[0].plot(s_ppp, d_ppp, 'x', color='red', markersize=10)
if pp_act is not None:
    s_ppa, d_ppa = project_long_section(
        np.array([pp_act[0]]), np.array([pp_act[1]]), np.array([pp_act[2]]),
        strike, dip_struct, origin=np.zeros(3)
    )
    axs[0].plot(s_ppa, d_ppa, 'x', color='purple', markersize=10)

if pp_plan is not None and pp_act is not None:
    axs[0].plot([s_ppp[0], s_ppa[0]], [d_ppp[0], d_ppa[0]], 'k--')
else:
    st.info("No long-section pierce point for one or both traces.")

axs[0].set_xlabel('Along strike (m)')
axs[0].set_ylabel('Down dip (m)')
axs[0].invert_yaxis()
axs[0].legend()
axs[0].set_title('Long section - structure plane')

# 2) True vertical section along planned azimuth
axs[1].plot(s_plan[mask_plan], z_plan[mask_plan], label='Planned')
axs[1].plot(s_act[mask_act],   z_act[mask_act],   '--', label='Actual')

# pierce points in section
if pp_plan_szo is not None and abs(pp_plan_szo[2]) <= section_half_width:
    axs[1].plot(pp_plan_szo[0], pp_plan_szo[1], 'x', color='red', markersize=10)
if pp_act_szo is not None and abs(pp_act_szo[2]) <= section_half_width:
    axs[1].plot(pp_act_szo[0], pp_act_szo[1], 'x', color='purple', markersize=10)
if pp_plan_szo is not None and pp_act_szo is not None:
    if abs(pp_plan_szo[2]) <= section_half_width and abs(pp_act_szo[2]) <= section_half_width:
        axs[1].plot([pp_plan_szo[0], pp_act_szo[0]], [pp_plan_szo[1], pp_act_szo[1]], 'k--')

# target plane exact trace within the section
axs[1].plot(s_line, z_line, 'g-', label='Target plane')

axs[1].set_xlabel('Section distance from collar (m)')
axs[1].set_ylabel('Elevation (m)')
axs[1].invert_yaxis()
axs[1].legend()
axs[1].set_title('Cross section - along planned azimuth')

# 3) Plan view
axs[2].plot(plan_pts[:,0], plan_pts[:,1], label='Planned')
axs[2].plot(act_pts[:,0],  act_pts[:,1],  '--', label='Actual')
if pp_plan is not None:
    axs[2].plot(pp_plan[0], pp_plan[1], 'x', color='red', markersize=10)
if pp_act is not None:
    axs[2].plot(pp_act[0], pp_act[1], 'x', color='purple', markersize=10)
axs[2].set_xlabel('X (m)')
axs[2].set_ylabel('Y (m)')
axs[2].legend()
axs[2].set_title('Plan view')

for ax in axs:
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True)

plt.tight_layout()
st.pyplot(fig)

# Optional diagnostics
st.caption(
    f"Max off-section distance - planned: {np.nanmax(np.abs(off_plan)):.1f} m, actual: {np.nanmax(np.abs(off_act)):.1f} m. "
    f"Points beyond ±{section_half_width:.0f} m are clipped from the section."
)

