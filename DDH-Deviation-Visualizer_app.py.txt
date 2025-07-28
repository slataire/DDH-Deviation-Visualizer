import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ----- Functions -----
def generate_trajectory(collar, length, dip, azimuth, lift_per_100, drift_per_100, step=10):
    x, y, z = [collar[0]], [collar[1]], [collar[2]]
    n_steps = int(length / step)
    dip_curr = dip
    az_curr = azimuth
    for _ in range(n_steps):
        dip_rad = np.radians(dip_curr)
        az_rad = np.radians(az_curr)
        dx = step * np.sin(az_rad) * np.cos(dip_rad)
        dy = step * np.cos(az_rad) * np.cos(dip_rad)
        dz = step * np.sin(dip_rad)
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
        z.append(z[-1] + dz)
        dip_curr += (lift_per_100 / 100) * step
        az_curr += (drift_per_100 / 100) * step
    return np.array(x), np.array(y), np.array(z)

def plane_normal(strike, dip):
    dipdir = (strike + 90) % 360
    dip_rad = np.radians(dip)
    dipdir_rad = np.radians(dipdir)
    nx = np.sin(dip_rad) * np.sin(dipdir_rad)
    ny = np.sin(dip_rad) * np.cos(dipdir_rad)
    nz = -np.cos(dip_rad)
    return np.array([nx, ny, nz])

def find_pierce_point(x, y, z, n, p0):
    p0 = np.array(p0)
    for i in range(len(x)-1):
        r1 = np.array([x[i], y[i], z[i]])
        r2 = np.array([x[i+1], y[i+1], z[i+1]])
        d1 = np.dot(n, r1 - p0)
        d2 = np.dot(n, r2 - p0)
        if d1 * d2 <= 0:
            t = d1 / (d1 - d2)
            return r1 + t * (r2 - r1)
    return None

def limits_with_margin(data, frac=0.05):
    lo, hi = data.min(), data.max()
    margin = (hi - lo) * frac
    return lo - margin, hi + margin

def project_to_plane(x, y, z, strike, dip, origin):
    strike_rad = np.radians(strike)
    dip_rad = np.radians(dip)
    dipdir_rad = np.radians((strike + 90) % 360)
    u_strike = np.array([np.sin(strike_rad), np.cos(strike_rad), 0])
    u_dip = np.array([
        np.sin(dipdir_rad) * np.cos(dip_rad),
        np.cos(dipdir_rad) * np.cos(dip_rad),
        -np.sin(dip_rad)
    ])
    coords = np.vstack([x, y, z]).T - origin
    s = coords @ u_strike
    d = coords @ u_dip
    return s, d

def project_to_section(x, y, z, azim, origin):
    azim_rad = np.radians(azim)
    u_sec = np.array([np.sin(azim_rad), np.cos(azim_rad), 0])
    coords = np.vstack([x, y, z]).T - origin
    sec_x = coords @ u_sec
    return sec_x

def add_end_arrow(ax, x, y, color='black'):
    ax.quiver(x[-2], y[-2],
              x[-1]-x[-2], y[-1]-y[-2],
              angles='xy', scale_units='xy', scale=1,
              width=0.01, color=color)

# ----- Streamlit UI -----
st.title("3D Drillhole Visualization App")

st.sidebar.header("Input Parameters")

surface_elev = st.sidebar.number_input("Surface Elevation (m)", value=260.0)
length = st.sidebar.number_input("Drillhole Length (m)", value=600.0)
target_depth = st.sidebar.number_input("Target Depth (MD, m)", value=400.0)

strike = st.sidebar.number_input("Structure Strike (deg)", value=300.0)
dip = st.sidebar.number_input("Structure Dip (deg)", value=70.0)

plan_dip = st.sidebar.number_input("Planned Dip (deg)", value=-70.0)
plan_azim = st.sidebar.number_input("Planned Azimuth (deg)", value=0.0)
plan_lift = st.sidebar.number_input("Planned Lift (deg/100m)", value=1.5)
plan_drift = st.sidebar.number_input("Planned Drift (deg/100m)", value=1.0)

act_dip = st.sidebar.number_input("Actual Dip (deg)", value=-70.0)
act_azim = st.sidebar.number_input("Actual Azimuth (deg)", value=0.0)
act_lift = st.sidebar.number_input("Actual Lift (deg/100m)", value=5.0)
act_drift = st.sidebar.number_input("Actual Drift (deg/100m)", value=2.0)

# ----- Calculations -----
collar = (0, 0, 0)
x_plan, y_plan, z_plan = generate_trajectory(collar, length, plan_dip, plan_azim, plan_lift, plan_drift)
x_act, y_act, z_act = generate_trajectory(collar, length, act_dip, act_azim, act_lift, act_drift)

elev_plan = surface_elev - z_plan
elev_act = surface_elev - z_act

i_target = int(target_depth // 10)
target_point = (x_plan[i_target], y_plan[i_target], z_plan[i_target])
n = plane_normal(strike, dip)

pp_plan = find_pierce_point(x_plan, y_plan, z_plan, n, target_point)
pp_act = find_pierce_point(x_act, y_act, z_act, n, target_point)

dist_3d = np.linalg.norm(pp_plan - pp_act)
elev_pp_plan = surface_elev - pp_plan[2]
elev_pp_act = surface_elev - pp_act[2]

origin = np.array([0, 0, 0])

# Projections
ls_plan_s, ls_plan_d = project_to_plane(x_plan, y_plan, z_plan, strike, dip, origin)
ls_act_s, ls_act_d = project_to_plane(x_act, y_act, z_act, strike, dip, origin)
ls_pp_plan_s, ls_pp_plan_d = project_to_plane([pp_plan[0]], [pp_plan[1]], [pp_plan[2]], strike, dip, origin)
ls_pp_act_s, ls_pp_act_d = project_to_plane([pp_act[0]], [pp_act[1]], [pp_act[2]], strike, dip, origin)

sec_plan_x = project_to_section(x_plan, y_plan, z_plan, plan_azim, origin)
sec_act_x = project_to_section(x_act, y_act, z_act, plan_azim, origin)
sec_pp_plan_x = project_to_section([pp_plan[0]], [pp_plan[1]], [pp_plan[2]], plan_azim, origin)
sec_pp_act_x = project_to_section([pp_act[0]], [pp_act[1]], [pp_act[2]], plan_azim, origin)

# ----- Plotting -----
fig, axs = plt.subplots(1, 3, figsize=(20, 7))

# Long section
axs[0].plot(ls_plan_s, ls_plan_d, label='Planned')
axs[0].plot(ls_act_s, ls_act_d, '--', label='Actual')
axs[0].plot(ls_pp_plan_s, ls_pp_plan_d, 'x', color='red', markersize=10)
axs[0].plot(ls_pp_act_s, ls_pp_act_d, 'x', color='purple', markersize=10)
axs[0].plot([ls_pp_plan_s[0], ls_pp_act_s[0]], [ls_pp_plan_d[0], ls_pp_act_d[0]], 'k--')
add_end_arrow(axs[0], ls_plan_s, ls_plan_d, 'blue')
add_end_arrow(axs[0], ls_act_s, ls_act_d, 'orange')

mid_x = 0.5 * (ls_pp_plan_s[0] + ls_pp_act_s[0])
mid_y = 0.5 * (ls_pp_plan_d[0] + ls_pp_act_d[0])
axs[0].text(mid_x, mid_y, f"{dist_3d:.1f} m (3D)", ha='center', va='bottom', fontsize=14, fontweight='bold')

axs[0].set_xlabel('Along Strike (m)')
axs[0].set_ylabel('Down Dip (m)')
axs[0].invert_yaxis()
axs[0].legend()
axs[0].set_title('Long Section')

# Cross section
axs[1].plot(sec_plan_x, elev_plan, label='Planned')
axs[1].plot(sec_act_x, elev_act, '--', label='Actual')
axs[1].plot(sec_pp_plan_x, elev_pp_plan, 'x', color='red', markersize=10)
axs[1].plot(sec_pp_act_x, elev_pp_act, 'x', color='purple', markersize=10)
axs[1].plot([sec_pp_plan_x[0], sec_pp_act_x[0]], [elev_pp_plan, elev_pp_act], 'k--')
add_end_arrow(axs[1], sec_plan_x, elev_plan, 'blue')
add_end_arrow(axs[1], sec_act_x, elev_act, 'orange')

dip_rad = np.radians(dip)
xs = np.linspace(sec_pp_plan_x[0] - 200, sec_pp_plan_x[0] + 200, 20)
zs = elev_pp_plan - (xs - sec_pp_plan_x[0]) * np.tan(dip_rad)
axs[1].plot(xs, zs, 'g-', label='Target Plane')

axs[1].set_xlabel('Section Distance (m)')
axs[1].set_ylabel('Elevation (m)')
axs[1].invert_yaxis()
axs[1].legend()
axs[1].set_title('Cross Section')

# Plan view
axs[2].plot(x_plan, y_plan, label='Planned')
axs[2].plot(x_act, y_act, '--', label='Actual')
axs[2].plot(pp_plan[0], pp_plan[1], 'x', color='red', markersize=10)
axs[2].plot(pp_act[0], pp_act[1], 'x', color='purple', markersize=10)
axs[2].plot([pp_plan[0], pp_act[0]], [pp_plan[1], pp_act[1]], 'k--')
add_end_arrow(axs[2], x_plan, y_plan, 'blue')
add_end_arrow(axs[2], x_act, y_act, 'orange')

plane_extent = 300
strike_rad = np.radians(strike)
dx = np.sin(strike_rad) * plane_extent
dy = np.cos(strike_rad) * plane_extent
axs[2].plot([pp_plan[0]-dx, pp_plan[0]+dx], [pp_plan[1]-dy, pp_plan[1]+dy], 'g-', label='Target Plane')

axs[2].set_xlabel('X (m)')
axs[2].set_ylabel('Y (m)')
axs[2].legend()
axs[2].set_title('Plan View')

plt.tight_layout()
st.pyplot(fig)
