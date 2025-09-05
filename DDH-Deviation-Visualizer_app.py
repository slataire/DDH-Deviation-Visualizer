import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="DDH deviation visualizer", layout="wide")

# -----------------
# Helpers
# -----------------
def wrap_az(az_deg: float) -> float:
    az = az_deg % 360.0
    if az < 0:
        az += 360.0
    return az

def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))

def min_curvature_path_dip_h(stations, collar_xyz=np.array([0.0, 0.0, 0.0])):
    """
    Minimum curvature path.
    stations: list of dicts with MD, Azimuth, Angle
      - Azimuth deg clockwise from North
      - Angle is dip-from-horizontal in degrees, negative means down
    Coordinates: X East, Y North, Z Up
    """
    if not stations:
        return np.array([collar_xyz[0]]), np.array([collar_xyz[1]]), np.array([collar_xyz[2]]), np.array([0.0])

    stations = sorted(stations, key=lambda d: float(d["MD"]))
    MDs = np.array([float(s["MD"]) for s in stations], dtype=float)
    AZs = np.deg2rad([wrap_az(float(s["Azimuth"])) for s in stations])
    DIP = np.array([float(s["Angle"]) for s in stations], dtype=float)  # signed dip-from-horizontal

    # Convert to inclination from vertical (0 vertical, 90 horizontal), ignore sign here
    INC = np.deg2rad(90.0 - np.abs(DIP))
    # Sign for vertical component: negative dip points down -> sign_z = -1
    SGN = np.where(DIP <= 0.0, -1.0, 1.0)

    X = [collar_xyz[0]]
    Y = [collar_xyz[1]]
    Z = [collar_xyz[2]]
    MD_path = [MDs[0]]

    for i in range(1, len(MDs)):
        dMD = MDs[i] - MDs[i-1]
        if dMD <= 0:
            continue
        inc1, inc2 = INC[i-1], INC[i]
        az1, az2 = AZs[i-1], AZs[i]
        s1, s2 = SGN[i-1], SGN[i]

        # standard minimum curvature dogleg
        cos_dog = np.sin(inc1)*np.sin(inc2)*np.cos(az2-az1) + np.cos(inc1)*np.cos(inc2)
        cos_dog = float(np.clip(cos_dog, -1.0, 1.0))
        dog = np.arccos(cos_dog)
        RF = 1.0 if dog < 1e-12 else (2.0/dog)*np.tan(dog/2.0)

        dN = 0.5*dMD*(np.sin(inc1)*np.cos(az1) + np.sin(inc2)*np.cos(az2))*RF
        dE = 0.5*dMD*(np.sin(inc1)*np.sin(az1) + np.sin(inc2)*np.sin(az2))*RF
        # vertical uses signed cos terms so that negative dip points down
        dZ = 0.5*dMD*(s1*np.cos(inc1) + s2*np.cos(inc2))*RF

        # add to path
        X.append(X[-1] + dE)
        Y.append(Y[-1] + dN)
        Z.append(Z[-1] + dZ)
        MD_path.append(MDs[i])

    return np.array(X), np.array(Y), np.array(Z), np.array(MD_path)

def make_planned_stations(length_m, step_m, az0_deg, dip0_deg, lift_per100, drift_per100):
    """
    Build planned stations.
    - Angle is dip-from-horizontal, negative down.
    - Positive lift makes the hole dip more downward, so we subtract lift from the angle value.
      Example: dip = -60; lift +2 deg/100 m -> dip becomes -62 after 100 m.
    - Drift adds to azimuth clockwise from North.
    """
    stations = []
    md = 0.0
    az = wrap_az(az0_deg)
    dip = clamp(dip0_deg, -90.0, 90.0)
    stations.append({"MD": md, "Azimuth": az, "Angle": dip})
    while md < length_m - 1e-9:
        d = min(step_m, length_m - md)
        md += d
        az = wrap_az(az + drift_per100*(d/100.0))
        dip = clamp(dip - lift_per100*(d/100.0), -90.0, 90.0)  # subtract to increase downward dip
        stations.append({"MD": md, "Azimuth": az, "Angle": dip})
    return stations

def extend_actual_stations(stations, to_depth_m, step_m, lift_per100, drift_per100):
    """
    Extend actual stations from last survey to to_depth_m using remaining average lift and drift.
    """
    if not stations:
        return stations
    stations = sorted(stations, key=lambda d: float(d["MD"]))
    last = stations[-1].copy()
    md = float(last["MD"])
    az = float(last["Azimuth"])
    dip = float(last["Angle"])
    while md < to_depth_m - 1e-9:
        d = min(step_m, to_depth_m - md)
        md += d
        az = wrap_az(az + drift_per100*(d/100.0))
        dip = clamp(dip - lift_per100*(d/100.0), -90.0, 90.0)
        stations.append({"MD": md, "Azimuth": az, "Angle": dip})
    return stations

def derive_lift_drift_last3(stations):
    """
    Compute lift and drift per 100 m from last 3 surveys.
    Returns tuple (lift_deg_per100m, drift_deg_per100m) or (None, None).
    Angle is dip-from-horizontal signed. Drift is based on unwrapped azimuth.
    """
    if len(stations) < 3:
        return None, None
    sta = sorted(stations, key=lambda d: float(d["MD"]))[-3:]
    MD = np.array([float(s["MD"]) for s in sta], dtype=float)
    AZ = np.array([float(s["Azimuth"]) for s in sta], dtype=float)
    DIP = np.array([float(s["Angle"]) for s in sta], dtype=float)

    # unwrap azimuth to avoid wrap at 360
    AZu = np.unwrap(np.deg2rad(AZ))
    drift_deg_per_m = np.rad2deg(np.polyfit(MD, AZu, 1)[0])
    lift_deg_per_m = np.polyfit(MD, DIP, 1)[0] * (-1.0)  # subtract meaning: positive lift makes dip more negative
    return float(lift_deg_per_m*100.0), float(drift_deg_per_m*100.0)

def strike_dip_to_axes(strike_deg, dip_deg_signed):
    """
    Plane axes for given strike and dip-from-horizontal (signed, negative down).
    Returns:
      s_hat - unit along strike (horizontal)
      d_hat - unit down-dip within plane (points downward for negative dip)
      n_hat - unit normal
    """
    strike = np.deg2rad(wrap_az(strike_deg))
    dip_abs = np.deg2rad(abs(dip_deg_signed))

    s_hat = np.array([np.sin(strike), np.cos(strike), 0.0])
    dipdir = strike + np.pi/2.0
    d_hat = np.array([np.sin(dipdir)*np.cos(dip_abs), np.cos(dipdir)*np.cos(dip_abs), -np.sin(dip_abs)])
    # normal
    n_hat = np.cross(s_hat, d_hat)
    # normalize
    s_hat /= np.linalg.norm(s_hat)
    d_hat /= np.linalg.norm(d_hat)
    n_hat /= np.linalg.norm(n_hat)
    return s_hat, d_hat, n_hat

def plane_z_at_xy(x, y, P0, n_hat):
    nx, ny, nz = n_hat
    if abs(nz) < 1e-12:
        return None
    x0, y0, z0 = P0
    return z0 - (nx*(x - x0) + ny*(y - y0))/nz

def segment_plane_intersection(p0, p1, P0, n_hat):
    u = p1 - p0
    denom = np.dot(n_hat, u)
    num = np.dot(n_hat, P0 - p0)
    if abs(denom) < 1e-12:
        if abs(np.dot(n_hat, p0 - P0)) < 1e-9:
            return p0.copy()
        return None
    t = num/denom
    if t < -1e-9 or t > 1.0 + 1e-9:
        return None
    t = float(clamp(t, 0.0, 1.0))
    return p0 + t*u

def find_plane_intersection(points_xyz, P0, n_hat):
    for i in range(1, len(points_xyz)):
        p = segment_plane_intersection(points_xyz[i-1], points_xyz[i], P0, n_hat)
        if p is not None:
            return p
    return None

def project_to_axis(points_xyz, origin_xyz, axis_hat):
    return (points_xyz - origin_xyz) @ axis_hat

# -----------------
# UI
# -----------------
st.title("Diamond drillhole deviation - planned vs actual")

# Inputs
col_top1, col_top2, col_top3 = st.columns(3)
with col_top1:
    plan_len = st.number_input("Planned length m", value=500.0, step=10.0, min_value=1.0)
    step_m = st.number_input("Computation step m", value=10.0, step=1.0, min_value=1.0)
with col_top2:
    plan_az0 = st.number_input("Planned start azimuth deg", value=90.0, step=1.0)
    plan_dip0 = st.number_input("Planned start dip-from-horizontal deg (negative down)", value=-60.0, step=1.0, help="Negative dip points down")
with col_top3:
    plan_lift = st.number_input("Planned lift deg per 100 m", value=2.0, step=0.1, help="Positive increases downward dip magnitude")
    plan_drift = st.number_input("Planned drift deg per 100 m", value=1.0, step=0.1)

# Build planned
planned_stations = make_planned_stations(plan_len, step_m, plan_az0, plan_dip0, plan_lift, plan_drift)
px, py, pz, pmd = min_curvature_path_dip_h(planned_stations)

# Actual surveys
st.subheader("Actual surveys")
method = st.radio("Provide surveys via", ["Manual entry", "CSV upload"], horizontal=True)
if method == "CSV upload":
    up = st.file_uploader("Upload CSV with columns: MD, Azimuth, Angle (dip-from-horizontal, negative down)", type=["csv"])
    if up is not None:
        df_in = pd.read_csv(up)
    else:
        df_in = pd.DataFrame(columns=["MD", "Azimuth", "Angle"])
else:
    df_in = st.data_editor(
        pd.DataFrame(
            [
                {"MD": 0.0, "Azimuth": plan_az0, "Angle": plan_dip0},
                {"MD": 50.0, "Azimuth": plan_az0, "Angle": plan_dip0},
            ]
        ),
        num_rows="dynamic",
        use_container_width=True,
    )

actual_stations_base = [
    {"MD": float(r["MD"]), "Azimuth": float(r["Azimuth"]), "Angle": float(r["Angle"])}
    for _, r in pd.DataFrame(df_in).dropna(subset=["MD", "Azimuth", "Angle"]).iterrows()
]

# Suggested lift and drift from last 3
sug_lift, sug_drift = derive_lift_drift_last3(actual_stations_base) if len(actual_stations_base) >= 3 else (None, None)

st.markdown("#### Remaining average lift and drift to extend actual to planned depth")
cols_sug = st.columns(2)
with cols_sug[0]:
    if sug_lift is not None:
        st.caption(f"Suggested lift from last 3 surveys: {sug_lift:.2f} deg per 100 m")
    else:
        st.caption("Suggested lift needs at least 3 surveys")
with cols_sug[1]:
    if sug_drift is not None:
        st.caption(f"Suggested drift from last 3 surveys: {sug_drift:.2f} deg per 100 m")
    else:
        st.caption("Suggested drift needs at least 3 surveys")

col_ext1, col_ext2 = st.columns(2)
with col_ext1:
    rem_lift = st.number_input(
        "Remaining avg lift deg per 100 m after last survey",
        value=float(np.round(sug_lift, 2)) if sug_lift is not None else 2.0,
        step=0.1,
    )
with col_ext2:
    rem_drift = st.number_input(
        "Remaining avg drift deg per 100 m after last survey",
        value=float(np.round(sug_drift, 2)) if sug_drift is not None else 1.0,
        step=0.1,
    )

# Extend actual to planned length
actual_stations = actual_stations_base.copy()
if actual_stations:
    last_md = sorted(actual_stations, key=lambda d: d["MD"])[-1]["MD"]
    if plan_len > last_md + 1e-6:
        actual_stations = extend_actual_stations(actual_stations, plan_len, step_m, rem_lift, rem_drift)

ax, ay, az, amd = min_curvature_path_dip_h(actual_stations) if actual_stations else (np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))

# Target plane inputs
st.subheader("Target plane and pierce points")
col_plane1, col_plane2, col_plane3 = st.columns(3)
with col_plane1:
    plane_strike = st.number_input("Plane strike deg", value=114.0, step=1.0)
with col_plane2:
    plane_dip = st.number_input("Plane dip-from-horizontal deg (negative down)", value=-58.0, step=1.0)
with col_plane3:
    target_depth = st.number_input("Target depth m below collar", value=max(0.0, plan_len - 50.0), step=5.0, min_value=0.0)

# Build plane axes and anchor point
s_hat, d_hat, n_hat = strike_dip_to_axes(plane_strike, plane_dip)
P0 = np.array([0.0, 0.0, -float(target_depth)], dtype=float)  # plane passes through this point

# Compute pierce points
plan_pts = np.column_stack([px, py, pz])
act_pts = np.column_stack([ax, ay, az])
pierce_plan = find_plane_intersection(plan_pts, P0, n_hat)
pierce_act = find_plane_intersection(act_pts, P0, n_hat)

# Distance on the plane and dashed connector endpoints in-plane coordinates
dist_on_plane = None
connector3d = None
if pierce_plan is not None and pierce_act is not None:
    u1 = (pierce_plan - P0) @ s_hat
    v1 = (pierce_plan - P0) @ d_hat
    u2 = (pierce_act - P0) @ s_hat
    v2 = (pierce_act - P0) @ d_hat
    dist_on_plane = float(np.hypot(u2 - u1, v2 - v1))
    connector3d = np.vstack([pierce_plan, pierce_act])

# -----------------
# 3D view
# -----------------
st.markdown("### 3D view")
fig3d = go.Figure()

# Planned and actual
fig3d.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode="lines", name="Planned", line=dict(width=6)))
fig3d.add_trace(go.Scatter3d(x=ax, y=ay, z=az, mode="lines", name="Actual", line=dict(width=6)))

# Collar at origin
fig3d.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", name="Collar", marker=dict(size=5)))

# Plane surface patch
span_u = max(50.0, 0.3*plan_len)
span_v = max(50.0, 0.3*plan_len)
uu = np.linspace(-span_u, span_u, 10)
vv = np.linspace(-span_v, span_v, 10)
UU, VV = np.meshgrid(uu, vv)
plane_grid = P0.reshape(1, 1, 3) + UU[..., None]*s_hat.reshape(1, 1, 3) + VV[..., None]*d_hat.reshape(1, 1, 3)
fig3d.add_trace(go.Surface(
    x=plane_grid[..., 0],
    y=plane_grid[..., 1],
    z=plane_grid[..., 2],
    opacity=0.4,
    showscale=False,
    name="Target plane"
))

# Pierce points and connector
if pierce_plan is not None:
    fig3d.add_trace(go.Scatter3d(x=[pierce_plan[0]], y=[pierce_plan[1]], z=[pierce_plan[2]],
                                 mode="markers", name="Pierce planned", marker=dict(size=6, symbol="x")))
if pierce_act is not None:
    fig3d.add_trace(go.Scatter3d(x=[pierce_act[0]], y=[pierce_act[1]], z=[pierce_act[2]],
                                 mode="markers", name="Pierce actual", marker=dict(size=6)))

if connector3d is not None:
    fig3d.add_trace(go.Scatter3d(
        x=connector3d[:, 0], y=connector3d[:, 1], z=connector3d[:, 2],
        mode="lines", name="Pierce separation", line=dict(width=3, dash="dash")
    ))

fig3d.update_layout(scene=dict(
    xaxis_title="X East m",
    yaxis_title="Y North m",
    zaxis_title="Z m (up)",
), margin=dict(l=0, r=0, b=0, t=0))
st.plotly_chart(fig3d, use_container_width=True)

# Separation readout
if dist_on_plane is not None:
    st.info(f"Pierce point separation on plane: {dist_on_plane:.2f} m")

# -----------------
# Sections
# -----------------
st.markdown("### Sections")

# Plan view XY
fig_plan = go.Figure()
fig_plan.add_trace(go.Scatter(x=px, y=py, mode="lines", name="Planned"))
fig_plan.add_trace(go.Scatter(x=ax, y=ay, mode="lines", name="Actual"))
# plane strike line through P0
strike_line = np.vstack([P0 + s_hat*(-span_u), P0 + s_hat*(span_u)])
fig_plan.add_trace(go.Scatter(x=strike_line[:, 0], y=strike_line[:, 1], mode="lines", name="Plane strike", line=dict(dash="dash")))
# pierce points and connector projected in plan
if pierce_plan is not None:
    fig_plan.add_trace(go.Scatter(x=[pierce_plan[0]], y=[pierce_plan[1]], mode="markers", name="Pierce planned"))
if pierce_act is not None:
    fig_plan.add_trace(go.Scatter(x=[pierce_act[0]], y=[pierce_act[1]], mode="markers", name="Pierce actual"))
if connector3d is not None:
    fig_plan.add_trace(go.Scatter(x=connector3d[:, 0], y=connector3d[:, 1], mode="lines", name="Pierce separation", line=dict(dash="dash")))
fig_plan.update_layout(xaxis_title="X East m", yaxis_title="Y North m", yaxis=dict(scaleanchor="x", scaleratio=1))
st.plotly_chart(fig_plan, use_container_width=True)

# Cross-section along hole azimuth
# Use first actual az if available else planned
sec_az = float(actual_stations[0]["Azimuth"]) if actual_stations else plan_az0
sec_u = np.array([np.sin(np.deg2rad(sec_az)), np.cos(np.deg2rad(sec_az)), 0.0])
sec_u = sec_u/np.linalg.norm(sec_u)
plan_s = project_to_axis(plan_pts, np.array([0.0, 0.0, 0.0]), sec_u)
act_s = project_to_axis(act_pts, np.array([0.0, 0.0, 0.0]), sec_u)

# plane trace in cross-section
p_vals = np.linspace(min(plan_s.min(initial=0.0), act_s.min(initial=0.0), -span_u),
                     max(plan_s.max(initial=0.0), act_s.max(initial=0.0), span_u), 80)
plane_z_cs = []
for p in p_vals:
    XY = sec_u[:2]*p  # since origin at 0,0
    z = plane_z_at_xy(XY[0], XY[1], P0, n_hat)
    plane_z_cs.append(np.nan if z is None else z)
plane_z_cs = np.array(plane_z_cs, dtype=float)

fig_cs = go.Figure()
fig_cs.add_trace(go.Scatter(x=plan_s, y=pz, mode="lines", name="Planned"))
fig_cs.add_trace(go.Scatter(x=act_s, y=az, mode="lines", name="Actual"))
fig_cs.add_trace(go.Scatter(x=p_vals, y=plane_z_cs, mode="lines", name="Plane trace", line=dict(dash="dash")))
# pierce points on cross-section
if pierce_plan is not None:
    s_pl = project_to_axis(pierce_plan.reshape(1, 3), np.array([0.0, 0.0, 0.0]), sec_u)[0]
    fig_cs.add_trace(go.Scatter(x=[s_pl], y=[pierce_plan[2]], mode="markers", name="Pierce planned"))
if pierce_act is not None:
    s_ac = project_to_axis(pierce_act.reshape(1, 3), np.array([0.0, 0.0, 0.0]), sec_u)[0]
    fig_cs.add_trace(go.Scatter(x=[s_ac], y=[pierce_act[2]], mode="markers", name="Pierce actual"))
fig_cs.update_layout(xaxis_title=f"Along section m (az {sec_az:.1f} deg)", yaxis_title="Z m (up)")
st.plotly_chart(fig_cs, use_container_width=True)

# Long-section along plane axes
plan_u = (plan_pts - P0) @ s_hat
plan_v = (plan_pts - P0) @ d_hat
act_u = (act_pts - P0) @ s_hat
act_v = (act_pts - P0) @ d_hat

fig_ls = go.Figure()
fig_ls.add_trace(go.Scatter(x=plan_u, y=plan_v, mode="lines", name="Planned"))
fig_ls.add_trace(go.Scatter(x=act_u, y=act_v, mode="lines", name="Actual"))
fig_ls.add_trace(go.Scatter(x=[-span_u, span_u], y=[0, 0], mode="lines", name="Plane baseline v=0", line=dict(dash="dash")))
if pierce_plan is not None:
    pu1 = (pierce_plan - P0) @ s_hat
    pv1 = (pierce_plan - P0) @ d_hat
    fig_ls.add_trace(go.Scatter(x=[pu1], y=[pv1], mode="markers", name="Pierce planned"))
if pierce_act is not None:
    pu2 = (pierce_act - P0) @ s_hat
    pv2 = (pierce_act - P0) @ d_hat
    fig_ls.add_trace(go.Scatter(x=[pu2], y=[pv2], mode="markers", name="Pierce actual"))
if pierce_plan is not None and pierce_act is not None:
    fig_ls.add_trace(go.Scatter(x=[pu1, pu2], y=[pv1, pv2], mode="lines", name="Pierce separation", line=dict(dash="dash")))
fig_ls.update_layout(xaxis_title="Along strike u m", yaxis_title="Down dip v m")
st.plotly_chart(fig_ls, use_container_width=True)

# Footer tip
st.caption("Angles are dip-from-horizontal. Negative dip points down. Positive lift increases downward dip magnitude. Collar is at the origin.")
