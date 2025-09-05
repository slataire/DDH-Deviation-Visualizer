import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="DDH Deviation Visualizer", layout="wide")

# -----------------
# Helper functions
# -----------------
def wrap_az(az):
    return az % 360

def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))

def min_curvature_path(stations):
    if not stations:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])

    stations = sorted(stations, key=lambda d: float(d["MD"]))
    MDs = np.array([s["MD"] for s in stations], float)
    AZs = np.deg2rad([wrap_az(s["Azimuth"]) for s in stations])
    DIP = np.array([s["Angle"] for s in stations], float)

    INC = np.deg2rad(90 - np.abs(DIP))  # inclination-from-vertical
    SGN = np.where(DIP <= 0.0, -1.0, 1.0)  # negative dip points down

    X, Y, Z = [0.0], [0.0], [0.0]

    for i in range(1, len(MDs)):
        dMD = MDs[i] - MDs[i-1]
        if dMD <= 0:
            continue
        inc1, inc2 = INC[i-1], INC[i]
        az1, az2 = AZs[i-1], AZs[i]
        s1, s2 = SGN[i-1], SGN[i]

        cos_dog = np.sin(inc1)*np.sin(inc2)*np.cos(az2-az1) + np.cos(inc1)*np.cos(inc2)
        cos_dog = np.clip(cos_dog, -1, 1)
        dog = np.arccos(cos_dog)
        RF = 1 if dog < 1e-12 else (2/dog)*np.tan(dog/2)

        dN = 0.5*dMD*(np.sin(inc1)*np.cos(az1)+np.sin(inc2)*np.cos(az2))*RF
        dE = 0.5*dMD*(np.sin(inc1)*np.sin(az1)+np.sin(inc2)*np.sin(az2))*RF
        dZ = 0.5*dMD*(s1*np.cos(inc1)+s2*np.cos(inc2))*RF

        X.append(X[-1] + dE)
        Y.append(Y[-1] + dN)
        Z.append(Z[-1] + dZ)

    return np.array(X), np.array(Y), np.array(Z)

def make_planned_stations(length_m, step_m, az0, dip0, lift_per100, drift_per100):
    stations = []
    md, az, dip = 0.0, wrap_az(az0), dip0
    while md <= length_m:
        stations.append({"MD": md, "Azimuth": az, "Angle": dip})
        md += step_m
        az = wrap_az(az + drift_per100*(step_m/100))
        dip = clamp(dip - lift_per100*(step_m/100), -90, 90)  # subtract -> more negative = steeper down
    return stations

def extend_actual(stations, to_depth, step_m, lift_per100, drift_per100):
    if not stations:
        return stations
    stations = sorted(stations, key=lambda d: d["MD"])
    last = stations[-1]
    md, az, dip = last["MD"], last["Azimuth"], last["Angle"]
    while md < to_depth - 1e-6:
        d = min(step_m, to_depth - md)
        md += d
        az = wrap_az(az + drift_per100*(d/100))
        dip = clamp(dip - lift_per100*(d/100), -90, 90)
        stations.append({"MD": md, "Azimuth": az, "Angle": dip})
    return stations

def derive_lift_drift_last3(stations):
    if len(stations) < 3:
        return None, None
    sta = sorted(stations, key=lambda d: d["MD"])[-3:]
    MD = np.array([s["MD"] for s in sta])
    AZ = np.array([s["Azimuth"] for s in sta])
    DIP = np.array([s["Angle"] for s in sta])
    AZu = np.unwrap(np.deg2rad(AZ))
    drift = np.rad2deg(np.polyfit(MD, AZu, 1)[0]) * 100
    lift = -np.polyfit(MD, DIP, 1)[0] * 100
    return lift, drift

def strike_dip_to_axes(strike, dip_signed):
    strike = np.deg2rad(wrap_az(strike))
    dip_abs = np.deg2rad(abs(dip_signed))
    s_hat = np.array([np.sin(strike), np.cos(strike), 0])
    dipdir = strike + np.pi/2
    d_hat = np.array([np.sin(dipdir)*np.cos(dip_abs), np.cos(dipdir)*np.cos(dip_abs), -np.sin(dip_abs)])
    n_hat = np.cross(s_hat, d_hat)
    return s_hat/np.linalg.norm(s_hat), d_hat/np.linalg.norm(d_hat), n_hat/np.linalg.norm(n_hat)

def segment_plane_intersection(p0, p1, P0, n_hat):
    u = p1 - p0
    denom = np.dot(n_hat, u)
    if abs(denom) < 1e-12:
        return None
    t = np.dot(n_hat, P0 - p0)/denom
    if 0 <= t <= 1:
        return p0 + t*u
    return None

def find_plane_intersection(points, P0, n_hat):
    for i in range(1, len(points)):
        p = segment_plane_intersection(points[i-1], points[i], P0, n_hat)
        if p is not None:
            return p
    return None

# -----------------
# UI
# -----------------
st.title("Drillhole vs Planned with Target Plane")

plan_len = st.number_input("Planned length (m)", value=500.0, step=10.0)
plan_az0 = st.number_input("Planned azimuth (deg)", value=90.0)
plan_dip0 = st.number_input("Planned dip-from-horizontal (deg, negative down)", value=-60.0)
plan_lift = st.number_input("Planned lift (deg/100m)", value=2.0)
plan_drift = st.number_input("Planned drift (deg/100m)", value=1.0)
step_m = st.number_input("Step (m)", value=10.0)

planned_stations = make_planned_stations(plan_len, step_m, plan_az0, plan_dip0, plan_lift, plan_drift)
px, py, pz = min_curvature_path(planned_stations)

st.subheader("Actual surveys")
method = st.radio("Input surveys", ["Manual", "CSV"], horizontal=True)
if method == "CSV":
    file = st.file_uploader("Upload CSV with MD, Azimuth, Angle", type="csv")
    if file:
        df = pd.read_csv(file)
    else:
        df = pd.DataFrame(columns=["MD","Azimuth","Angle"])
else:
    df = st.data_editor(pd.DataFrame([{"MD":0,"Azimuth":plan_az0,"Angle":plan_dip0}]), num_rows="dynamic")

actual_stations = df.dropna().to_dict("records")
sug_lift, sug_drift = derive_lift_drift_last3(actual_stations)

st.write("Suggested from last 3:", f"Lift {sug_lift:.2f}" if sug_lift else "-", f"Drift {sug_drift:.2f}" if sug_drift else "-")

rem_lift = st.number_input("Remaining avg lift (deg/100m)", value=sug_lift if sug_lift else 2.0)
rem_drift = st.number_input("Remaining avg drift (deg/100m)", value=sug_drift if sug_drift else 1.0)

if actual_stations:
    actual_stations = extend_actual(actual_stations, plan_len, step_m, rem_lift, rem_drift)
    ax, ay, az = min_curvature_path(actual_stations)
else:
    ax, ay, az = [0.0], [0.0], [0.0]

# Plane
st.subheader("Target plane")
plane_strike = st.number_input("Plane strike (deg)", value=114.0)
plane_dip = st.number_input("Plane dip-from-horizontal (deg, negative down)", value=-58.0)
target_depth = st.number_input("Target depth (m)", value=plan_len-50)

s_hat, d_hat, n_hat = strike_dip_to_axes(plane_strike, plane_dip)
P0 = np.array([0,0,-target_depth])

plan_pts = np.column_stack([px, py, pz])
act_pts = np.column_stack([ax, ay, az])
pierce_plan = find_plane_intersection(plan_pts, P0, n_hat)
pierce_act = find_plane_intersection(act_pts, P0, n_hat)

# 3D plot
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=px,y=py,z=pz,mode="lines",name="Planned"))
fig.add_trace(go.Scatter3d(x=ax,y=ay,z=az,mode="lines",name="Actual"))
fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[0],mode="markers",name="Collar"))

# Plane patch
span=0.3*plan_len
uu,vv=np.meshgrid(np.linspace(-span,span,10),np.linspace(-span,span,10))
grid=P0+uu[...,None]*s_hat+vv[...,None]*d_hat
fig.add_trace(go.Surface(x=grid[...,0],y=grid[...,1],z=grid[...,2],opacity=0.4,showscale=False,name="Target plane"))

# Pierce points and connector
if pierce_plan is not None:
    fig.add_trace(go.Scatter3d(x=[pierce_plan[0]],y=[pierce_plan[1]],z=[pierce_plan[2]],mode="markers",name="Pierce planned"))
if pierce_act is not None:
    fig.add_trace(go.Scatter3d(x=[pierce_act[0]],y=[pierce_act[1]],z=[pierce_act[2]],mode="markers",name="Pierce actual"))
if pierce_plan is not None and pierce_act is not None:
    fig.add_trace(go.Scatter3d(x=[pierce_plan[0],pierce_act[0]],y=[pierce_plan[1],pierce_act[1]],z=[pierce_plan[2],pierce_act[2]],mode="lines",name="Pierce separation",line=dict(dash="dash")))
    st.info(f"Pierce separation on plane: {np.linalg.norm((pierce_act-pierce_plan) - np.dot((pierce_act-pierce_plan), n_hat)*n_hat):.2f} m")

fig.update_layout(scene=dict(xaxis_title="X",yaxis_title="Y",zaxis_title="Z (up)"),margin=dict(l=0,r=0,b=0,t=0))
st.plotly_chart(fig, use_container_width=True)
