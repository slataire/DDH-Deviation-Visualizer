import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Drillhole vs Plan", layout="wide")

# -----------------
# Helper functions
# -----------------
def wrap_az(az):
    return az % 360

def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))

def min_curvature_path(stations, collar, conv="dip_from_horizontal"):
    if len(stations) < 1:
        return np.array([collar[0]]), np.array([collar[1]]), np.array([collar[2]])

    stations = sorted(stations, key=lambda d: float(d["MD"]))
    MDs = np.array([float(s["MD"]) for s in stations])
    Azs = np.array([wrap_az(float(s["Azimuth"])) for s in stations])
    Angs = np.array([float(s["Angle"]) for s in stations])

    A = np.deg2rad(Azs)
    if conv == "dip_from_horizontal":
        I = np.deg2rad(90.0 - Angs)  # incl from vertical
    else:
        I = np.deg2rad(Angs)

    X, Y, Z = [collar[0]], [collar[1]], [collar[2]]
    for i in range(1, len(MDs)):
        dMD = MDs[i] - MDs[i-1]
        if dMD <= 0:
            continue
        inc1, inc2 = I[i-1], I[i]
        az1, az2 = A[i-1], A[i]
        cos_dog = np.sin(inc1)*np.sin(inc2)*np.cos(az2-az1) + np.cos(inc1)*np.cos(inc2)
        cos_dog = np.clip(cos_dog, -1.0, 1.0)
        dog = np.arccos(cos_dog)
        RF = 1.0 if dog < 1e-6 else (2.0/dog)*np.tan(dog/2.0)
        dN = 0.5*dMD*(np.sin(inc1)*np.cos(az1)+np.sin(inc2)*np.cos(az2))*RF
        dE = 0.5*dMD*(np.sin(inc1)*np.sin(az1)+np.sin(inc2)*np.sin(az2))*RF
        dZ = 0.5*dMD*(np.cos(inc1)+np.cos(inc2))*RF
        X.append(X[-1] + dE)
        Y.append(Y[-1] + dN)
        Z.append(Z[-1] + dZ)
    return np.array(X), np.array(Y), np.array(Z)

def make_planned_stations(length, step, az0, dip0, lift, drift):
    stations = []
    md = 0.0
    az, dip = az0, dip0
    while md <= length:
        stations.append({"MD": md, "Azimuth": az, "Angle": dip})
        md += step
        az = wrap_az(az + drift*(step/100.0))
        dip = clamp(dip + lift*(step/100.0), 0, 90)
    return stations

# -----------------
# UI
# -----------------
st.title("Diamond Drillhole Tracking with 3D Visualization")

collar_x = st.number_input("Collar X", value=0.0)
collar_y = st.number_input("Collar Y", value=0.0)
collar_z = st.number_input("Collar Z", value=0.0)
collar = np.array([collar_x, collar_y, collar_z])

conv = st.radio("Angle convention", ["dip_from_horizontal", "incl_from_vertical"])

plan_len = st.number_input("Planned length (m)", value=500.0)
plan_az0 = st.number_input("Planned start azimuth", value=90.0)
if conv == "dip_from_horizontal":
    plan_ang0 = st.number_input("Planned start dip (deg from horizontal)", value=60.0)
else:
    plan_ang0 = st.number_input("Planned start inclination (deg from vertical)", value=30.0)

plan_lift = st.number_input("Planned lift deg/100m", value=2.0)
plan_drift = st.number_input("Planned drift deg/100m", value=1.0)
step_m = st.number_input("Step size (m)", value=10.0)

# Planned hole
planned_stations = make_planned_stations(plan_len, step_m, plan_az0, plan_ang0, plan_lift, plan_drift)
px, py, pz = min_curvature_path(planned_stations, collar, conv)

# Actual hole from CSV or manual
st.header("Actual Surveys")
method = st.radio("Input method", ["Manual", "CSV"])
if method == "CSV":
    file = st.file_uploader("Upload survey CSV (MD,Azimuth,Angle)", type="csv")
    if file:
        df = pd.read_csv(file)
        actual_stations = df.to_dict("records")
    else:
        actual_stations = []
else:
    df = st.data_editor(pd.DataFrame([{"MD":0,"Azimuth":plan_az0,"Angle":plan_ang0}]), num_rows="dynamic")
    actual_stations = df.to_dict("records")

if actual_stations:
    ax, ay, az = min_curvature_path(actual_stations, collar, conv)
else:
    ax, ay, az = [collar[0]], [collar[1]], [collar[2]]

# -----------------
# 3D Plot
# -----------------
fig = go.Figure()

fig.add_trace(go.Scatter3d(x=px, y=py, z=pz,
                           mode="lines",
                           name="Planned",
                           line=dict(width=6, color="blue")))
fig.add_trace(go.Scatter3d(x=ax, y=ay, z=az,
                           mode="lines",
                           name="Actual",
                           line=dict(width=6, color="red")))
fig.add_trace(go.Scatter3d(x=[collar[0]], y=[collar[1]], z=[collar[2]],
                           mode="markers",
                           marker=dict(size=6, color="green"),
                           name="Collar"))

fig.update_layout(scene=dict(
    xaxis_title="X East (m)",
    yaxis_title="Y North (m)",
    zaxis_title="Z (m, up)"
), margin=dict(l=0,r=0,b=0,t=0))

st.plotly_chart(fig, use_container_width=True)
