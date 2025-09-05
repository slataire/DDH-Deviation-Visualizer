import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ----------------- GEOMETRY HELPERS -----------------

def deg2rad(a):
    return np.deg2rad(a)

def step_vector(ds, az_deg, dip_deg):
    """
    ds in meters.
    azimuth az_deg is clockwise from North.
    dip_deg is inclination from horizontal, positive down (0 horizontal, 90 vertical down).
    Returns dE, dN, dDepth (depth positive down).
    """
    az = deg2rad(az_deg)
    dip = deg2rad(dip_deg)
    horiz = ds * np.cos(dip)
    dE = horiz * np.sin(az)
    dN = horiz * np.cos(az)
    dDepth = ds * np.sin(dip)  # positive down
    return dE, dN, dDepth

def project_to_section(E, N, Depth, E0, N0, az_section_deg):
    """
    Vertical section plane through (E0, N0) oriented along az_section_deg.
    x_section increases along the planned azimuth direction.
    y_section is Depth (positive down).
    Returns x_section, y_section, out_of_plane_offset.
    """
    az = deg2rad(az_section_deg)
    # in-plane unit vector along section line
    ux = np.sin(az)
    uy = np.cos(az)
    # horizontal unit normal to the section plane (used for offset reporting)
    nx = np.cos(az)
    ny = -np.sin(az)

    dE = E - E0
    dN = N - N0

    x_sec = dE * ux + dN * uy
    offset = dE * nx + dN * ny
    y_sec = Depth  # already positive down
    return x_sec, y_sec, offset

# ----------------- TRAJECTORY BUILDERS -----------------

def trajectory_planned(E0, N0, dip0_deg, az0_deg, length_m, lift_per_100=0.0, drift_per_100=0.0, step=5.0):
    """
    Generate planned trajectory with constant lift and drift per 100 m.
    Lift reduces dip (shallower) by lift_per_100 deg per 100 m.
    Drift increases az clockwise by drift_per_100 deg per 100 m.
    """
    num = int(np.ceil(length_m / step))
    E = [E0]
    N = [N0]
    D = [0.0]  # depth positive down
    az = az0_deg
    dip = dip0_deg
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
    """
    Build a current hole trajectory from survey stations and then extrapolate
    with remaining lift and drift after the last survey to the end of hole.
    surveys is a list of dicts with keys depth, azimuth, dip. Depth is meters along hole.
    """
    if len(surveys) == 0:
        raise ValueError("Provide at least one survey row.")

    # sort and ensure first station at depth 0 if needed
    surveys = sorted(surveys, key=lambda s: s["depth"])
    if surveys[0]["depth"] > 0:
        # assume initial orientation equals first station orientation at collar
        surveys = [{"depth": 0.0, "azimuth": surveys[0]["azimuth"], "dip": surveys[0]["dip"]}] + surveys

    E = [E0]
    N = [N0]
    D = [0.0]
    az = surveys[0]["azimuth"]
    dip = surveys[0]["dip"]
    curr_depth = 0.0

    # integrate between surveys with linear interpolation of az and dip
    for i in range(len(surveys) - 1):
        d0 = surveys[i]["depth"]
        d1 = surveys[i + 1]["depth"]
        if d1 <= d0:
            continue
        az0 = surveys[i]["azimuth"]
        az1 = surveys[i + 1]["azimuth"]
        dip0 = surveys[i]["dip"]
        dip1 = surveys[i + 1]["dip"]

        # rates per meter within this segment
        d_az_per_m = (az1 - az0) / (d1 - d0)
        d_dip_per_m = (dip1 - dip0) / (d1 - d0)
        az = az0
        dip = dip0

        s = d0
        while s < d1 - 1e-6:
            ds = min(step, d1 - s)
            dE, dN, dDep = step_vector(ds, az, dip)
            E.append(E[-1] + dE)
            N.append(N[-1] + dN)
            D.append(D[-1] + dDep)
            s += ds
            az += d_az_per_m * ds
            dip += d_dip_per_m * ds
        curr_depth = d1
        az = az1
        dip = dip1

    # extrapolate to total_length with remaining lift and drift
    if total_length > curr_depth + 1e-6:
        d_az_per_m = rem_drift_per_100 / 100.0
        d_dip_per_m = -rem_lift_per_100 / 100.0
        s = curr_depth
        while s < total_length - 1e-6:
            ds = min(step, total_length - s)
            dE, dN, dDep = step_vector(ds, az, dip)
            E.append(E[-1] + dE)
            N.append(N[-1] + dN)
            D.append(D[-1] + dDep)
            s += ds
            az += d_az_per_m * ds
            dip += d_dip_per_m * ds

    return np.array(E), np.array(N), np.array(D)

# ----------------- STREAMLIT UI -----------------

st.set_page_config(page_title="Drillhole Cross Section", layout="wide")

st.title("Drillhole Cross Section - vertical plane through planned collar")

with st.sidebar:
    st.subheader("Planned hole inputs")
    collar_E = st.number_input("Collar Easting (m)", value=500000.0, step=1.0, format="%.3f")
    collar_N = st.number_input("Collar Northing (m)", value=5500000.0, step=1.0, format="%.3f")
    planned_az = st.number_input("Planned azimuth deg (clockwise from North)", value=355.0, step=0.1)
    planned_dip = st.number_input("Planned dip deg (positive down from horizontal)", value=68.0, step=0.1)
    eoh = st.number_input("Planned end of hole length m", value=730.0, step=1.0)
    plan_lift = st.number_input("Planned lift deg per 100 m", value=4.0, step=0.1)
    plan_drift = st.number_input("Planned drift deg per 100 m", value=2.0, step=0.1)

    st.subheader("Current hole surveys")
    st.caption("Edit the table. Depth in meters, azimuth clockwise from North, dip positive down from horizontal.")
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
            "depth": st.column_config.NumberColumn("depth", help="meters along hole", step=1.0),
            "azimuth": st.column_config.NumberColumn("azimuth", step=0.1),
            "dip": st.column_config.NumberColumn("dip", step=0.1),
        },
    )

    rem_lift = st.number_input("Remaining lift after last survey deg per 100 m", value=0.0, step=0.1)
    rem_drift = st.number_input("Remaining drift after last survey deg per 100 m", value=0.0, step=0.1)
    step_len = st.number_input("Integration step m", value=5.0, min_value=0.5, step=0.5)

# build trajectories
surveys = []
for _, r in enumerate(df):
    try:
        d = float(r["depth"]); a = float(r["azimuth"]); ip = float(r["dip"])
        surveys.append({"depth": d, "azimuth": a, "dip": ip})
    except Exception:
        pass

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

# project to section plane that passes through the planned collar, oriented along planned azimuth
xP, yP, offP = project_to_section(E_plan, N_plan, D_plan, collar_E, collar_N, planned_az)
xC, yC, offC = project_to_section(E_curr, N_curr, D_curr, collar_E, collar_N, planned_az)

max_depth = max(float(np.nanmax(yP)), float(np.nanmax(yC)))
max_x = max(float(np.nanmax(np.abs(xP))), float(np.nanmax(np.abs(xC))))
pad_x = max(10.0, 0.05 * max_x)
pad_y = max(10.0, 0.05 * max_depth)

# ----------------- PLOT -----------------
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(xP, yP, label="Planned", linewidth=2)
ax.plot(xC, yC, label="Current", linewidth=2)

# collar marker at top
ax.scatter([0.0], [0.0], s=60, zorder=5, label="Collar")

ax.set_xlabel("Section distance along planned azimuth (m)")
ax.set_ylabel("Depth (m, positive down)")

# put collar near the top by forcing 0 at top and increasing down
ax.set_ylim(max_depth + pad_y, -pad_y)  # 0 near top, increasing downwards
ax.set_xlim(-pad_x, max_x + pad_x)

ax.grid(True, linestyle=":", linewidth=0.8)
ax.legend()

st.subheader("Cross section")
st.pyplot(fig)

# ----------------- OFF-PLANE INFO -----------------
mean_off_plan = float(np.nanmean(np.abs(offC))) if offC.size else 0.0
max_off_plan = float(np.nanmax(np.abs(offC))) if offC.size else 0.0
st.caption(
    f"Current hole off-plane offset relative to the section: mean {mean_off_plan:.1f} m, max {max_off_plan:.1f} m. "
    f"Section is a vertical plane through the planned collar, oriented at azimuth {planned_az:.1f} deg."
)

