# streamlit_app.py


# Distance between pierce points if available
pierce_distance = None
if pt_actual_pierce is not None:
pierce_distance = float(np.linalg.norm(pt_actual_pierce - pt_planned_pierce))


# 3D Plot
fig = go.Figure()


fig.add_trace(go.Scatter3d(
x=P_planned[:, 0], y=P_planned[:, 1], z=P_planned[:, 2],
mode="lines", name="Planned trace",
line=dict(width=6)
))


fig.add_trace(go.Scatter3d(
x=P_actual[:, 0], y=P_actual[:, 1], z=P_actual[:, 2],
mode="lines", name="Actual trace",
line=dict(width=4)
))


fig.add_trace(go.Scatter3d(
x=[pt_planned_pierce[0]], y=[pt_planned_pierce[1]], z=[pt_planned_pierce[2]],
mode="markers", name="Planned pierce",
marker=dict(size=6, symbol="diamond")
))


if pt_actual_pierce is not None:
fig.add_trace(go.Scatter3d(
x=[pt_actual_pierce[0]], y=[pt_actual_pierce[1]], z=[pt_actual_pierce[2]],
mode="markers", name="Actual pierce",
marker=dict(size=6, symbol="x")
))


# Plot target plane as surface
fig.add_trace(go.Surface(
x=Xpl, y=Ypl, z=Zpl,
showscale=False,
name="Target plane",
opacity=0.5
))


# Optional helper - vertical line at collar
fig.add_trace(go.Scatter3d(x=[collar[0], collar[0]], y=[collar[1], collar[1]], z=[collar[2], collar[2] - plane_size],
mode="lines", name="Collar vertical", line=dict(width=2, dash="dot")))


fig.update_scenes(aspectmode="data")
fig.update_layout(
scene=dict(
xaxis_title="X East m",
yaxis_title="Y North m",
zaxis_title="Z Elevation m",
),
legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)


st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


# Info panel
st.subheader("Results")
colA, colB, colC = st.columns(3)
colA.metric("Planned pierce X,Y,Z m", f"{pt_planned_pierce[0]:.2f}, {pt_planned_pierce[1]:.2f}, {pt_planned_pierce[2]:.2f}")
if pt_actual_pierce is not None:
colB.metric("Actual pierce X,Y,Z m", f"{pt_actual_pierce[0]:.2f}, {pt_actual_pierce[1]:.2f}, {pt_actual_pierce[2]:.2f}")
else:
colB.metric("Actual pierce X,Y,Z m", "No intersection within EOH")


if pierce_distance is not None:
colC.metric("Linear distance between pierce points m", f"{pierce_distance:.2f}")
else:
colC.metric("Linear distance between pierce points m", "-")


st.caption(
"Conventions - Azimuth clockwise from North. Dip 0 is horizontal, -90 is vertical down. Coordinates are X East, Y North, Z Up."
)


st.markdown(
"""
**Notes**
- The planned path integrates lift and drift continuously using minimum curvature for each small step.
- The actual path uses minimum curvature between survey tests, then applies your assumed lift and drift per 100 m after the last test to EOH.
- The target plane is positioned to pass through the planned point at your target depth and oriented by the strike and dip you set.
- If the actual trace does not intersect the plane before EOH, the app reports no intersection.
- Reduce the computation step for a smoother curve and more precise intersections.
"""
)
