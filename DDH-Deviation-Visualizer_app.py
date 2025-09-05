# --- 3D view with expanded field of view and padding ---
st.markdown("### 3D view")

# Collect all points to determine bounds
pts = [np.column_stack([px, py, pz]), np.column_stack([ax, ay, az])]
extras = [P0.reshape(1, 3)]
if pierce_plan is not None:
    extras.append(pierce_plan.reshape(1, 3))
if pierce_act is not None:
    extras.append(pierce_act.reshape(1, 3))

all_pts = np.vstack(pts + extras)
xmin, ymin, zmin = np.min(all_pts, axis=0)
xmax, ymax, zmax = np.max(all_pts, axis=0)

# Add generous padding
range_x = xmax - xmin
range_y = ymax - ymin
range_z = zmax - zmin
max_span = max(range_x, range_y, range_z, 1.0)
pad = max(0.2 * max_span, 25.0)  # at least 25 m padding

xr = [xmin - pad, xmax + pad]
yr = [ymin - pad, ymax + pad]
zr = [zmin - pad, zmax + pad]

# Plane patch size scaled to scene
span = 0.6 * max_span + 50.0  # bigger than before so pierce points are not near edges
uu, vv = np.meshgrid(np.linspace(-span, span, 20), np.linspace(-span, span, 20))
plane_grid = P0.reshape(1, 1, 3) + uu[..., None]*s_hat.reshape(1, 1, 3) + vv[..., None]*d_hat.reshape(1, 1, 3)

fig3d = go.Figure()

# Planned and actual
fig3d.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode="lines", name="Planned", line=dict(width=6)))
fig3d.add_trace(go.Scatter3d(x=ax, y=ay, z=az, mode="lines", name="Actual", line=dict(width=6)))

# Collar at origin
fig3d.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", name="Collar", marker=dict(size=5)))

# Plane surface
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
    fig3d.add_trace(go.Scatter3d(
        x=[pierce_plan[0]], y=[pierce_plan[1]], z=[pierce_plan[2]],
        mode="markers", name="Pierce planned", marker=dict(size=6, symbol="x")
    ))
if pierce_act is not None:
    fig3d.add_trace(go.Scatter3d(
        x=[pierce_act[0]], y=[pierce_act[1]], z=[pierce_act[2]],
        mode="markers", name="Pierce actual", marker=dict(size=6)
    ))
if pierce_plan is not None and pierce_act is not None:
    fig3d.add_trace(go.Scatter3d(
        x=[pierce_plan[0], pierce_act[0]],
        y=[pierce_plan[1], pierce_act[1]],
        z=[pierce_plan[2], pierce_act[2]],
        mode="lines", name="Pierce separation", line=dict(width=3, dash="dash")
    ))

# Apply expanded ranges and a comfortable camera
fig3d.update_layout(
    scene=dict(
        xaxis_title="X East m", xaxis=dict(range=xr),
        yaxis_title="Y North m", yaxis=dict(range=yr),
        zaxis_title="Z m (up)", zaxis=dict(range=zr),
        aspectmode="cube"
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    scene_camera=dict(eye=dict(x=1.6, y=1.6, z=1.1))  # pull back a bit
)

st.plotly_chart(fig3d, use_container_width=True)

