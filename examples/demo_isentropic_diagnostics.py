import time

import matplotlib.pyplot as plt
import numpy as np
import gt4py as gt
import gt4py.gtscript as gtscript
from gt4py.gtscript import computation
from gt4py.gtscript import PARALLEL
from gt4py.gtscript import interval

# grid size
nx = 32
ny = 32
nz = 64

# brunt-vaisala frequency
bv = .01

# physical constants
rd = 287.05
g = 9.81
p_ref = 1.0e5
cp = 1004.

# gridtools4py settings
backend = "numpy"  # "debug", "numpy", "gtx86", "gtmc", "gtcuda" (not working)
backend_opts = {'verbose': True} if backend.startswith('gt') else {}
dtype = np.float64
origin = (3, 3, 0)
rebuild = True

externals = {"rd": rd, "g": g, "p_ref": p_ref, "cp": cp}

start_time = time.time()

@gtscript.stencil(backend=backend, externals=externals, **backend_opts)
def diagnostic_step(
    in_theta: gtscript.Field[dtype],
    in_hs: gtscript.Field[dtype],
    in_s: gtscript.Field[dtype],
    inout_p: gtscript.Field[dtype],
    out_exn: gtscript.Field[dtype],
    inout_mtg: gtscript.Field[dtype],
    inout_h: gtscript.Field[dtype],
    *,
    dtheta: float,
    pt: float
):
    # retrieve the pressure
    with computation(FORWARD), interval(0,1):
        inout_p = pt
    with computation(FORWARD), interval(1, None):
        inout_p = inout_p[0, 0, -1] + g * dtheta * in_s[0, 0, -1]

    # compute the Exner function
    with computation(PARALLEL), interval(0, None):
        out_exn = cp * (inout_p[0, 0, 0] / p_ref) ** (rd / cp)

    # compute the Montgomery potential
    with computation(BACKWARD), interval(-2,-1):
        mtg_s = in_theta[0, 0, 1] * out_exn[0, 0, 1] + g * in_hs[0, 0, 1]
        inout_mtg = mtg_s + 0.5 * dtheta * out_exn[0, 0, 1]
    with computation(BACKWARD), interval(0,-2):
        inout_mtg = inout_mtg[0, 0, 1] + dtheta * out_exn[0, 0, 1]

    # compute the geometric height of the isentropes
    with computation(BACKWARD), interval(-1, None):
        inout_h = in_hs[0, 0, 0]
    with computation(BACKWARD), interval(0, -1):
        inout_h = inout_h[0, 0, 1] - rd * (
            in_theta[0, 0, 0] * out_exn[0, 0, 0]
            + in_theta[0, 0, 1] * out_exn[0, 0, 1]
        ) * (inout_p[0, 0, 0] - inout_p[0, 0, 1]) / (
            cp * g * (inout_p[0, 0, 0] + inout_p[0, 0, 1])
        )
        
print("\n- Compilation time: ", time.time() - start_time )

def zeros(storage_shape, backend, dtype, origin=None, mask=None):
    origin = origin or (0, 0, 0)
    origin = tuple(origin[i] if storage_shape[i] > 2 * origin[i] else 0 for i in range(3))
    domain = tuple(storage_shape[i] - 2 * origin[i] for i in range(3))
    
    gt_storage = gt.storage.zeros(shape=storage_shape, dtype=dtype, default_origin=origin, mask=mask, backend=backend)
    return gt_storage

# define the vertical grid
theta1d = np.linspace(340., 280., nz+1)
theta = zeros((nx, ny, nz+1), backend, dtype, origin)
theta[...] = theta1d[np.newaxis, np.newaxis, :]

# the vertical grid spacing
dtheta = 60. / nz

# let us assume the topography consists of a bell-shaped isolated mountain
hs = zeros((nx, ny, nz+1), backend, dtype, origin)
x1d = np.linspace(-150e3, 150e3, nx)
y1d = np.linspace(-150e3, 150e3, ny)
x, y = np.meshgrid(x1d, y1d, indexing="ij")
hs[:, :, -1] = 1000. * np.exp(- (x / 50e3)**2 - (y / 50e3)**2)

# initialize the Exner function (needed to compute the isentropic density)
exn = np.zeros((nx, ny, nz+1), dtype=dtype)
exn[:, :, -1] = cp
for k in range(nz - 1, -1, -1):
    exn[:, :, k] = exn[:, :, k + 1] - dtheta * (g ** 2) / (
        (bv ** 2) * (theta[:, :, k] ** 2)
    )

# retrieve the air pressure (needed to compute the isentropic density)
p = p_ref * ((exn / cp) ** (cp / rd))

# diagnose the isentropic density
s = zeros((nx, ny, nz+1), backend, dtype, origin)
s[:, :, :-1] = -(p[:, :, :-1] - p[:, :, 1:]) / (g * dtheta)

# allocate the output storages
out_p = zeros((nx, ny, nz+1), backend, dtype, origin)
out_exn = zeros((nx, ny, nz+1), backend, dtype, origin)
out_mtg = zeros((nx, ny, nz+1), backend, dtype, origin)
out_h = zeros((nx, ny, nz+1), backend, dtype, origin)

# compute all the diagnostic variables
diagnostic_step(
    in_theta=theta, in_hs=hs, in_s=s, inout_p=out_p, out_exn=out_exn, 
    inout_mtg=out_mtg, inout_h=out_h, dtheta=dtheta, pt=p[0, 0, 0],
    origin=(0, 0, 0), domain=(nx, ny, nz+1)
)

out_p =np.asarray(out_p)
out_exn =np.asarray(out_exn)
out_mtg =np.asarray(out_mtg)

j = int(ny / 2)
xx = 1e-3 * np.repeat(x[:, j, np.newaxis], nz+1, axis=1)
yy = 1e-3 * np.asarray(out_h[:, j, :])

fig = plt.figure(figsize=(10, 10))

ax00 = fig.add_subplot(2, 2, 1)
surf = ax00.contourf(xx, yy, 1e-2*out_p[:, j, :], cmap="Blues")
plt.colorbar(surf, orientation="vertical")
ax00.plot(xx[:, -1], yy[:, -1], color="black", linewidth=1.5)
ax00.set_xlim((-100, 100))
ax00.set_xlabel("$x$ [km]")
ax00.set_ylim((0, 14))
ax00.set_ylabel("$z$ [km]")
ax00.set_title("Air pressure [hPa]")

ax01 = fig.add_subplot(2, 2, 2)
surf = ax01.contourf(xx, yy, out_exn[:, j, :], cmap="Greens")
ax01.plot(xx[:, -1], yy[:, -1], color="black", linewidth=1.5)
plt.colorbar(surf, orientation="vertical")
ax01.set_xlim((-100, 100))
ax01.set_xlabel("$x$ [km]")
ax01.set_ylim((0, 14))
ax01.set_ylabel("$z$ [km]")
ax01.set_title("Exner function [J kg$^{-1}$ K$^{-1}$]")

ax10 = fig.add_subplot(2, 2, 3)
surf = ax10.contourf(xx, yy, 1e-3*out_mtg[:, j, :], cmap="Reds")
ax10.plot(xx[:, -1], yy[:, -1], color="black", linewidth=1.5)
plt.colorbar(surf, orientation="vertical")
ax10.set_xlim((-100, 100))
ax10.set_xlabel("$x$ [km]")
ax10.set_ylim((0, 14))
ax10.set_ylabel("$z$ [km]")
ax10.set_title("Montgomery potential [10$^3$ m$^2$ s$^{-2}$]")

ax11 = fig.add_subplot(2, 2, 4)
for k in range(0, nz, 3):
    ax11.plot(xx[:, k], yy[:, k], color="gray", linewidth=1.1)
ax11.plot(xx[:, -1], yy[:, -1], color="black", linewidth=1.5)
ax11.set_xlim((-100, 100))
ax11.set_xlabel("$x$ [km]")
ax11.set_ylim((0, 14))
ax11.set_ylabel("$z$ [km]")
ax11.set_title("Vertical levels")

# fig.suptitle("Diagnostic fields at $y = {:3.3f}$ km".format(1e-3*y1d[j]))
fig.tight_layout()
