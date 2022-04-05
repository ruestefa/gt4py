import time

import matplotlib.pyplot as plt
import numpy as np

import gt4py as gt
import gt4py.gtscript as gtscript
from gt4py.gtscript import computation
from gt4py.gtscript import PARALLEL
from gt4py.gtscript import interval

# use case
use_case = "zhao"  # "zhao", "hopf_cole"

# diffusion coefficient
mu = 0.1

# grid
factor = 1
nx = 10 * 2**factor + 1
ny = 10 * 2**factor + 1

# time
cfl = 1.
timestep = cfl / (nx-1)**2
niter = 4**factor * 100

# output
print_period = 50

@gtscript.function
def absolute_value(phi):
    abs_phi = phi[0, 0, 0] * (phi[0, 0, 0] >= 0.) - phi[0, 0, 0] * (phi[0, 0, 0] < 0.)
    return abs_phi

@gtscript.function
def advection_x(dx, u, abs_u, phi):
    adv_phi_x = u[0, 0, 0] / (60. * dx) * (
        + 45. * (phi[1, 0, 0] - phi[-1, 0, 0])
        -  9. * (phi[2, 0, 0] - phi[-2, 0, 0])
        +       (phi[3, 0, 0] - phi[-3, 0, 0])
    ) - abs_u[0, 0, 0] / (60. * dx) * (
        +       (phi[3, 0, 0] + phi[-3, 0, 0])
        -  6. * (phi[2, 0, 0] + phi[-2, 0, 0])
        + 15. * (phi[1, 0, 0] + phi[-1, 0, 0])
        - 20. * phi[0, 0, 0]
    )
    return adv_phi_x

@gtscript.function
def advection_y(dy, v, abs_v, phi):
    adv_phi_y = v[0, 0, 0] / (60. * dy) * (
        + 45. * (phi[0, 1, 0] - phi[0, -1, 0])
        -  9. * (phi[0, 2, 0] - phi[0, -2, 0])
        +       (phi[0, 3, 0] - phi[0, -3, 0])
    ) - abs_v[0, 0, 0] / (60. * dy) * (
        +       (phi[0, 3, 0] + phi[0, -3, 0])
        -  6. * (phi[0, 2, 0] + phi[0, -2, 0])
        + 15. * (phi[0, 1, 0] + phi[0, -1, 0])
        - 20. * phi[0, 0, 0]
    )
    return adv_phi_y

@gtscript.function
def advection(dx, dy, u, v):
    abs_u = absolute_value(phi=u)
    abs_v = absolute_value(phi=v)
    
    adv_u_x = advection_x(dx=dx, u=u, abs_u=abs_u, phi=u)
    adv_u_y = advection_y(dy=dy, v=v, abs_v=abs_v, phi=u)
    adv_u = adv_u_x[0, 0, 0] + adv_u_y[0, 0, 0]
    
    adv_v_x = advection_x(dx=dx, u=u, abs_u=abs_u, phi=v)
    adv_v_y = advection_y(dy=dy, v=v, abs_v=abs_v, phi=v)
    adv_v = adv_v_x[0, 0, 0] + adv_v_y[0, 0, 0]
    
    return adv_u, adv_v

@gtscript.function
def diffusion_x(dx, phi):
    diff_phi = (
        -       phi[-2, 0, 0]
        + 16. * phi[-1, 0, 0]
        - 30. * phi[0, 0, 0]
        + 16. * phi[1, 0, 0]
        -       phi[2, 0, 0]
    ) / (12. * dx**2)
    return diff_phi

@gtscript.function
def diffusion_y(dy, phi):
    diff_phi = (
        -       phi[0, -2, 0]
        + 16. * phi[0, -1, 0]
        - 30. * phi[0, 0, 0]
        + 16. * phi[0, 1, 0]
        -       phi[0, 2, 0]
    ) / (12. * dy**2)
    return diff_phi

@gtscript.function
def diffusion(dx, dy, u, v):
    diff_u_x = diffusion_x(dx=dx, phi=u)
    diff_u_y = diffusion_y(dy=dy, phi=u)
    diff_u = diff_u_x[0, 0, 0] + diff_u_y[0, 0, 0]
    
    diff_v_x = diffusion_x(dx=dx, phi=v)
    diff_v_y = diffusion_y(dy=dy, phi=v)
    diff_v = diff_v_x[0, 0, 0] + diff_v_y[0, 0, 0]
    
    return diff_u, diff_v

# gridtools4py settings
backend = "gtc:numpy"  # "debug", "numpy", "gtx86", "gtmc", "gtcuda" (not working)
backend_opts = {}  # {'verbose': True} if backend.startswith('gt') else {}
dtype = np.float64
origin = (3, 3, 0)
rebuild = False

externals={
    "absolute_value": absolute_value,
    "advection_x": advection_x,
    "advection_y": advection_y,
    "advection": advection,
    "diffusion_x": diffusion_x,
    "diffusion_y": diffusion_y,
    "diffusion": diffusion
}

start_time = time.time()

@gtscript.stencil(backend=backend, externals=externals, rebuild=rebuild, **backend_opts)
def rk_stage(
    in_u_now: gtscript.Field[dtype],
    in_v_now: gtscript.Field[dtype],
    in_u_tmp: gtscript.Field[dtype],
    in_v_tmp: gtscript.Field[dtype],
    out_u: gtscript.Field[dtype],
    out_v: gtscript.Field[dtype],
    *,
    dt: float,
    dx: float,
    dy: float,
    mu: float
):
    with computation(PARALLEL), interval(...):
        adv_u, adv_v = advection(dx=dx, dy=dy, u=in_u_tmp, v=in_v_tmp)
        diff_u, diff_v = diffusion(dx=dx, dy=dy, u=in_u_tmp, v=in_v_tmp)
        out_u = in_u_now[0, 0, 0] + dt * (- adv_u[0, 0, 0] + mu * diff_u[0, 0, 0])
        out_v = in_v_now[0, 0, 0] + dt * (- adv_v[0, 0, 0] + mu * diff_v[0, 0, 0])

@gtscript.stencil(backend=backend)
def copy(in_phi: gtscript.Field[dtype], out_phi: gtscript.Field[dtype]):
    with computation(PARALLEL), interval(...):
        out_phi = in_phi[0, 0, 0]
    
print("\n- Compilation time: ", time.time() - start_time )

def solution_factory(t, x, y, slice_x=None, slice_y=None):
    nx, ny = x.shape[0], y.shape[0]
    
    slice_x = slice_x or slice(0, nx)
    slice_y = slice_y or slice(0, ny)
    
    mi = slice_x.stop - slice_x.start
    mj = slice_y.stop - slice_y.start
    
    x2d = np.tile(x[slice_x, np.newaxis, np.newaxis], (1, mj, 1))
    y2d = np.tile(y[np.newaxis, slice_y, np.newaxis], (mi, 1, 1))
        
    if use_case == "zhao":
        u = - 4. * mu * np.pi * np.exp(- 5. * np.pi**2 * mu * t) * \
            np.cos(2. * np.pi * x2d) * np.sin(np.pi * y2d) / \
            (2. + np.exp(- 5. * np.pi**2 * mu * t) * np.sin(2. * np.pi * x2d) * np.sin(np.pi * y2d))
        v = - 2. * mu * np.pi * np.exp(- 5.0 * np.pi**2 * mu * t) * \
            np.sin(2. * np.pi * x2d) * np.cos(np.pi * y2d) / \
            (2. + np.exp(- 5. * np.pi**2 * mu * t) * np.sin(2. * np.pi * x2d) * np.sin(np.pi * y2d))
    elif use_case == "hopf_cole":
        u = .75 - 1. / (4. * (1. + np.exp(- t - 4.*x2d + 4.*y2d) / (32.*mu)))
        v = .75 + 1. / (4. * (1. + np.exp(- t - 4.*x2d + 4.*y2d) / (32.*mu)))
    else:
        raise NotImplementedError()
    
    return u, v


def set_initial_solution(x, y, u, v):
    u[...], v[...] = solution_factory(0., x, y)
    
    
def enforce_boundary_conditions(t, x, y, u, v):
    nx, ny = x.shape[0], y.shape[0]
    
    slice_x, slice_y = slice(0, 3), slice(0, ny)
    u[slice_x, slice_y], v[slice_x, slice_y] = solution_factory(t, x, y, slice_x, slice_y)
    
    slice_x, slice_y = slice(nx-3, nx), slice(0, ny)
    u[slice_x, slice_y], v[slice_x, slice_y] = solution_factory(t, x, y, slice_x, slice_y)
    
    slice_x, slice_y = slice(3, nx-3), slice(0, 3)
    u[slice_x, slice_y], v[slice_x, slice_y] = solution_factory(t, x, y, slice_x, slice_y)
    
    slice_x, slice_y = slice(3, nx-3), slice(ny-3, ny)
    u[slice_x, slice_y], v[slice_x, slice_y] = solution_factory(t, x, y, slice_x, slice_y)


def zeros(storage_shape, backend, dtype, origin=None, mask=None):
    origin = origin or (0, 0, 0)
    origin = tuple(origin[i] if storage_shape[i] > 2 * origin[i] else 0 for i in range(3))
    domain = tuple(storage_shape[i] - 2 * origin[i] for i in range(3))

    gt_storage = gt.storage.zeros(backend=backend, dtype=dtype, shape=storage_shape, mask=mask, default_origin=origin)
    return gt_storage

x = np.linspace(0., 1., nx)
dx = 1. / (nx - 1)
y = np.linspace(0., 1., ny)
dy = 1. / (ny - 1)

u_now = zeros((nx, ny, 1), backend, dtype, origin)
v_now = zeros((nx, ny, 1), backend, dtype, origin)
u_new = zeros((nx, ny, 1), backend, dtype, origin)
v_new = zeros((nx, ny, 1), backend, dtype, origin)

set_initial_solution(x, y, u_new, v_new)

rk_fraction = (1./3., .5, 1.)

t = 0.

start_time = time.time()

for i in range(niter):
    copy(in_phi=u_new, out_phi=u_now, origin=(0, 0, 0), domain=(nx, ny, 1))
    copy(in_phi=v_new, out_phi=v_now, origin=(0, 0, 0), domain=(nx, ny, 1))
        
    for k in range(3):
        dt = rk_fraction[k] * timestep
         
        rk_stage(
            in_u_now=u_now, in_v_now=v_now, in_u_tmp=u_new, in_v_tmp=v_new,
            out_u=u_new, out_v=v_new, dt=dt, dx=dx, dy=dy, mu=mu,
            origin=(3, 3, 0), domain=(nx-6, ny-6, 1)
        )
        
        enforce_boundary_conditions(t + dt, x, y, u_new, v_new)
        
    t += timestep
    if print_period > 0 and ((i+1) % print_period == 0 or i+1 == niter):
        u_ex, v_ex = solution_factory(t, x, y)
        err_u = np.linalg.norm(u_new[3:-3, 3:-3] - u_ex[3:-3, 3:-3]) * np.sqrt(dx * dy)
        err_v = np.linalg.norm(v_new[3:-3, 3:-3] - v_ex[3:-3, 3:-3]) * np.sqrt(dx * dy)
        print(
            "Iteration {:6d}: ||u - uex|| = {:8.4E} m/s, ||v - vex|| = {:8.4E} m/s".format(
                i + 1, err_u, err_v
            )
        )

print("\n- Running time: ", time.time() - start_time )

