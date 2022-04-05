import matplotlib.pyplot as plt
import numpy as np

import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage
from gt4py.gtscript import computation
from gt4py.gtscript import PARALLEL
from gt4py.gtscript import interval

backend="gtc:numpy" # "debug", "numpy", "gtx86", "gtcuda"
dtype = np.float64


@gtscript.stencil(backend=backend) # this decorator triggers compilation of the stencil
def horizontal_diffusion(
    in_field: gtscript.Field[dtype],
    out_field: gtscript.Field[dtype],
    coeff: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):
        lap_field = 4.0 * in_field[0, 0, 0] - (
            in_field[1, 0, 0] + in_field[-1, 0, 0] + in_field[0, 1, 0] + in_field[0, -1, 0]
        )
        res = lap_field[1, 0, 0] - lap_field[0, 0, 0]
        flx_field = 0 if (res * (in_field[1, 0, 0] - in_field[0, 0, 0])) > 0 else res
        res = lap_field[0, 1, 0] - lap_field[0, 0, 0]
        fly_field = 0 if (res * (in_field[0, 1, 0] - in_field[0, 0, 0])) > 0 else res
        out_field = in_field[0, 0, 0] - coeff[0, 0, 0] * (
            flx_field[0, 0, 0] - flx_field[-1, 0, 0] + fly_field[0, 0, 0] - fly_field[0, -1, 0]
        )


# Initialize example data
N = 30
shape = [N] * 3
origin = (2, 2, 0)

indices = np.arange(N)
ii = np.zeros((N, N, N)) + np.reshape(indices, (N, 1, 1))
jj = np.zeros((N, N, N)) + np.reshape(indices, (1, N, 1))
kk = np.zeros((N, N, N)) + np.reshape(indices, (1, 1, N))

xx = ii / N
yy = jj / N
zz = kk / N

in_data = 5. + 8. * (2. + np.cos(np.pi * (xx + 1.5 * yy)) + np.sin(2 * np.pi * (xx + 1.5 * yy))) / 4.
out_data = np.zeros(shape)
coeff_data = 0.025 * np.ones(shape)

# Plot initialization
# projection = np.array(np.sum(in_data, axis=2))
# plt.imshow(projection)

in_storage = gt_storage.from_array(
    in_data, backend, default_origin=origin, dtype=dtype 
)
out_storage = gt_storage.from_array(
    out_data, backend, default_origin=origin, dtype=dtype, 
)
coeff_storage = gt_storage.from_array(
    coeff_data, backend, default_origin=origin, dtype=dtype, 
)

horizontal_diffusion(in_storage, out_storage, coeff_storage)

if backend=="gtcuda":
    out_storage.synchronize() # does a copy if the cpu or gpu buffer is modified.
    
# projection = np.asarray(np.sum(out_storage, axis=2))
# plt.imshow(projection)
