import numpy as np
from scipy.interpolate import interpn


def point_spatial_transformer(points, df_arr):
    d, w, h, c = df_arr.shape

    x = np.linspace(0, d-1, d)
    y = np.linspace(0, w-1, w)
    z = np.linspace(0, h-1, h)
    
    dx = interpn((x, y, z), df_arr[:,:,:,0], points)
    dy = interpn((x, y, z), df_arr[:,:,:,1], points)
    dz = interpn((x, y, z), df_arr[:,:,:,2], points)

    dp = np.stack([dx, dy, dz], axis=-1)
    return dp


def landmark_error(mov_kp, fix_kp, df_arr):
    """
    mov_kp: (N,3)
    fix_kp: (N,3)
    df_arr: (d,w,h,3)
    """
    warp_kp = fix_kp + point_spatial_transformer(fix_kp, df_arr)
    err = np.sqrt(np.sum(np.square(warp_kp-mov_kp), axis=1))
    return err.mean()