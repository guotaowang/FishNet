import numpy as np
import torch
'''
3D rotation w.r.t. three axes
'''
def rotx(ang):
    return np.array([[1,       0,          0      ],
                     [0, np.cos(ang), -np.sin(ang)],
                     [0, np.sin(ang), np.cos(ang)]])
def roty(ang):
    return np.array([[np.cos(ang),  0, np.sin(ang)],
                     [      0,      1,      0     ],
                     [-np.sin(ang), 0, np.cos(ang)]])
def rotz(ang):
    return np.array([[np.cos(ang), -np.sin(ang), 0],
                     [np.sin(ang), np.cos(ang),  0],
                     [     0,           0,       1]])
'''
Conversion from one representation to another
'''
def sphere_to_twod(data, radius): # 球转ERP
    # Input: [..., 2(theta, phi)]
    # Output: [..., 2(h, w)]
    C_sph = (0, 0)    # theta, phi center coordinate

    out = np.zeros_like(data)
    out[..., 0] = radius * (data[..., 1] - C_sph[0])
    out[..., 1] = radius * (data[..., 0] - C_sph[1])
    return out


def sphere_to_threed(data): # 球转3D
    # Input: [..., 2(theta, phi)]
    # Output: [..., 3(x, y, z)]
    out = np.zeros_like(data)[..., :-1]
    out = np.concatenate([out, out, out], axis=-1)

    out[..., 0] = np.sin(data[..., 1]) * np.cos(data[..., 0])
    out[..., 1] = np.sin(data[..., 1]) * np.sin(data[..., 0])
    out[..., 2] = np.cos(data[..., 1])
    return out


def threed_to_sphere(data):  # 3D转球
    # Input: [..., 3(x, y, z)]
    # Output: [..., 2(theta, phi)]
    out = np.zeros_like(data)[..., :-1]
    out[..., 0] = np.arctan2(data[..., 1], data[..., 0])

    z = np.sqrt(np.sum(data[..., :-1] * data[..., :-1], axis=-1))
    out[..., 1] = np.arctan2(z, data[..., 2])
    return out


def normalize_threed(data):
    # Normalize every 3D coordinate to norm=1
    out = np.sqrt(np.sum(data * data, axis=-1))
    for i in range(data.shape[-1]):
        data[..., i] /= out
    return data


def compute_patch(input_resolution, patch, ang_y, ang_z, is_discrete=False):
    '''
    Convert normal patch to deformed patch
    Input: 
        patch (tangential plane defined on (1,0,0))
        ang (angle to rotate vertically)
        (horizontal rotation is trivial)
        is_discrete (True if you need integer index to access pixel)
    Output:
        out (deformed patch)
    '''
    height = input_resolution
    RES = (height, height * 2)
    RAD = RES[1] / (2 * np.pi) # 半径（448/2*pi）

    patch = patch @ roty(ang_y)
    patch = patch @ rotz(ang_z) # 4，4，3
    
    out = normalize_threed(patch) # 2D坐标归一化
    out = threed_to_sphere(out)  # 3D转球
    out = sphere_to_twod(out, RAD)  # 球转ERP 4，4，2

    # Handle overflow
    out[..., 0] = np.where(out[..., 0] > RES[0],
                           out[..., 0] - RES[0],
                           out[..., 0])
    out[..., 0] = np.where(out[..., 0] < 0,
                           out[..., 0] + RES[0],
                           out[..., 0])
    out[..., 1] = np.where(out[..., 1] > RES[1],
                           out[..., 1] - RES[1],
                           out[..., 1])
    out[..., 1] = np.where(out[..., 1] < 0,
                           out[..., 1] + RES[1],
                           out[..., 1])

    if is_discrete:
        out = out.astype(int)

    return out


def compute_all_patches(patch_size, resolution, is_discrete=False):
    patch_no = resolution // patch_size  # 384 / 4 = 96

    P = np.arctan(np.pi / (patch_no * 2)) # 每个小切面对应的圆心角大小，弧度。arctan代表将圆心角大小转换为对应的角度大小。
    R = patch_size

    # linspace for y is reverse in order, in order to make patch upright
    x, y = np.meshgrid(np.linspace(-P, P, R, dtype=np.float64),
                       np.linspace(P, -P, R, dtype=np.float64))

    x = np.expand_dims(x, -1) # 4,4,1
    y = np.expand_dims(y, -1) # 4,4,1
    z = np.ones_like(x) # 4,4,1

    patch = np.concatenate([z, x, y], axis=-1)# 4,4,3

    patches = []

    # To fix orientation issue, both for-loops iterate in a reverse order
    lat_range = patch_no // 2    # 28
    lon_range = patch_no * 2 - 1 # 111
    for lat in range(lat_range, -lat_range, -1):
        patches_lat = []
        for lon in range(lon_range, -1, -1):
            patch_lon = compute_patch(patch=patch,
                                      ang_y=P * (2 * lat - 1),
                                      ang_z=P * (2 * lon + 1),
                                      is_discrete=is_discrete,
                                      input_resolution=resolution)
            patches_lat.append(patch_lon)
        patches.append(patches_lat)

    return np.array(patches)

## 计算变形卷积的偏移量
def compute_deform_offset(patch_size, resolution, is_discrete=False):
    R = patch_size

    patches = compute_all_patches(patch_size, resolution, is_discrete=is_discrete) # 96,192,4,4,2
    deform_offset = []

    for i in range(patches.shape[0]):
        col_offset = []

        for j in range(patches.shape[1]):
            # Destination (deformed patch)
            dst = patches[i, j].flatten()
            
            # Source (normal patch, before deformation)
            xx, yy = np.meshgrid(np.arange(R*j, R*(j+1)), np.arange(R*i, R*(i+1)))
            xx = np.expand_dims(xx, axis=-1)
            yy = np.expand_dims(yy, axis=-1)
            src = np.concatenate((yy, xx), axis=-1).flatten()
            
            col_offset.append(np.expand_dims(np.expand_dims(src-dst, axis=-1), axis=-1))
        # First concatenate w.r.t. last dimension (i.e., width)
        col_offset = np.concatenate(col_offset, axis=-1)

        deform_offset.append(col_offset)

    # Finally concatenate w.r.t. second last dimension (i.e., height)
    deform_offset = np.concatenate(deform_offset, axis=-2)

    # (16*16*2, 14, 28)
    return deform_offset

def visualize_patch(patch_size, resolution):
    height = resolution ## 224
    R = patch_size # 4

    dst2 = np.zeros((resolution,resolution*2,2))

    patches = compute_all_patches(patch_size, resolution, is_discrete=True)# 56，112，4，4，2

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            idx = patches[i,j].astype(int).reshape((-1, 2)) # 2,16
            idx[0] %= height
            idx[1] %= height * 2

            dst2[R*i:R*(i+1), R*j:R*(j+1)] = idx.reshape(R, R, 2)

    return dst2

def visualize_patch2(patch_size, resolution):
    height = resolution ## 224
    R = patch_size # 4

    dst2 = np.zeros((resolution,resolution*2,2))

    patches = compute_all_patches(patch_size, resolution, is_discrete=True)# 56，112，4，4，2

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            idx = patches[i,j].astype(int).reshape((-1, 2)) # 2,16
            idx[0] %= height
            idx[1] %= height * 2

            dst2[idx.T[0],idx.T[1]] = [(mm, nn) for mm in range(R*i, R*(i+1)) for nn in range(R*j, R*(j+1))]
    grid = torch.stack(((torch.from_numpy(dst2[:,:,1].astype(np.float32)/((resolution*2-1)/2)-1)).float(), (torch.from_numpy(dst2[:,:,0].astype(np.float32)/((resolution-1)/2)-1)).float()),dim=2)
    return grid

def visualize_patch3(patch_size, resolution):
    height = resolution ## 224
    R = patch_size # 4

    dst2 = np.zeros((resolution,resolution*2,2))

    patches = compute_all_patches(patch_size, resolution, is_discrete=True)# 56，112，4，4，2

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            idx = patches[i,j].astype(int).reshape((-1, 2)) # 2,16
            idx[0] %= height
            idx[1] %= height * 2

            dst2[idx.T[0],idx.T[1]] = [(mm, nn) for mm in range(R*i, R*(i+1)) for nn in range(R*j, R*(j+1))]
    return dst2
    