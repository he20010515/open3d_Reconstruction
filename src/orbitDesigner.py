import math
import os.path
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from math import sin
from math import cos

root_path = r'D:\Workspace\ShareFiles'
obj_file_list = [
    'Churyumov-Gerasimenko Malmer.obj',
    'Eros Gaskell 200k poly.obj',
    'Itokawa Hayabusa 200k poly.obj',
    'SHAPE_SFM_3M_v20180804.obj'
]
pcd_file_list = [
    s.split('.')[0] + '.pcd' for s in obj_file_list
]

FRAME_PRE_SECOND = 10


def updateLocation(scanid: int, R, deltaPhi, deltaTheta):
    nowtime = scanid / FRAME_PRE_SECOND

    x = R * sin(nowtime * deltaTheta) * cos(nowtime * deltaPhi)
    y = R * sin(nowtime * deltaPhi) * sin(nowtime * deltaTheta)
    z = R * cos(nowtime * deltaTheta)
    return x, y, z


def argument_function(t: np.ndarray, R, n=4):
    x_ = np.zeros_like(t)
    y_ = np.zeros_like(t)
    z_ = np.zeros_like(t)
    deltaTheta = 1

    deltaPhi = 2 * n * deltaTheta
    for index, now_time in enumerate(t):
        x, y, z = updateLocation(now_time * FRAME_PRE_SECOND, deltaTheta=deltaTheta, deltaPhi=deltaPhi, R=R)
        x_[index] = x
        y_[index] = y
        z_[index] = z
    return x_, y_, z_


if __name__ == '__main__':
    pcd_file = o3d.io.read_point_cloud(
        os.path.join(root_path, pcd_file_list[1])
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    assert isinstance(pcd_file, o3d.geometry.PointCloud)
    r = 2.3 * np.mean(np.linalg.norm(np.asarray(pcd_file.points), axis=-1))
    print('r = ', r)

    x, y, z = np.asarray(pcd_file.uniform_down_sample(10).points).transpose()
    ax.scatter(
        x, y, z, c='b', marker='.', s=2, linewidth=0, alpha=1, cmap='spectral'
    )
    endtime = np.pi
    t = np.linspace(0, np.pi, 1000)
    x, y, z = argument_function(t, R=r)
    ax.plot(
        x, y, z
    )
    plt.show()

    pass
