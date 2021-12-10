try:
    import bpy

    from bpy import data as D
    from bpy import context as C
    from mathutils import *

    import blensor

except:
    pass
import math
from numpy import arange
from math import *
import time
import numpy as np


class Orbit():

    def __init__(self, FRAME_PRE_SECOND,
                 deltaTheta,
                 deltaPhi,
                 start_time: float,
                 end_time: float,
                 R: float):
        self.FRAME_PRE_SECOND = FRAME_PRE_SECOND
        self.deltaTheta = deltaTheta  # 维度变换速度
        self.deltaPhi = deltaPhi  # 经度变换速度
        self.start_time = start_time
        self.end_time = end_time
        self.R = R
        pass

    def orbit_scan(self):
        start = int(self.start_time * self.FRAME_PRE_SECOND)
        end = int(self.end_time * self.FRAME_PRE_SECOND)
        while start <= end:
            x, y, z = self._update_location(start, self.R, self.deltaPhi, self.deltaPhi)
            bpy.data.objects['Camera'].location[0] = x
            bpy.data.objects['Camera'].location[1] = y
            bpy.data.objects['Camera'].location[2] = z
            start += 1
            if self._scan(start):
                pass
            else:
                pass
        pass

    def move(self, time):
        scanid = time * self.FRAME_PRE_SECOND
        x, y, z = self._update_location(scanid, self.R, self.deltaPhi, self.deltaPhi)
        bpy.data.objects['Camera'].location[0] = x
        bpy.data.objects['Camera'].location[1] = y
        bpy.data.objects['Camera'].location[2] = z

    def _scan(self, scanid):
        beginObjects = set([obj.name for obj in bpy.data.objects])  # 当前所有扫描对象
        bpy.ops.blensor.delete_scans()  # 清除所有scans
        bpy.ops.blensor.scan()  # 扫描
        endObjects = set([obj.name for obj in bpy.data.objects])
        try:
            newScanName = (endObjects - beginObjects).pop()
        except:
            return False
        cloud = bpy.data.objects[newScanName]
        pointsNums = len(cloud.data.vertices)
        head = (
                "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA ascii\n" % (
            pointsNums, pointsNums))
        body = []
        for sp in cloud.data.vertices:
            string = '%#5.3f\t%#5.3f\t%#5.3f \n' % (sp.co[0], sp.co[1], sp.co[2])
            body.append(string)

        with open(f"/home/heyuwei/data/{scanid}.pcd", 'w') as f:
            f.write(head)
            f.writelines(body)
        return True

    def _update_location(self, scanid: int, R, deltaPhi, deltaTheta):
        nowtime = scanid / self.FRAME_PRE_SECOND
        x = R * sin(nowtime * deltaTheta) * cos(nowtime * deltaPhi)
        y = R * sin(nowtime * deltaPhi) * sin(nowtime * deltaTheta)
        z = R * cos(nowtime * deltaTheta)
        return x, y, z

    def show_orbit(self, pcd_file_path):
        try:
            import matplotlib.pyplot as plt
            import open3d as o3d
        except:
            return
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        pcd_file = o3d.io.read_point_cloud(pcd_file_path)
        assert isinstance(pcd_file, o3d.geometry.PointCloud)
        x, y, z = np.asarray(pcd_file.uniform_down_sample(10).points).transpose()
        ax.scatter(
            x, y, z, c='b', marker='.', s=2, linewidth=0, alpha=1, cmap='spectral'
        )

        t = np.linspace(self.start_time, self.end_time, 1000)
        x, y, z = np.vectorize(lambda nowTime: self._argument_function(nowTime))(t)

        ax.plot(
            x, y, z
        )
        plt.show()
        pass

    def _argument_function(self, nowtime):
        x = self.R * sin(nowtime * self.deltaTheta) * cos(nowtime * self.deltaPhi)
        y = self.R * sin(nowtime * self.deltaPhi) * sin(nowtime * self.deltaTheta)
        z = self.R * cos(nowtime * self.deltaTheta)
        return x, y, z


if __name__ == '__main__':
    lam = 0.2
    orbit = Orbit(200, np.pi, 12 * np.pi, 0 + lam, 1 - lam, 20)
    orbit.show_orbit(r"D:\Workspace\ShareFiles\Eros Gaskell 200k poly.pcd")
    pass
