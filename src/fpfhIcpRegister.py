import copy
import functools
import random

import open3d as o3d
import palettable
import numpy as np
from tqdm import tqdm


class ICPregister(object):
    def __init__(self):
        self.origionData = []  # list of o3d.PointCloud Object
        self.trans = []
        self.setup()
        self.colors = [[l[0] / 255, l[1] / 255, l[2] / 255] for l in palettable.mycarta.get_map("CubeYF_20").colors]
        pass

    def setup(self,
              downSamping=True,
              voxelSize=0.05):
        pass

    def loadDataSet(self, filePathList: [str, ]):
        self.voxelSize = 1.2
        for path in filePathList:
            d = o3d.io.read_point_cloud(path)
            assert isinstance(d, o3d.geometry.PointCloud)
            if d.points.__len__() >= 1000:

                print(f"{path} Load successful;PointCloudSize:{d.points.__len__()}")
                cl, ind = d.remove_statistical_outlier(
                    nb_neighbors=200, std_ratio=4 * self.voxelSize
                )
                d = d.select_by_index(ind)
                print(f"{path} remove_statistical_outlier;PointCloudSize:{d.points.__len__()}")

                self.origionData.append(d)
            else:
                print(f"{path} Bad data;       PointCloudSize{d.points.__len__()}")
        for i in range(len(self.origionData)):
            self.origionData[i].paint_uniform_color(self.colors[(i * 1) % len(self.colors)])
        print(
            f"All Data Set Load SuccessFull,good:bad:all -- {self.origionData.__len__()}:{filePathList.__len__() - self.origionData.__len__()}:{filePathList.__len__()}")

    def compute(self):
        for i in tqdm(range(len(self.origionData))):
            if i == 0:
                self.trans.append(np.identity(4))
                continue
            reg = self._registerPairCloud(self.origionData[i - 1], self.origionData[i])
            self.trans.append(reg.transformation)
        pass

    def _registerPairCloud(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud):
        sourceDown, sourceFpfh = self._prepareData(source)
        targetDown, targetFpfh = self._prepareData(target)

        distanceThreshold = self.voxelSize * 3
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            sourceDown, targetDown, sourceFpfh, targetFpfh, False,
            distanceThreshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distanceThreshold),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(distanceThreshold),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        ICP = True
        if ICP:
            distance_threshold = self.voxelSize * 0.9
            resultICP = o3d.pipelines.registration.registration_icp(
                source, target, max_correspondence_distance=distance_threshold, init=result.transformation,
            )
            assert isinstance(resultICP, o3d.pipelines.registration.RegistrationResult)
            return resultICP
        else:
            return result

    def _prepareData(self, pointCloud: o3d.geometry.PointCloud):
        pcdDown = pointCloud.voxel_down_sample(self.voxelSize)
        rediusFeature = self.voxelSize * 5.0
        rediusNormal = self.voxelSize * 2.0
        pcdDown.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=rediusNormal, max_nn=30)
        )

        pcdFpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcdDown,
            o3d.geometry.KDTreeSearchParamHybrid(radius=rediusFeature, max_nn=100)
        )
        return pcdDown, pcdFpfh

    def showResult(self):
        # o3d.visualization.draw_geometries(self.origionData)

        tempData = copy.deepcopy(self.origionData)
        for i in tqdm(range(self.origionData.__len__())):
            if i == 0:
                continue
            for j in range(i):
                tempData[i].transform(np.linalg.inv(self.trans[j + 1]))
        o3d.visualization.draw_geometries(tempData)

        pass

    def getResult(self) -> o3d.geometry.PointCloud:
        tempData = copy.deepcopy(self.origionData)
        for i in tqdm(range(self.origionData.__len__())):
            if i == 0:
                continue
            for j in range(i):
                tempData[i].transform(np.linalg.inv(self.trans[j + 1]))
        tempCloud = functools.reduce(lambda a, b: a + b, tempData)
        assert isinstance(tempCloud, o3d.geometry.PointCloud)
        return tempCloud


if __name__ == '__main__':
    register = ICPregister()
    # paths = [
    #     f"data/BaseBlender/s00{i}can00001.pcd" for i in range(2)
    # ]
    # paths = [
    #     f"./data/Itokawa Hayabusa 200k poly/data{4 * i}.pcd" for i in range(2, 10)
    # ]

    paths = [
        f"./data/Eros Gaskell 200k poly2/data{i}.pcd" for i in range(1, 5)
    ]
    register.loadDataSet(paths)
    register.compute()
    cloud = register.getResult()
    cloud.estimate_normals()
    # print('Begin Compute mesh')
    # mesh, vector = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #     cloud
    # )
    # o3d.visualization.draw_geometries([cloud.voxel_down_sample(voxel_size=register.voxelSize)])
    register.showResult()
    # o3d.visualization.draw_geometries([mesh, cloud])
    # o3d.visualization.draw_geometries([mesh])
