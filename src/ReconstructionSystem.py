import functools

import open3d as o3d
import palettable
import numpy as np
from tqdm import tqdm
import copy


class ReconstructionSystem():
    p = {
        "voxelSize": 0.018,
        "radiusFeature": 5.,  # radiusFeature*voxelSize
        "radiusNormal": 2.,  # radiusFeature*voxelSize
        "FPFHdistanceThreshold": 1.5,
        "ICPdistanceThreshold": 0.8,
        "useICP": True,  # 是否使用ICP精配准
        "useRANSC": False,  # 是否使用RANSC粗配准,如果是FALSE,使用fast配准 ,
    }

    def __init__(self):
        self.origionData = []
        self.trans = []
        self.colors = [[l[0] / 255, l[1] / 255, l[2] / 255] for l in palettable.mycarta.get_map("CubeYF_20").colors]

    def _preprocessData(self, cloud: o3d.geometry.PointCloud) -> (
            o3d.geometry.PointCloud, o3d.pipelines.registration.Feature):
        pcdDown = cloud.voxel_down_sample(self.p['voxelSize'])
        rediusFeature = self.p['voxelSize'] * self.p['radiusFeature']
        rediusNormal = self.p['voxelSize'] * self.p['radiusNormal']
        pcdDown.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=rediusNormal, max_nn=30)
        )
        pcdFpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcdDown,
            o3d.geometry.KDTreeSearchParamHybrid(radius=rediusFeature, max_nn=100)
        )
        return pcdDown, pcdFpfh

    def _registerPairCloud(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud):
        sourceDown, sourceFpfh = self._preprocessData(source)
        targetDown, targetFpfh = self._preprocessData(target)
        distanceThreshold = self.p['voxelSize'] * self.p['FPFHdistanceThreshold']
        if self.p["useRANSC"]:
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                sourceDown, targetDown, sourceFpfh, targetFpfh, False,
                distanceThreshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                6,
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distanceThreshold),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 1000)
            )
        else:
            result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
                sourceDown, targetDown, sourceFpfh, targetFpfh,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=distanceThreshold,
                    # iteration_number=2000
                )
            )
        if self.p["useICP"]:
            distance_threshold = self.p['voxelSize'] * self.p['ICPdistanceThreshold']
            result = o3d.pipelines.registration.registration_icp(
                source, target, distance_threshold, result.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-5
                )
            )
            assert isinstance(result, o3d.pipelines.registration.RegistrationResult)
        else:
            pass
        return result

    def setInputClouds(self, path_list: [str, ...]):
        tqdmbar = tqdm(path_list)
        for path in tqdmbar:
            d = o3d.io.read_point_cloud(path)
            assert isinstance(d, o3d.geometry.PointCloud)
            if d.points.__len__() >= 1000:

                # tqdmbar.write(f"{path} Load successful;PointCloudSize:{d.points.__len__()}")
                # cl, ind = d.remove_statistical_outlier(
                #     nb_neighbors=200, std_ratio=2 * self.p['voxelSize']
                # )
                # d = d.select_by_index(ind)
                # tqdmbar.write(f"{path} remove_statistical_outlier;PointCloudSize:{d.points.__len__()}")

                xyz = np.asarray(d.points)
                xyz = xyz - np.mean(xyz, axis=0)
                d = o3d.geometry.PointCloud()
                d.points = o3d.utility.Vector3dVector(xyz)

                self.origionData.append(d)
            else:
                tqdmbar.write(f"{path} Bad data;       PointCloudSize{d.points.__len__()}")
        for i in range(len(self.origionData)):
            self.origionData[i].paint_uniform_color(self.colors[(i * 1) % len(self.colors)])
        pass

    def showResult(self):
        # o3d.visualization.draw_geometries(self.origionData)
        tempData = copy.deepcopy(self.origionData)
        tempData = [cloud.voxel_down_sample(self.p['voxelSize']) for cloud in tempData]
        for i in tqdm(range(self.origionData.__len__())):
            if i == 0:
                continue
            for j in range(i):
                tempData[i].transform(self.trans[j + 1])
        targetData = functools.reduce(lambda x, y: x + y, tempData)
        assert isinstance(targetData, o3d.geometry.PointCloud)
        o3d.visualization.draw_geometries([targetData.voxel_down_sample(self.p['voxelSize'])])
        pass

    def getResult(self):
        pass

    def compute(self):
        tqdmbar = tqdm(range(len(self.origionData)))
        for i in tqdmbar:
            if i == 0:
                self.trans.append(np.identity(4))
                continue
            reg = self._registerPairCloud(self.origionData[i], self.origionData[i - 1])
            self.trans.append(reg.transformation)
            tqdm.write(f"Fitness:{reg.fitness},rmse:{reg.inlier_rmse},nums of pairs:{reg.correspondence_set.__len__()}")
        pass

    def compute2(self, K=20):
        tqdmbar = tqdm(range(len(self.origionData) // K))
        tempCloudList = []
        for i in tqdmbar:
            tempCloud = o3d.geometry.PointCloud()
            for k in range(K):
                if k == 0:
                    tempCloud += self.origionData[i * K + k]
                    continue
                reg = self._registerPairCloud(self.origionData[i * K + k], tempCloud)
                tempCloud += self.origionData[i * K + k].transform(reg.transformation)
                tempCloud = tempCloud.voxel_down_sample(self.p['voxelSize'])
                tempCloudList.append(tempCloud)
                o3d.io.write_point_cloud(f"id = {i}.pcd", tempCloud)
        ansCloud = o3d.geometry.PointCloud()
        for index, cloud in enumerate(tempCloudList):
            if index == 0:
                ansCloud += cloud
                continue
            reg = self._registerPairCloud(cloud, ansCloud)
            ansCloud += cloud.transform(reg.transformation)
            ansCloud = ansCloud.voxel_down_sample(voxel_size=self.p['voxelSize'])

        return ansCloud

    def compute3(self):
        targetDataList = copy.deepcopy(self.origionData)
        for i in tqdm(range(len(targetDataList))):
            if i == 0:
                continue
            reg = self._registerPairCloud(targetDataList[i], self.origionData[i - 1])
            for k in range(i, len(targetDataList)):
                targetDataList[k] = targetDataList[k].transform(reg.transformation)
        targetCloud = o3d.geometry.PointCloud()
        for cloud in targetDataList:
            targetCloud += cloud
        return targetCloud.voxel_down_sample(self.p['voxelSize'])

    def compute4(self):
        targetCloud = o3d.geometry.PointCloud()
        for index in range(len(self.origionData)):
            if index == 0:
                targetCloud += self.origionData[index]
                continue
            reg = self._registerPairCloud(source=self.origionData[index],
                                          target=targetCloud)
            targetCloud += self.origionData[index].transform(reg.transformation)
            targetCloud = targetCloud.voxel_down_sample(self.p['voxelSize'])
            o3d.io.write_point_cloud(f'index={index}.pcd', targetCloud)
            print(f"Index = {index}\t,Fitness = {reg.fitness}")
        return targetCloud


"""
第二个数据集Voxel 取0.05比较好

"""
