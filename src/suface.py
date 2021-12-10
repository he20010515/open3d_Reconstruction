import open3d as o3d


class surfaceCreater():
    def __init__(self):
        pass

    def set_input_cloud(self, cloud: o3d.geometry.PointCloud):
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
        )
        self.cloud = cloud

        return None

    def compute(self, method: str, alpha=0.01, radii=(0.04, 0.08, 0.16, 0.32), depth=8):
        cloud = self.cloud
        if method not in ["alphaShapes",
                          "ballPivoting",
                          "Poisson"]:
            raise IndexError('算法必须是"["alphaShapes","ballPivoting","Poisson"]中的一个')
        if method == "alphaShapes":
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                cloud, alpha
            )
        elif method == "ballPivoting":
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd=cloud, radii=o3d.utility.DoubleVector(radii)
            )
        else:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd=cloud, depth=depth
            )
            mesh = mesh[0]

        return mesh

    pass


if __name__ == '__main__':
    cloud = o3d.io.read_point_cloud("D:/Workspace/ShareFiles/Eros Gaskell 200k poly.pcd")

    assert isinstance(cloud, o3d.geometry.PointCloud)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30)
    )

    radd = o3d.utility.DoubleVector([10])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        cloud, 0.999)
    o3d.visualization.draw_geometries(
        [cloud, mesh]
    )
