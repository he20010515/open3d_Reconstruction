import open3d as o3d
from src.ReconstructionSystem import ReconstructionSystem
from src.suface import surfaceCreater

if __name__ == '__main__':
    pcdPath = r"D:\Workspace\ShareFiles\Churyumov-Gerasimenko Malmer.pcd"
    cloud = o3d.io.read_point_cloud(pcdPath)
    c = surfaceCreater()
    c.set_input_cloud(cloud)
    # meshalpha = c.compute(method="alphaShapes", alpha=0.3)
    meshball = c.compute(method="ballPivoting")
    # meshPossion = c.compute(method="Poisson")
    o3d.visualization.draw_geometries(
        [meshball]
    )
