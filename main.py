from src.ReconstructionSystem import ReconstructionSystem
import open3d as o3d

if __name__ == '__main__':
    paths = [
        f"./data/Eros Gaskell 200k poly3/{i}.pcd" for i in range(10, 15)
    ]
    paths = [
        f"./data/Eros Gaskell 200k poly3/{i}.pcd" for i in range(15, 20)
    ]
    # paths = [
    #     f"./data/Eros Gaskell 200k poly3/{i}.pcd" for i in range(10, 20)
    # ]
    # paths = [
    #     f"./data/Eros Gaskell 200k poly3/{i}.pcd" for i in range(20, 30)
    # ]
    # paths = [
    #     f"./data/doubleRoute/{2 * i}.pcd" for i in range(50, 60)
    # ]
    paths = [
        f"./data/Churyumov-Gerasimenko Malmer/{i}.pcd" for i in range(0, 100)
    ]
    system = ReconstructionSystem()
    system.setInputClouds(paths)
    cloud = system.compute4()
    o3d.visualization.draw_geometries([cloud])
    o3d.io.write_point_cloud(
        'cloud.pcd', cloud
    )
