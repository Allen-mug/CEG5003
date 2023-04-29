from .dataset import SceneFlowDataset, KITTIDataset, DrivingStereoDataset, MvsecDataset, MyDataset

__datasets__ = {
    "sceneflow": SceneFlowDataset,
    "kitti": KITTIDataset,
    "drivingstereo": DrivingStereoDataset,
    "mvsec": MvsecDataset,
    "tlab": MyDataset,
}
