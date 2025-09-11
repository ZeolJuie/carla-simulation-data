import numpy as np

pred_occ_path = "/home/zmh/codes/FlashOCC/data/nuscenes/gts/scene-0002/0bcb08a96d264c8ca9f2119c6b0dbeb2/labels.npz"
# pred_occ_path = "/home/zmh/codes/carla-simulation-data/nuscenes_carla/gts/scene-0014/sample_016950/labels.npz"

pred_occ = np.load(pred_occ_path, allow_pickle=True)

print(np.sum(pred_occ["mask_camera"]))

breakpoint()