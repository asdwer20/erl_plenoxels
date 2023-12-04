import dataclasses
import os
import json
import yaml
import numpy as np
from pathlib import Path

from dataclasses import dataclass
from typing import List
from typing import Dict


TRAIN_IMG_DIR = "train"
TEST_IMG_DIR = "train"

TRAIN_TRAJ_FNAME = "traj.txt"
REPLICA_INFO_FNAME = "replicaCAD_info.json"

TRAIN_JSON_FNAME = "transforms_train.json"
TEST_JSON_FNAME = "transforms_test.json"
VAL_JSON_FNAME = "transforms_val.json"

@dataclass(frozen=True)
class Frame:
    file_path: str
    transform_matrix: List[List[float]]
    rotation: float = 0.0  # This is apparently ignored in NerF, can't seem to find an explanation for this parameter


@dataclass(frozen=True)
class DatasetOutput:
    camera_angle_x: float
    frames: List[Frame]


if __name__ == "__main__":
    # Train Trajectory
    train_traj = np.loadtxt(TRAIN_TRAJ_FNAME)
    with open(REPLICA_INFO_FNAME, 'r') as f:
        replica_config = json.load(f)
    camera_angle_x = np.arctan2((0.5*replica_config['camera']['w'])/replica_config['camera']['fx'], 1)*2

    # Loading Training Data
    filelist = os.listdir(TRAIN_IMG_DIR)
    filelist.sort()

    train_frame_list: List[Frame] = []
    test_frame_list: List[Frame] = []
    for i,filename in enumerate(filelist):
        # Optional Limiting
        if i == 200:
            break
        filename = Path(filename).stem
        pose = train_traj[i].reshape((4,4))
        frame = Frame(
            transform_matrix=pose.tolist(),
            file_path=os.path.join(TRAIN_IMG_DIR, filename)
        )
        if i % 2 == 0:
            train_frame_list.append(frame)
        else:
            test_frame_list.append(frame)
    train_json = DatasetOutput(
        camera_angle_x=camera_angle_x,
        frames=train_frame_list
    )
    test_json = DatasetOutput(
        camera_angle_x=camera_angle_x,
        frames=test_frame_list
    )
    with open(TRAIN_JSON_FNAME, 'w') as f:
        json.dump(dataclasses.asdict(train_json), f)
    with open(TEST_JSON_FNAME, 'w') as f:
        json.dump(dataclasses.asdict(test_json), f)
    with open(VAL_JSON_FNAME, 'w') as f:
        json.dump(dataclasses.asdict(test_json), f)
