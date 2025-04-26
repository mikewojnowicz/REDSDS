
import os
import re
import numpy as np
import random 

from typing import List 

import torch
from torch.utils.data import Dataset, DataLoader

class Basketball_Dataset_Train(Dataset):
    """
    Create a dataset which has __getitem__ defined to give a batch in the form expected by RED-SDS

    1) Their model trains on minibatches. 
    2) Their model requires a different data representation:
        torch tensors rather than numpy arrays with permuted dimensions
    3) ...etc.
    
    However, we deliver data in a way that matches experiments for the HSRDM paper:

    1) We feed their model minibatches that never straddle an example boundary.
    """
    def __init__(self, coords, example_stop_idxs, traj_length):
        self.coords = coords
        self.example_stop_idxs = example_stop_idxs
        self.traj_length = traj_length

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        """
        Returns:
            traj, torch.Tensor with shape 
                (traj_length, num_players * court_dims=20)
        """
        T = len(self.coords)
        num_players  = 10
        court_dims = 2
        total_dim =  num_players * court_dims 
        traj = torch.zeros((self.traj_length, total_dim))
        ### make item by finding timestep interval that doesn't overlap with example boundaries
        next_example_stop_idx, t_end = -np.inf, np.inf
        while next_example_stop_idx < t_end:
            t_start = random.randint(0, T - self.traj_length)
            next_example_stop_idx = next((item for item in self.example_stop_idxs if item > t_start), None)
            t_end = t_start + self.traj_length

        coords = self.coords[t_start:t_start + self.traj_length]
        # "flatten" 10 players x 2 court dims into 20 dims
        # player 0's (x,y), player 1's (x,y), etc.
        traj =  torch.Tensor(coords).reshape(self.traj_length, total_dim) 
        return traj


DATA_DIR = "data/basketball/baller2vec_format/processed/"

def make_basketball_dataset_train(
    data_type : str, traj_length: int,
) -> Basketball_Dataset_Train:
    
    """
    Arguments:
        data_type: str, in ["train_1", "train_5", "train_20"]
    """

    if data_type == "train_1":
        coords_filepath = os.path.join(DATA_DIR, "player_coords_train__with_1_games.npy")
        example_stop_idxs_filepath = os.path.join(DATA_DIR, "example_stop_idxs_train__with_1_games.npy")
    elif data_type == "train_5":
        coords_filepath = os.path.join(DATA_DIR, "player_coords_train__with_5_games.npy")
        example_stop_idxs_filepath = os.path.join(DATA_DIR, "example_stop_idxs_train__with_5_games.npy")
    elif data_type == "train_20":
        coords_filepath = os.path.join(DATA_DIR, "player_coords_train__with_20_games.npy")
        example_stop_idxs_filepath = os.path.join(DATA_DIR, "example_stop_idxs_train__with_20_games.npy")
    else: 
        raise ValueError(f"I don't understand data_type {data_type}, which should tell me whether to train or test"
                         f"and also the training size.  See function docstring.")  

    coords = np.load(coords_filepath)
    example_stop_idxs = np.load(example_stop_idxs_filepath)

    # Create a dataset which has __getitem__ defined to give a batch in the form expected by GroupNet
    return Basketball_Dataset_Train(coords, example_stop_idxs, traj_length)


def get_basketball_test_start_and_stop_idxs():

    # TODO: Make this not hardcoded... get it from somewhere else.
    root_dir="/Users/miw267/Repos/team-dynamics-time-series/results/basketball/CLE_starters/artifacts/rebuttal_L=5_K=10_model_type_Linear_And_Out_Of_Bounds_Entity_Recurrence__and__All_Player_Locations_System_Recurrence_train_1_CAVI_its_20_timestamp__12-04-2023_21h10m50s_forecasts_random_forecast_starting_points_True_T_forecast_30/"
    subdirs = [name for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))]


    # Regular expression to extract start and stop idx
    pattern = re.compile(r"start_idx_(\d+)_stop_idx_(\d+)")

    start_idxs=[]
    stop_idxs=[]
    for d in subdirs:
        match = pattern.search(d)
        if match:
            start_idx, stop_idx = map(int, match.groups())
            start_idxs.append(start_idx)
            stop_idxs.append(stop_idx)
        else:
            raise ValueError

    return start_idxs, stop_idxs

def make_basketball_dataset_test__as_list() -> List[torch.Tensor]:    
    """
    Returns:
        List of 78 examples.  Each list element is an torch Tensor of shape
            (T_e, J*D),
        where 
            T_e is the number of timesteps in that example, 
            J is the number of players,
            D is the court dimension (2)
    """
    coords_filepath = os.path.join(DATA_DIR, "player_coords_test__with_5_games.npy")
    example_stop_idxs_filepath = os.path.join(DATA_DIR, "example_stop_idxs_test__with_5_games.npy")
    coords = np.load(coords_filepath)
    example_stop_idxs = np.load(example_stop_idxs_filepath)

    start_idxs, stop_idxs = get_basketball_test_start_and_stop_idxs()
    if not (len(start_idxs)==len(stop_idxs)==78):
        raise ValueError("There should be 78 examples in the test set.")

    num_players, court_dims = 10, 2
    total_dim =  num_players * court_dims 

    test_set_list = [None]*78
    for e in range(78):
        start_idx, stop_idx = start_idxs[e], stop_idxs[e]
        T_example = stop_idx-start_idx
        # we use 1 for the batch dimension, to be consistent with the rest of the code.
        test_set_list[e]=torch.Tensor(coords[start_idx:stop_idx]).reshape(1,T_example, total_dim)
    return test_set_list
