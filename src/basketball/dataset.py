
import os
import re
import numpy as np
import random 

from typing import List, Optional 

import torch
from torch.utils.data import Dataset, DataLoader

from src.basketball.examples import get_start_and_stop_timestep_idxs_from_event_idx

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
    def __init__(self, coords, example_stop_idxs, traj_length, n_players, player_idx: Optional[int]=None):

        if n_players not in [1,10]:
            raise ValueError("n_players must be 1 or 10")

        if player_idx not in [0,1,2,3,4,5,6,7,8,9]:
            raise ValueError("player_idx must be between 0 and 9, inclusive")
        
        
        self.coords = coords
        self.example_stop_idxs = example_stop_idxs
        self.traj_length = traj_length
        self.n_players = n_players 
        self.player_idx = player_idx 
        
    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        """
        Returns:
            traj, torch.Tensor with shape 
                (traj_length, num_players * court_dims=20)
        """
        T = len(self.coords)
        court_dims = 2
        total_dim =  self.n_players * court_dims 
        traj = torch.zeros((self.traj_length, total_dim))
        ### make item by finding timestep interval that doesn't overlap with example boundaries
        next_example_stop_idx, t_end = -np.inf, np.inf
        while next_example_stop_idx < t_end:
            t_start = random.randint(0, T - self.traj_length)
            next_example_stop_idx = next((item for item in self.example_stop_idxs if item > t_start), None)
            t_end = t_start + self.traj_length

        if self.n_players==10:
            coords = self.coords[t_start:t_start + self.traj_length, :10,:] #shape (T, J, D)
        elif self.n_players==1:
            coords = self.coords[t_start:t_start + self.traj_length, self.player_idx,:]
            # ensure that we preserve the second dimension for player
            coords = coords[:,None,:] #shape (T, J, D)

        # "flatten" (n_players,2) court dims into (n_players*2) dims
        # player 0's (x,y), player 1's (x,y), etc.
        traj =  torch.Tensor(coords).reshape(self.traj_length, total_dim) 
        return traj


DATA_DIR = "data/basketball/baller2vec_format/processed/"


def make_basketball_dataset_train(
    n_train_games: int, traj_length: int, n_players: int, player_idx: int, 
) -> Basketball_Dataset_Train:
    
    """
    Arguments:
        data_type: str, in ["train_1", "train_5", "train_20"]
    """

    if n_train_games==1:
        coords_filepath = os.path.join(DATA_DIR, "player_coords_train__with_1_games.npy")
        example_stop_idxs_filepath = os.path.join(DATA_DIR, "example_stop_idxs_train__with_1_games.npy")
    elif n_train_games==5:
        coords_filepath = os.path.join(DATA_DIR, "player_coords_train__with_5_games.npy")
        example_stop_idxs_filepath = os.path.join(DATA_DIR, "example_stop_idxs_train__with_5_games.npy")
    elif n_train_games==20:
        coords_filepath = os.path.join(DATA_DIR, "player_coords_train__with_20_games.npy")
        example_stop_idxs_filepath = os.path.join(DATA_DIR, "example_stop_idxs_train__with_20_games.npy")
    else: 
        raise ValueError(f"I don't understand num of training games {n_train_games}, which should tell me whether to train or test"
                         f"and also the training size.  See function docstring.")  

    coords = np.load(coords_filepath)
    example_stop_idxs = np.load(example_stop_idxs_filepath)

    # Create a dataset which has __getitem__ defined to give a batch in the form expected by GroupNet
    return Basketball_Dataset_Train(coords, example_stop_idxs, traj_length, n_players, player_idx)


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


def FROM_OTHER_REPO_REMOVE_make_context_sets_in_meters(past_length) -> torch.Tensor:
    """
    context sets looks like past_trajs
    has shape:   (num_examples, num_players_plus_ball=11, past_length, court_dims=2)
    """

    """load stuff to make forecast"""
    coords_filepath = os.path.join(DATA_DIR, "player_coords_test__with_5_games.npy")
    example_stop_idxs_filepath = os.path.join(DATA_DIR, "example_stop_idxs_test__with_5_games.npy")
    random_context_times_filepath = os.path.join(DATA_DIR, "random_context_times.npy")

    xs_test = np.load(coords_filepath)
    example_end_times_test = np.load(example_stop_idxs_filepath)
    random_context_times = np.load(random_context_times_filepath)

    """ make forecasts"""
    event_idxs_to_analyze = [
        i for (i, random_context_time) in enumerate(random_context_times) if not np.isnan(random_context_time)
    ]

    num_examples = len(event_idxs_to_analyze)
    num_players = 10 
    num_players_plus_ball=11
    court_dims=2

    context_sets_normalized = np.zeros((num_examples, num_players_plus_ball, past_length, court_dims))

    for e, event_idx_to_analyze in enumerate(event_idxs_to_analyze):
        start_idx, stop_idx = get_start_and_stop_timestep_idxs_from_event_idx(
            example_end_times_test, event_idx_to_analyze
        )
        xs_test_example = xs_test[start_idx:stop_idx]
        T_context = int(random_context_times[event_idx_to_analyze])

        context_sets_normalized[e:,:num_players] = xs_test_example[T_context-past_length: T_context].swapaxes(0,1) # pre-swap shape (T,J,D); post-swap: (J,T,D)

    context_sets = unnormalize_coords_to_meters(context_sets_normalized)
    return torch.Tensor(context_sets)


def make_basketball_dataset_test__as_list(n_players: int, player_idx: Optional[int]=None) -> List[torch.Tensor]:    
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

    court_dims = 2
    total_dim =  n_players * court_dims 

    test_set_list = [None]*78
    for e in range(78):
        start_idx, stop_idx = start_idxs[e], stop_idxs[e]
        T_example = stop_idx-start_idx
        # we use 1 for the batch dimension, to be consistent with the rest of the code.

        if n_players==10:
            coords_clip=coords[start_idx:stop_idx, :10] #shape (T, J, D)
        elif n_players==1:
             # ensure that we preserve the second dimension for player
            coords_clip = coords[start_idx:stop_idx, player_idx][:,None,:] #shape (T, J, D)
        test_set_list[e]=torch.Tensor(coords_clip).reshape(1,T_example, total_dim)
    return test_set_list

def make_basketball_dataset_test_context_set__as_list(n_players: int, player_idx: Optional[int]=None) -> List[torch.Tensor]:    
    """
    Returns:
        List of 78 examples.  Each list element is an torch Tensor of shape
            (T_e, J*D),
        where 
            T_e is the number of timesteps in that example, 
            J is the number of players,
            D is the court dimension (2)
    """
    if n_players!=1:
        raise NotImplementedError(f"This used to be implemented, but it got phased out. "
                                  f"It needs to be brought back in.  That is, as of now, "
                                  f"we're using the independent entity-to-system strategy.")
    if player_idx is None:
        raise NotImplementedError(f"Current implementation assumes n_players=1, "
                                  f"so we need to specify which player_idx to train on.")

    coords_filepath = os.path.join(DATA_DIR, "player_coords_test__with_5_games.npy")
    example_stop_idxs_filepath = os.path.join(DATA_DIR, "example_stop_idxs_test__with_5_games.npy")
    random_context_times_filepath = os.path.join(DATA_DIR, "random_context_times.npy")

    coords = np.load(coords_filepath)
    example_end_times_test  = np.load(example_stop_idxs_filepath)
    random_context_times = np.load(random_context_times_filepath)

    event_idxs_to_analyze = [
        i for (i, random_context_time) in enumerate(random_context_times) if not np.isnan(random_context_time)
    ]

    # TODO: don't hardcode past_length
    num_examples = len(event_idxs_to_analyze)
    num_players_total = 10 
    court_dims = 2
    past_length = 20 

    context_sets_all_players = np.zeros((num_examples, num_players_total, past_length, court_dims))

    for e, event_idx_to_analyze in enumerate(event_idxs_to_analyze):
        start_idx, stop_idx = get_start_and_stop_timestep_idxs_from_event_idx(
            example_end_times_test, event_idx_to_analyze
        )
        xs_test_example = coords[start_idx:stop_idx]
        T_context = int(random_context_times[event_idx_to_analyze])
        context_sets_all_players[e:] = xs_test_example[T_context-past_length: T_context].swapaxes(0,1) # pre-swap shape (T,J,D); post-swap: (J,T,D)

    # context sets looks like past_trajs
    # has shape:   (num_examples, num_players_total=10, past_length, court_dims=2)
    # 
    # for consistency with how the rest of the code in this repo is structured,
    # here i convert to a list of E examples where each example has shape
    #   (n_players, past_length, dim_of_court)
    # where `n_players` is the num of players used to train this model.  as of now we are forcing
    # n_players=1 as the only valid argument, so we train on one player at a time.
    #
    # the wrapping list is legacy construction, as previously each example was allowed
    # to have a different length 

    total_dim =  n_players * court_dims  # `n_players` is the num of players used to train this model

    context_set_list = [None]*num_examples
    for e in range(num_examples):
        coords_clip = context_sets_all_players[e,player_idx] # shape (past_length, D)
        context_set_list[e]=torch.Tensor(coords_clip).reshape(1,past_length, total_dim)
    return context_set_list
