import os

import numpy as np 


"""
Goal: 
    When using the INDEPENDENT TRAINING strategy 
    (n_players=1, which gives one model for each of 10 players),
    we obtain one .npy file with forecasts for each of 10 players.
    
    Each one has shape  
        (E,S,T_pred,J_single,D) = (78,20,30,1,2)
    
    We want to "stack" these 10 forecast files into a single forecast file with shape
        (E,S,T_pred,J,D) = (78,20,30,10,2)
"""

def stack_forecasts(forecasts_dir: str, n_train: int) -> None:
    # Function 
    E,S,T_pred,J,D = 78,20,30,10,2
    forecasts_test=np.zeros((E,S,T_pred,J,D))

    for j in range(5):
        basename=f"forecasts_test__SNLDS__n_train_{n_train}__step_20000__player_{j}.npy"
        path=os.path.join(forecasts_dir,basename)
        forecasts_test_for_player_j=np.load(path) # shape (78, 20, 30, 1, 2)
        forecasts_test[:,:,:,j,:]=forecasts_test_for_player_j[:,:,:,0,:]

    basename_stacked=f"forecasts_test__SNLDS__n_train_{n_train}__step_20000_all_players_individ_strat.npy"
    save_path_stacked=os.path.join(forecasts_dir,basename_stacked)
    np.save(save_path_stacked,forecasts_test)


###
# Main 
###
forecasts_dir="results/basketball/basketball_TMLR_rebuttal_n_train_games__1/forecasts"
n_train = 1 
stack_forecasts(forecasts_dir, n_train)

forecasts_dir="results/basketball/basketball_TMLR_rebuttal_n_train_games__5_and_20/forecasts"
n_train = 5 
stack_forecasts(forecasts_dir, n_train)

forecasts_dir="results/basketball/basketball_TMLR_rebuttal_n_train_games__5_and_20/forecasts"
n_train = 20
stack_forecasts(forecasts_dir, n_train)
