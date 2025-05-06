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

forecasts_dir="results/basketball/basketball_post_multivariate_forecasting_bug_fix/forecasts/"
#forecasts_dir="results/basketball/2025_05_05_174256/forecasts/"


for n_train in [1,5,20]:

    E,S,T_pred,J,D = 78,20,30,10,2
    forecasts_test=np.zeros((E,S,T_pred,J,D))

    for j in range(10):
        basename=f"forecasts_test__SNLDS__n_train_{n_train}__step_30000__player_{j}.npy"
        path=os.path.join(forecasts_dir,basename)
        forecasts_test_for_player_j=np.load(path) # shape (78, 20, 30, 1, 2)
        forecasts_test[:,:,:,j,:]=forecasts_test_for_player_j[:,:,:,0,:]

    basename_stacked=f"forecasts_test__SNLDS__n_train_{n_train}__step_30000_all_players_individ_strat.npy"
    save_path_stacked=os.path.join(forecasts_dir,basename_stacked)
    np.save(save_path_stacked,forecasts_test)