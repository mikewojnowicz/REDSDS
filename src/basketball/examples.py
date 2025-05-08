from typing import Tuple
import numpy as np 

"""
This module copypasta'd from (my fork of) GroupNet repo.
"""

###
# Types
###

NumpyArray1D = np.array 

###
# Helper Functions
###

### The two functions below are used to convert `example_end_times` into pairs
### of indices designating the beginning and end of an event.


def get_start_and_stop_timestep_idxs_from_event_idx__using_one_indexing(
    event_stop_idxs: NumpyArray1D, event_idx_in_one_indexing: int
) -> Tuple[int]:
    """
    Arguments:
        event_stop_idxs: whose usage is very well documented in `run_CAVI_with_JAX`
    """
    start_idx = event_stop_idxs[(event_idx_in_one_indexing - 1)] + 1
    stop_idx = event_stop_idxs[event_idx_in_one_indexing]
    return (start_idx, stop_idx)


def get_start_and_stop_timestep_idxs_from_event_idx(
    event_stop_idxs: NumpyArray1D, event_idx_in_zero_indexing: int
) -> Tuple[int]:
    """
    Arguments:
        event_stop_idxs: whose usage is very well documented in `run_CAVI_with_JAX`
    """
    return get_start_and_stop_timestep_idxs_from_event_idx__using_one_indexing(
        event_stop_idxs, event_idx_in_zero_indexing + 1
    )
