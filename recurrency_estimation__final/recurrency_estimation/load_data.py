import os
import sys
import numpy as np
import xarray as xr
from tqdm import tqdm

sys.path.append("../../RowlandEtAl/popping-off/popoff/popoff/")
sys.path.append("../../RowlandEtAl/popping-off/popoff/")
sys.path.append("../../RowlandEtAl/popping-off/scripts/")

sys.path.append("../../RowlandEtAl/Vape/utils/")
sys.path.append("../../RowlandEtAl/Vape/")

import popoff

import pop_off_functions as pof
import pop_off_plotting as pop
from popoff import loadpaths
from Session import SessionLite
from linear_model import PoolAcrossSessions, LinearModel, MultiSessionModel


def load_data(remove_toosoon=False, label_urh_arm=False):
    ## save flu to scratch
    pas = PoolAcrossSessions(
        remove_targets=False,
        subsample_sessions=False,
        remove_toosoon=remove_toosoon,
        pre_start=-8,
    )

    if label_urh_arm:
        ## Create sessions object from PAS:
        sessions = {}
        i_s = 0
        for (
            ses
        ) in (
            pas.sessions.values()
        ):  # load into sessions dict (in case pas skips an int as key)
            ses.signature = f"{ses.mouse}_R{ses.run_number}"
            sessions[i_s] = ses
            i_s += 1
        print(sessions)
        assert len(sessions) == 11
        pof.label_urh_arm(sessions=sessions)

    session_dict = {}
    for i_session in tqdm(range(11)):
        F = pas.linear_models[i_session].flu
        # with shape (neurons, trials, time)
        subsampling_fact = 3
        F_rebinned = np.sum(
            F.reshape(
                (
                    F.shape[0],
                    F.shape[1],
                    F.shape[2] // subsampling_fact,
                    subsampling_fact,
                )
            ),
            axis=-1,
        )

        time_range = np.array([-8000, 6000])  # in milli-seconds
        time_range_wanted = np.array([-7000, -500])  # in milli-seconds
        freq = 30 // subsampling_fact
        assert np.diff(time_range) * freq // 1000 == F_rebinned.shape[2]
        index_wanted = (time_range_wanted - time_range[0]) * freq // 1000

        F_rebinned = F_rebinned[:, :, index_wanted[0] : index_wanted[1]]

        area = np.zeros(F_rebinned.shape[0], dtype="int")
        area[pas.linear_models[i_session].session.s1_bool] = 1
        area[pas.linear_models[i_session].session.s2_bool] = 2
        trial_type = pas.linear_models[i_session].session.outcome
        stim_type = pas.linear_models[i_session].session.photostim

        data_session = xr.DataArray(
            F_rebinned,
            dims=("neuron", "trial", "time"),
            coords={
                "neuron_num": ("neuron", range(F_rebinned.shape[0])),
                "trial_num": ("trial", range(F_rebinned.shape[1])),
                "area": ("neuron", area),
                "trial_type": ("trial", trial_type),
                "stim_type": ("trial", stim_type),
            },
        )
        data_session = data_session.set_index(
            {"neuron": ("neuron_num", "area"), "trial": ("trial_num", "trial_type", "stim_type")}
        )
        session_dict[str(pas.linear_models[i_session].session)] = data_session
    return session_dict

def load_data_prepost(remove_toosoon=False, label_urh_arm=False):
    ## save flu to scratch
    pas = PoolAcrossSessions(
        remove_targets=False,
        subsample_sessions=False,
        remove_toosoon=remove_toosoon,
        pre_start=-8,
    )

    if label_urh_arm:
        ## Create sessions object from PAS:
        sessions = {}
        i_s = 0
        for (
            ses
        ) in (
            pas.sessions.values()
        ):  # load into sessions dict (in case pas skips an int as key)
            ses.signature = f"{ses.mouse}_R{ses.run_number}"
            sessions[i_s] = ses
            i_s += 1
        print(sessions)
        assert len(sessions) == 11
        pof.label_urh_arm(sessions=sessions)

    session_dict = {}
    for i_session in tqdm(range(11)):
        F = pas.linear_models[i_session].flu
        # with shape (neurons, trials, time)
        subsampling_fact = 1
        F_rebinned = np.sum(
            F.reshape(
                (
                    F.shape[0],
                    F.shape[1],
                    F.shape[2] // subsampling_fact,
                    subsampling_fact,
                )
            ),
            axis=-1,
        )

        time_range = np.array([-8000, 6000])  # in milli-seconds
        
        pre_time_range_wanted = np.array([-7000, -500])  # in milli-seconds
        freq = 30 // subsampling_fact
        assert np.diff(time_range) * freq // 1000 == F_rebinned.shape[2]
        pre_index_wanted = (pre_time_range_wanted - time_range[0]) * freq // 1000
        pre_frames = np.arange(pre_index_wanted[0],pre_index_wanted[1])
        
        post_time_range_wanted = np.array([325, 5825])  # in milli-seconds
        freq = 30 // subsampling_fact
        assert np.diff(time_range) * freq // 1000 == F_rebinned.shape[2]
        post_index_wanted = (post_time_range_wanted - time_range[0]) * freq // 1000
        post_frames = np.arange(post_index_wanted[0] - 1,post_index_wanted[1] - 1)

        area = np.zeros(F_rebinned.shape[0], dtype="int")
        
        area[pas.linear_models[i_session].session.s1_bool] = 1
        area[pas.linear_models[i_session].session.s2_bool] = 2
        
        trial_type = pas.linear_models[i_session].session.outcome
        stim_type = pas.linear_models[i_session].session.photostim
        
        frame_type = np.zeros(F_rebinned.shape[2], dtype="int")
        frame_type[pre_frames] = -1
        frame_type[post_frames] = 1
        
        is_target = pas.linear_models[i_session].session.is_target[:,:,0]
        
        data_session = xr.DataArray(
            F_rebinned,
            dims=("neuron", "trial", "time"),
            coords={
                "neuron_num": ("neuron", np.arange(F_rebinned.shape[0])),
                "trial_num": ("trial", np.arange(F_rebinned.shape[1])),
                "frame_num": ("time", np.arange(F_rebinned.shape[2])),
                "area": ("neuron", area),
                "trial_type": ("trial", trial_type),
                "stim_type": ("trial", stim_type),
                "frame_type" : ("time", frame_type),
                "is_target" : (("neuron","trial"), is_target),
            },
        )
        data_session = data_session.set_index(
            {"neuron": ("neuron_num", "area"), 
             "trial": ("trial_num", "trial_type", "stim_type"), 
             "time": ("frame_num", "frame_type")}
        )
        session_dict[str(pas.linear_models[i_session].session)] = data_session
    return session_dict
