import numpy as np
from scipy.optimize import curve_fit
from concurrent.futures import ThreadPoolExecutor
from .factor_analysis import split_shared_residual_part
import xarray as xr
from scipy.special import comb
from itertools import combinations

# import dask


def var_cc_of_session(session, num_bootstrap, num_fits=10):
    """takes session and computes var cc"""
    hit_trials = session.sel(trial_type="hit").trial_num.values
    miss_trials = session.sel(trial_type="miss").trial_num.values

    res1 = var_cc_bias_corrected(session, hit_trials, num_fits=num_fits)
    var_cc_hit = [res1[0]]
    mean_var_hit = [res1[1]]
    mean_rho_hit = [res1[2]]

    if num_bootstrap:
        subsample = min(len(hit_trials), len(miss_trials))
        res2 = bootstrap_session(
            var_cc_bias_corrected,
            session,
            hit_trials,
            num_bootstrap,
            subsample=subsample,
            num_fits=num_fits,
        )
        var_cc_hit += list(zip(*res2))[0]
        mean_var_hit += list(zip(*res2))[1]
        mean_rho_hit += list(zip(*res2))[2]

    res1 = var_cc_bias_corrected(session, miss_trials, num_fits=num_fits)
    var_cc_miss = [res1[0]]
    mean_var_miss = [res1[1]]
    mean_rho_miss = [res1[2]]

    if num_bootstrap:
        res2 = bootstrap_session(
            var_cc_bias_corrected,
            session,
            miss_trials,
            num_bootstrap,
            subsample=subsample,
            num_fits=num_fits,
        )
        var_cc_miss += list(zip(*res2))[0]
        mean_var_miss += list(zip(*res2))[1]
        mean_rho_miss += list(zip(*res2))[2]

    return var_cc_hit, var_cc_miss, mean_var_hit, mean_var_miss, mean_rho_hit, mean_rho_miss

def var_cc_of_session_HitAndMiss(session, num_bootstrap, num_fits=10):
    HitAndMiss_trials = session.where(session.trial_type.isin(("hit", "miss")), drop=True).trial_num.values
    
    res1 = var_cc_bias_corrected(session, HitAndMiss_trials, num_fits=num_fits)
    var_cc_HitAndMiss = [res1[0]]
    mean_var_HitAndMiss = [res1[1]]
    mean_rho_HitAndMiss = [res1[2]]

    if num_bootstrap:
        res2 = bootstrap_session(
            var_cc_bias_corrected,
            session,
            HitAndMiss_trials,
            num_bootstrap,
            subsample=subsample,
            num_fits=num_fits,
        )
        var_cc_HitAndMiss += list(zip(*res2))[0]
        mean_var_HitAndMiss += list(zip(*res2))[1]
        mean_rho_HitAndMiss += list(zip(*res2))[2]
        
    return var_cc_HitAndMiss, mean_var_HitAndMiss, mean_rho_HitAndMiss

def var_cc_of_session_TestOnly_HitAndMiss(session, num_bootstrap, num_fits=10):
    HitAndMiss_trials = session.where(session.trial_type.isin(("hit", "miss")), drop=True).trial_num.values
    Test_trials = session.where(session.stim_type.isin((1)), drop=True).trial_num.values
    
    TestOnly_HitAndMiss_trials = np.intersect1d(HitAndMiss_trials, Test_trials)
    
    res1 = var_cc_bias_corrected(session, TestOnly_HitAndMiss_trials, num_fits=num_fits)
    var_cc_HitAndMiss = [res1[0]]
    mean_var_HitAndMiss = [res1[1]]
    mean_rho_HitAndMiss = [res1[2]]

    if num_bootstrap:
        res2 = bootstrap_session(
            var_cc_bias_corrected,
            session,
            TestOnly_HitAndMiss_trials,
            num_bootstrap,
            subsample=subsample,
            num_fits=num_fits,
        )
        var_cc_TestOnly_HitAndMiss += list(zip(*res2))[0]
        mean_var_TestOnly_HitAndMiss_trials += list(zip(*res2))[1]
        mean_rho_TestOnly_HitAndMiss_trials += list(zip(*res2))[2]
        
    return var_cc_TestOnly_HitAndMiss_trials, mean_var_TestOnly_HitAndMiss_trials, mean_rho_TestOnly_HitAndMiss_trials


def prepost_var_cc_and_taupost_of_session_TestOnly_Hit(session, num_bootstrap, num_fits=1000):
    Hit_trials = session.where(session.trial_type.isin(("hit")), drop=True).trial_num.values
    Test_trials = session.where(session.stim_type.isin((1)), drop=True).trial_num.values
    
    Test_Hit_trials = np.intersect1d(Hit_trials, Test_trials)
    
    res1 = prepost_var_cc_and_taupost(session, Test_Hit_trials, num_fits=num_fits)
    var_cc_list_TestHit = [res1[0]]
    mean_var_list_TestHit = [res1[1]]
    mean_rho_list_TestHit = [res1[2]]
    taupost_list_TestHit = [res1[3]]

    if num_bootstrap:
        res2 = bootstrap_session(
            var_cc_bias_corrected,
            session,
            Test_Hit_trials,
            num_bootstrap,
            subsample=subsample,
            num_fits=num_fits,
        )
        var_cc_list_TestHit += list(zip(*res2))[0]
        mean_var_list_TestHit += list(zip(*res2))[1]
        mean_rho_list_TestHit += list(zip(*res2))[2]
        taupost_list_TestHit += list(zip(*res2))[3]
        
    return var_cc_list_TestHit, mean_var_list_TestHit, mean_rho_list_TestHit, taupost_list_TestHit

def prepost_var_cc_and_taupost_of_session_TestOnly_Miss(session, num_bootstrap, num_fits=1000):
    Miss_trials = session.where(session.trial_type.isin(("miss")), drop=True).trial_num.values
    Test_trials = session.where(session.stim_type.isin((1)), drop=True).trial_num.values
    
    Test_Miss_trials = np.intersect1d(Miss_trials, Test_trials)
    
    res1 = prepost_var_cc_and_taupost(session, Test_Miss_trials, num_fits=num_fits)
    var_cc_list_TestMiss = [res1[0]]
    mean_var_list_TestMiss = [res1[1]]
    mean_rho_list_TestMiss = [res1[2]]
    taupost_list_TestMiss = [res1[3]]

    if num_bootstrap:
        res2 = bootstrap_session(
            var_cc_bias_corrected,
            session,
            Test_Miss_trials,
            num_bootstrap,
            subsample=subsample,
            num_fits=num_fits,
        )
        var_cc_list_TestMiss += list(zip(*res2))[0]
        mean_var_list_TestMiss += list(zip(*res2))[1]
        mean_rho_list_TestMiss += list(zip(*res2))[2]
        taupost_list_TestMiss += list(zip(*res2))[3]
        
    return var_cc_list_TestMiss, mean_var_list_TestMiss, mean_rho_list_TestMiss, taupost_list_TestMiss

def prepost_var_cc_and_taupost_of_session_TestOnly_HitAndMiss(session, num_bootstrap, num_fits=1000):
    HitAndMiss_trials = session.where(session.trial_type.isin(("hit","miss")), drop=True).trial_num.values
    Test_trials = session.where(session.stim_type.isin((1)), drop=True).trial_num.values
    
    Test_HitAndMiss_trials = np.intersect1d(HitAndMiss_trials, Test_trials)
    
    res1 = prepost_var_cc_and_taupost(session, Test_HitAndMiss_trials, num_fits=num_fits)
    var_cc_list_TestHitAndMiss = [res1[0]]
    mean_var_list_TestHitAndMiss = [res1[1]]
    mean_rho_list_TestHitAndMiss = [res1[2]]
    taupost_list_TestHitAndMiss = [res1[3]]

    if num_bootstrap:
        res2 = bootstrap_session(
            var_cc_bias_corrected,
            session,
            Test_HitAndMiss_trials,
            num_bootstrap,
            subsample=subsample,
            num_fits=num_fits,
        )
        var_cc_list_TestHitAndMiss += list(zip(*res2))[0]
        mean_var_list_TestHitAndMiss += list(zip(*res2))[1]
        mean_rho_list_TestHitAndMiss += list(zip(*res2))[2]
        taupost_list_TestHitAndMiss += list(zip(*res2))[3]
        
    return var_cc_list_TestHitAndMiss, mean_var_list_TestHitAndMiss, mean_rho_list_TestHitAndMiss, taupost_list_TestHitAndMiss

def bootstrap_session(
    func, session, trials_to_choose_from, num_bootstrap, *args, subsample=None, **kwargs
):
    res_list = []
    if subsample is not None:
        replace = False
        num_samples = subsample
    else:
        replace = True
        num_samples = num_bootstrap
    for _ in range(num_bootstrap):
        current_trials = np.random.choice(
            trials_to_choose_from, num_samples, replace=replace
        )
        res_list.append(func(session, current_trials, *args, **kwargs))
    return res_list


def var_cc_bias_corrected(session, trial_list, num_fits=10, n_choose=15):
    def inner_loop(trials_chosen):
        var_cc_one_fit, mean_var, mean_rho = var_cc_bias_corrected_one_fit(
            session.isel(trial=trials_chosen).values
        )
        return var_cc_one_fit, mean_var, mean_rho

    with ThreadPoolExecutor(max_workers=10) as executor:
        random_trials = [
            np.random.choice(trial_list, size=n_choose, replace=False)
            for _ in range(num_fits)
        ]
        results = executor.map(inner_loop, random_trials)

    var_cc_list, mean_var_list, mean_rho_list = list(zip(*results))
    return (
        np.mean(var_cc_list),
        np.mean(mean_var_list),
        np.mean(mean_rho_list),
        np.std(var_cc_list) / np.sqrt(num_fits),
    )

def prepost_var_cc_and_taupost(session, trial_list, num_fits=1000, n_choose=10):
    def inner_loop_var_cc(trials):
        
        # select appropriate fluorescence array
        F = session.isel(trial=trials).values
        
        
        # re-bin fluorescence array
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
        
        # select appropiate frames (pre-stim)
        time_range = np.array([-8000, 6000])  # in milli-seconds
        
        pre_time_range_wanted = np.array([-7000, -500])  # in milli-seconds
        pre_freq = 30 // subsampling_fact
        assert np.diff(time_range) * pre_freq // 1000 == F_rebinned.shape[2]
        pre_index_wanted = (pre_time_range_wanted - time_range[0]) * pre_freq // 1000
        pre_frames = np.arange(pre_index_wanted[0],pre_index_wanted[1])
        
        # calculate observables
        var_cc_one_fit, mean_var, mean_rho = var_cc_bias_corrected_one_fit(F_rebinned[:,:,pre_frames])
        
        return var_cc_one_fit, mean_var, mean_rho
    
    
    def inner_loop_taupost(trials, targets):
        # select appropriate fluorescence array
        F = session.isel(trial=trials).values
        
        subsampling_fact = 1
        
        # select appropiate frames (post-stim)
        time_range = np.array([-8000, 6000])  # in milli-seconds
        
        pre_time_range_wanted = np.array([-8000, 0])  # in milli-seconds
        pre_freq = 30 // subsampling_fact
        assert np.diff(time_range) * pre_freq // 1000 == F.shape[2]
        pre_index_wanted = (pre_time_range_wanted - time_range[0]) * pre_freq // 1000
        pre_frames = np.arange(pre_index_wanted[0],pre_index_wanted[1])
        
        post_time_range_wanted = np.array([325, 5825])  # in milli-seconds
        post_freq = 30 // subsampling_fact
        assert np.diff(time_range) * post_freq // 1000 == F.shape[2]
        post_index_wanted = (post_time_range_wanted - time_range[0]) * post_freq // 1000
        post_frames = np.arange(post_index_wanted[0],post_index_wanted[1])
        
        # calculate observable
        taupost = taupost_one_fit(F[:,:,:], targets, pre_frames[:-1], post_frames[1:])
        
        return taupost

    
    with ThreadPoolExecutor(max_workers=10) as executor:
        
        # compute number of combinations
        print(trial_list)
        num_combs = int(comb(len(trial_list), n_choose))
        print(num_combs)
        
        # if number of combinations smaller than num_fits, use all of them
        if num_combs < num_fits:
            list_of_tuples = list(combinations(trial_list, n_choose))
            trials_chosen = [np.asarray(elem) for elem in list_of_tuples]
            #print(trials_chosen)
            
        # if number of combinations larger than num_fits, only use a subsample of size (num_fits,) 
        else:    
            trials_chosen = [
                np.random.choice(trial_list, size=n_choose, replace=False)
                for _ in range(num_fits)
            ]
        
        # prepare is_target for propagation into the inner loop
        is_target = session.is_target.values
        targets_chosen = [is_target[:,trials] for trials in trials_chosen]
        
        # start inner loops
        var_cc_results = executor.map(inner_loop_var_cc, trials_chosen)
        taupost_results = executor.map(inner_loop_taupost, trials_chosen, targets_chosen)

    # get results from inner loops          
    var_cc_list, mean_var_list, mean_rho_list = list(zip(*var_cc_results))
    taupost_list = list(taupost_results)
    
    # pad results with nans in case that there are less combination than num_fits 
    var_cc_array = np.zeros(num_fits)
    var_cc_array[:] = np.nan
    var_cc_array[:len(var_cc_list)] = var_cc_list
    
    mean_var_array = np.zeros(num_fits)
    mean_var_array[:] = np.nan
    mean_var_array[:len(mean_var_list)] = mean_var_list
    
    mean_rho_array = np.zeros(num_fits)
    mean_rho_array[:] = np.nan
    mean_rho_array[:len(mean_rho_list)] = mean_rho_list
    
    taupost_array = np.zeros(num_fits)
    taupost_array[:] = np.nan
    taupost_array[:len(taupost_list)] = taupost_list
    
    return (
        var_cc_array,
        mean_var_array, 
        mean_rho_array,
        taupost_array
    )

def var_cc_bias_corrected_one_fit(arr):
    # arr has shape neurons, trials, time
    
    num_trials = arr.shape[1]

    time = []
    var_cc_with_bias_list = []
    for current_num_trials in range(5, num_trials + 1):
        # num_points = np.ceil(num_trials / current_num_trials).astype("int") * 10
        num_points = 50
        var_cc_with_bias_timestep = []
        for _ in range(num_points):
            current_trials = np.random.choice(
                np.arange(num_trials), current_num_trials, replace=False
            )
            var_cc_with_bias_timestep += [
                calc_var_cc(arr[:, current_trials, :].reshape((arr.shape[0], -1)))
            ]
        time += [current_num_trials]
        var_cc_with_bias_list += [np.mean(var_cc_with_bias_timestep)]

    popt, _ = curve_fit(analytical_offset, time, var_cc_with_bias_list)
    var_cc_without_bias = popt[0]

    arr_reshaped = arr.reshape((arr.shape[0], -1))
    mean_var = np.mean(np.diag(np.cov(arr_reshaped)))
    mean_rho = calc_mean_rho(arr_reshaped)

    return var_cc_without_bias, mean_var, mean_rho

def taupost_one_fit(arr, targets, pre_frames, post_frames):
    # arr has shape neurons, trials, time
    
    # select all neurons that were ever targeted
    targeted_neurons = np.any(targets[:,:], axis=1)
    
    # find appropriate size for stim-locked fluorescence array
    n_neurons = arr.shape[0]
    n_trials = arr.shape[1]
    n_frames = arr.shape[2]
    
    # create stim-locked fluorescence array & fill with nans
    TargetNeuronAverages = np.zeros((n_neurons,n_frames))
    TargetNeuronAverages[:] = np.nan
    
    # create trial array to index later
    trials = np.arange(n_trials)
    
    #iterate on level of single neuron and collect responses on trials where it was actually stimulated
    for i_neuron in np.arange(n_neurons)[targeted_neurons]:
        TargetNeuronAverages[i_neuron] = np.nanmean(arr[i_neuron][trials[targets[i_neuron]]], axis=0)
    
    Target_delta = np.nanmean(TargetNeuronAverages, axis=0) - np.nanmean(np.nanmean(TargetNeuronAverages, axis=0)[0:pre_frames[-1]])
    
    t = post_frames

    try:
        Target_popt, _ = curve_fit(analytical_decay, t, 
                                    Target_delta[post_frames], 
                                    p0=(1, 100),
                                    maxfev = 10000)

        taupost = Target_popt[1]
    except:
        taupost = np.inf
        
    return taupost

def analytical_offset(x, offset, A):
    y = offset ** 2 + A / x
    return y

def analytical_decay(t, A, tau):
    y= A * np.exp(-t/tau)
    return y


# @dask.delayed
def calc_var_cc(arr):
    # arr has shape (neurons, time)

    corr_coef = np.cov(arr)
    mask = mask_upper_triangle(corr_coef)
    var = np.var(corr_coef[mask])
    return var

def calc_mean_rho(arr):
    corr_coef = np.corrcoef(arr)
    mask = mask_upper_triangle(corr_coef)
    mean = np.mean(corr_coef[mask])
    return mean

def calc_R(mu_V, sigma_CC, N=100000):
    s = sigma_CC / mu_V
    
    f_R = lambda s, N: np.sqrt(1-np.sqrt(1/(1+N * s**2)))
    
    

def mask_upper_triangle(matrix):
    assert matrix.ndim == 2
    assert matrix.shape[1] == matrix.shape[0]

    col_index = np.tile(np.arange(len(matrix)), (len(matrix), 1))
    row_index = np.tile(np.arange(len(matrix))[:, np.newaxis], (1, len(matrix)))

    mask = col_index > row_index
    return mask
