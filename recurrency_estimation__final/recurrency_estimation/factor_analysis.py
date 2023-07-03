from sklearn.decomposition import FactorAnalysis, PCA
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import xarray as xr
from scipy.special import comb
from itertools import combinations


def factor_analysis(train_data, apply_data, n_factors, return_per_var_explained=False):
    # data has shape (time, neurons)
    transformer = FactorAnalysis(n_components=n_factors)  # , random_state=0)
    transformer = transformer.fit(train_data)
    
    if return_per_var_explained:
        shared_var_exps = np.zeros(n_factors+1)
        individual_var_exps = np.zeros(n_factors+1)
        for i_f, factor in enumerate(np.arange(1,n_factors+2)):
            np.random.seed()
            if_transformer = FactorAnalysis(n_components=factor) #, random_state=0)
            if_S_transformed = if_transformer.fit_transform(train_data)
            
            if_n_rows = if_transformer.components_.shape[0]
            if_shared_variance = np.sum(if_transformer.components_**2, axis=1)
            
            shared_var_exps[i_f-1] = np.sum(if_shared_variance)
            individual_var_exps[i_f-1] = np.sum(if_transformer.noise_variance_)
        
        cum_var_explained = shared_var_exps/(shared_var_exps+individual_var_exps)*100
        per_var_explained = np.diff(np.concatenate(([0],cum_var_explained)))[:-1]
        
        return (
            transformer.components_,
            transformer.transform(apply_data),
            transformer.mean_,
            transformer.noise_variance_,
            per_var_explained
        )
    
    else:
        return (
            transformer.components_,
            transformer.transform(apply_data),
            transformer.mean_,
            transformer.noise_variance_
        )

    
def principal_component_analysis(train_data, apply_data, n_factors, return_per_var_explained=False):
    # data has shape (time, neurons)
    transformer = PCA(n_components=n_factors)  # , random_state=0)
    transformer = transformer.fit(train_data)
    
    per_var_explained = transformer.explained_variance_ratio_
    
    if return_per_var_explained:
        return (
            transformer.components_,
            transformer.transform(apply_data),
            transformer.mean_,
            transformer.noise_variance_,
            per_var_explained
        )
    
    else:
        return (
            transformer.components_,
            transformer.transform(apply_data),
            transformer.mean_,
            transformer.noise_variance_
        )

    
def split_shared_residual_part(session, n_fact, fa_type='lfa'):
    arr = session.values.reshape((session.values.shape[0], -1)).transpose()

    # Use only hit and miss trials to fit FA
    session_train = session.where(
        (session.trial_type == "miss") | (session.trial_type == "hit"), drop=True
    )
    train_arr = session_train.values.reshape((session.values.shape[0], -1)).transpose()

    arr = np.nan_to_num(arr)
    train_arr = np.nan_to_num(train_arr)

    max_tries = 3
    for n_try in range(max_tries + 1):
        try:
            if fa_type=='lfa':
                F, L, mean, var = factor_analysis(
                    train_arr, apply_data=arr, n_factors=n_fact
                )
            elif fa_type=='pca':
                F, L, mean, var = principal_component_analysis(
                    train_arr, apply_data=arr, n_factors=n_fact
                )
        except ValueError as error:
            # The LU factorization is sometimes unstable, leading to infinite numbers
            if n_try < max_tries:
                print("Warning: nan or inf occured during factor analysis")
                continue
            else:
                raise error

    shared = L @ F
    residual = arr - L @ F  # - np.mean(arr, axis=0)

    session_shared = session.copy(data=shared.transpose().reshape(session.values.shape))
    session_residual = session.copy(
        data=residual.transpose().reshape(session.values.shape)
    )

    return session_shared, session_residual


def per_var_explained_of_session(session, num_factors=5, num_fits=1000, fa_type='lfa'):
    hit_trials = session.sel(trial_type="hit").trial_num.values
    miss_trials = session.sel(trial_type="miss").trial_num.values

    res1 = per_var_explained_resampled(session, num_factors, hit_trials, num_fits=num_fits, fa_type=fa_type)
    per_var_explained_hit, cos_similarity_hit, peristim_average_hit = res1
    
    res2 = per_var_explained_resampled(session, num_factors, miss_trials, num_fits=num_fits, fa_type=fa_type)
    per_var_explained_miss, cos_similarity_miss, peristim_average_miss = res2
    
    return per_var_explained_hit, cos_similarity_hit, peristim_average_hit, per_var_explained_miss, cos_similarity_miss, peristim_average_miss


def per_var_explained_resampled(session, num_factors, trial_list, num_fits=1000, n_choose=15, fa_type='lfa'):
    def inner_loop(trials_chosen):
        pre_F_rebinned = session.isel(trial=trials_chosen).values
        pre_F_concatenated = np.reshape(pre_F_rebinned, (pre_F_rebinned.shape[0], pre_F_rebinned.shape[1]*pre_F_rebinned.shape[2]))
        
        if fa_type=='lfa':
            res = factor_analysis(pre_F_concatenated.T, pre_F_concatenated.T, num_factors, 
                                  return_per_var_explained=True)
        elif fa_type=='pca':
            res = principal_component_analysis(pre_F_concatenated.T, pre_F_concatenated.T, num_factors, 
                                  return_per_var_explained=True)
        
        components = res[0]
        chosen_trials_transformed_arr = res[1]
        per_var_explained = res[4]
        
        peristim_average = np.mean(chosen_trials_transformed_arr.reshape((len(trials_chosen), session.values.shape[2], num_factors)), axis=0)
        
        subset_trials_transformed_arr = np.reshape(np.reshape(all_trials_transformed_arr,
                                        (session.values.shape[1], session.values.shape[2], num_factors))[trials_chosen,:,:], 
                                        (len(trials_chosen)*session.values.shape[2], num_factors)
                                          )
        dot_product = np.mean(np.multiply(chosen_trials_transformed_arr, subset_trials_transformed_arr), axis=0)
        cos_similarity = dot_product / (
                            np.linalg.norm(chosen_trials_transformed_arr) * np.linalg.norm(subset_trials_transformed_arr) )
        
        return per_var_explained, cos_similarity, peristim_average
    
    # calculate all hit&miss trials LFA/PCA to compare to
    apply_arr = session.values.reshape((session.values.shape[0], -1)).transpose()
    session_train = session.where(
        (session.trial_type == "miss") | (session.trial_type == "hit"), drop=True
    )
    train_arr = session_train.values.reshape((session.values.shape[0], -1)).transpose()

    apply_arr = np.nan_to_num(apply_arr)
    train_arr = np.nan_to_num(train_arr)
    
    if fa_type=='lfa':
        all_trials_res = factor_analysis(train_arr, apply_arr, num_factors, 
                                                      return_per_var_explained=True)
    elif fa_type=='pca':
        all_trials_res = principal_component_analysis(train_arr, apply_arr, num_factors, 
                                                      return_per_var_explained=True)
    all_trials_transformed_arr = all_trials_res[1]

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
        
        # start inner loop on multiple threads
        per_var_explained_results = executor.map(inner_loop, trials_chosen)
    
    # get results from inner loops
    per_var_explained, cos_similarity, peristim_average = list(zip(*per_var_explained_results))
    print(cos_similarity[0].shape)
    
    # pad results with nans in case that there are less combination than num_fits 
    per_var_explained_array = np.zeros((num_fits, num_factors))
    per_var_explained_array[:] = np.nan
    per_var_explained_array[:len(per_var_explained),:] = per_var_explained
    
    cos_similarity_array = np.zeros((num_fits, num_factors))
    cos_similarity_array[:] = np.nan
    cos_similarity_array[:len(cos_similarity),:] = cos_similarity
    
    peristim_average_array = np.zeros((num_factors, session.values.shape[2]))
    peristim_average_array = np.nanmean(peristim_average, axis=0)
    
    return (
        per_var_explained_array,
        cos_similarity_array,
        peristim_average_array,
    )
        
        
        
        
