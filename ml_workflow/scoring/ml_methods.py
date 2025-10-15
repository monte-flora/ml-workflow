import numpy as np
import xarray as xr
import itertools
import collections 
from collections import OrderedDict

from functools import partial
from sklearn.metrics._base import _average_binary_score
from sklearn.utils.multiclass import type_of_target
from scipy.ndimage import gaussian_filter

from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score, precision_recall_curve

#from mlxtend.evaluate import permutation_test

def get_bootstrap_score(y,
                        preds,
                        n_bootstrap,
                        known_skew,
                        bl_predictions=None,
                        ml_reduced_predictions=None,
                        forecast_time_indices=None, metric_mapper=None):
    """
    Bootstrap compute various performance metrics. Bootstrapping design is built to limit the 
    impact of autocorrelated data from artifically shrinking the uncertainty. 
    """
    base_random_state = np.random.RandomState(22)
    random_num_set = base_random_state.choice(10000, size=n_bootstrap, replace=False)

    even_or_odd = base_random_state.choice([2, 3], size=n_bootstrap)

    if metric_mapper is None:
        metric_mapper = OrderedDict({'naupdc':norm_aupdc,
                             'auc': roc_auc_score,
                             'bss':brier_skill_score,
                             'reliability':bss_reliability,
                             'ncsi': norm_csi, })

    ml_all_results = {key : (['n_boot'],
                         np.zeros((n_bootstrap))) for key in metric_mapper.keys()}
    if ml_reduced_predictions is not None:
        ml_reduced_results = {key : (['n_boot'],
                         np.zeros((n_bootstrap))) for key in metric_mapper.keys()}
    bl_results = {key : (['n_boot'],
                         np.zeros((n_bootstrap))) for key in metric_mapper.keys()}


    for n in range(n_bootstrap):
        # For each bootstrap resample, only sample from some subset of time steps 
        # to reduce autocorrelation effects. Not perfect, but will improve variance assessment. 
        new_random_state = np.random.RandomState(random_num_set[n])
        these_idxs = new_random_state.choice(len(y), size= len(y))
        if forecast_time_indices is not None:
            # val with either be a 2, 3, or 4 and then only resample from those samples. 
            val = even_or_odd[n]
            # Find those forecast time indices that are even or odd 
            where_is_fti = np.where(forecast_time_indices%val==0)[0]
            # Find the "where_is_item" has the same indices as "where_is_fti"
            idxs_subset = list(set(these_idxs).intersection(where_is_fti))
            # Resample idxs_subset for each iteration 
            these_idxs = new_random_state.choice(idxs_subset, size=len(idxs_subset),)

        if len(these_idxs) > 0 and np.mean(y[these_idxs])>0.0:
            for metric in metric_mapper.keys():
                if metric =='naupdc' or metric == 'ncsi':
                    ml_all_results[metric][1][n] = metric_mapper[metric](y[these_idxs],
                                                                preds[these_idxs], known_skew=known_skew)
                    if ml_reduced_predictions is not None:
                        ml_reduced_results[metric][1][n] = metric_mapper[metric](y[these_idxs],
                                                                ml_reduced_predictions[these_idxs],
                                                                             known_skew=known_skew)
                    if bl_predictions is not None:
                        bl_results[metric][1][n] = metric_mapper[metric](y[these_idxs],
                                                                bl_predictions[these_idxs], known_skew=known_skew)
                else:
                    ml_all_results[metric][1][n] = metric_mapper[metric](y[these_idxs], preds[these_idxs])

                    if ml_reduced_predictions is not None:
                        ml_reduced_results[metric][1][n] = metric_mapper[metric](y[these_idxs],
                                                                ml_reduced_predictions[these_idxs], )
                    if bl_predictions is not None:
                        bl_results[metric][1][n] = metric_mapper[metric](y[these_idxs], bl_predictions[these_idxs])

    ml_all_ds = xr.Dataset(ml_all_results)
    if ml_reduced_predictions is not None:
        ml_reduced_ds = xr.Dataset(ml_reduced_results)

    if (ml_reduced_predictions is not None) and (bl_predictions is not None):
        return ml_all_ds, ml_reduced_ds, bl_ds

    elif bl_predictions is not None:
        bl_ds = xr.Dataset(bl_results)
        return ml_all_ds, bl_ds

    else:
        return ml_all_ds


'''
def get_bootstrap_score(ml_all_predictions, 
                        bl_predictions, 
                        n_bootstrap,
                        y, 
                        known_skew, ml_reduced_predictions=None, 
                        forecast_time_indices=None, metric_mapper=None):
    """
    Bootstrap compute various performance metrics. Bootstrapping design is built to limit the 
    impact of autocorrelated data from artifically shrinking the uncertainty. 
    """
    base_random_state = np.random.RandomState(22)
    random_num_set = base_random_state.choice(10000, size=n_bootstrap, replace=False)
    
    even_or_odd = base_random_state.choice([2, 3], size=n_bootstrap)

    if metric_mapper is None:
        metric_mapper = OrderedDict({'naupdc':norm_aupdc, 
                             'auc': roc_auc_score, 
                             'bss':brier_skill_score, 
                             'reliability':bss_reliability, 
                             'ncsi': norm_csi, })
    
    ml_all_results = {key : (['n_boot'],
                         np.zeros((n_bootstrap))) for key in metric_mapper.keys()}
    if ml_reduced_predictions is not None:
        ml_reduced_results = {key : (['n_boot'],
                         np.zeros((n_bootstrap))) for key in metric_mapper.keys()}
    bl_results = {key : (['n_boot'], 
                         np.zeros((n_bootstrap))) for key in metric_mapper.keys()}
    
    
    for n in range(n_bootstrap):
        # For each bootstrap resample, only sample from some subset of time steps 
        # to reduce autocorrelation effects. Not perfect, but will improve variance assessment. 
        new_random_state = np.random.RandomState(random_num_set[n])
        these_idxs = new_random_state.choice(len(y), size= len(y))
        if forecast_time_indices is not None:
            # val with either be a 2, 3, or 4 and then only resample from those samples. 
            val = even_or_odd[n]
            # Find those forecast time indices that are even or odd 
            where_is_fti = np.where(forecast_time_indices%val==0)[0]
            # Find the "where_is_item" has the same indices as "where_is_fti"
            idxs_subset = list(set(these_idxs).intersection(where_is_fti))
            # Resample idxs_subset for each iteration 
            these_idxs = new_random_state.choice(idxs_subset, size=len(idxs_subset),)

        if len(these_idxs) > 0 and np.mean(y[these_idxs])>0.0:
            for metric in metric_mapper.keys():
                if metric =='naupdc' or metric == 'ncsi':
                    ml_all_results[metric][1][n] = metric_mapper[metric](y[these_idxs], 
                                                                ml_all_predictions[these_idxs], known_skew=known_skew)
                    if ml_reduced_predictions is not None:
                        ml_reduced_results[metric][1][n] = metric_mapper[metric](y[these_idxs], 
                                                                ml_reduced_predictions[these_idxs], 
                                                                             known_skew=known_skew)
                    
                    bl_results[metric][1][n] = metric_mapper[metric](y[these_idxs], 
                                                                bl_predictions[these_idxs], known_skew=known_skew)
                else:
                    ml_all_results[metric][1][n] = metric_mapper[metric](y[these_idxs], ml_all_predictions[these_idxs])
                    
                    if ml_reduced_predictions is not None:
                        ml_reduced_results[metric][1][n] = metric_mapper[metric](y[these_idxs], 
                                                                ml_reduced_predictions[these_idxs], )
                    bl_results[metric][1][n] = metric_mapper[metric](y[these_idxs], bl_predictions[these_idxs])
    
    ml_all_ds = xr.Dataset(ml_all_results)
    if ml_reduced_predictions is not None:
        ml_reduced_ds = xr.Dataset(ml_reduced_results)
    bl_ds = xr.Dataset(bl_results)
    
    if ml_reduced_predictions is not None:
        return ml_all_ds, ml_reduced_ds, bl_ds
    else:
        return ml_all_ds,  bl_ds
'''

def stat_testing(new_score, baseline_score):
    """
    Compute a p-value between two sets using permutation testing 
    to determined statistical significance. In this case,
    assess whether the ML performance is greater than the baseline.
    """
    p_value = permutation_test(new_score,
                              baseline_score,
                             'x_mean != y_mean',
                              method='approximate',
                               num_rounds=1000,
                               seed=0)
    return p_value


def bootstrap_generator(n_bootstrap, seed=42):
    """
    Create a repeatable bootstrap generator.
    """
    base_random_state = np.random.RandomState(seed)
    random_num_set = base_random_state.choice(10000, size=n_bootstrap, replace=False)
    return random_num_set

def scorer(model,X,y,evaluation_fn):
    """
    Get the score from model, X,y
    """
    if hasattr(model, 'predict_proba'):
        predictions = model.predict_proba(X)[:,1]
    else:
        predictions = model.predict(X)
        
    return evaluation_fn(y,predictions)
    
def significance_of_importance(model, 
                               X,
                               y,
                               best_pair,
                               evaluation_fn,
                               n_bootstrap=100):
    """
    Evaluate importance of remaining pairs for a given multipass iteration.
    
    Parameters: 
    -----------------
        model , 
        X, 
        y, 
        best_pair, 
        evaluation_fn, 
        n_bootstrap, 
    """
    #other_pairs = [i for i in list(itertools.combinations(X.columns, r=len(best_pair))) 
    #                       if collections.Counter(i)!= collections.Counter(best_pair)]
    
    other_pairs = list(itertools.combinations(X.columns, r=len(best_pair)))
    
    original_score = scorer(model,X,y,evaluation_fn)
    permuted_score = []
    
    random_num_set = bootstrap_generator(n_bootstrap, seed=1)
    permuted_score = {tuple(p):np.zeros(n_bootstrap) for p in other_pairs}
    
    for i, pair in enumerate(other_pairs):
        for n in range(n_bootstrap):
            X_permuted = X.copy()
            new_random_state = np.random.RandomState(random_num_set[n])
            for name in pair:
                X_permuted[name] = np.random.permutation(X[name].values)
            permuted_score[pair][n] = scorer(model, X_permuted, y, evaluation_fn) - original_score
            
    sorted_scores=collections.OrderedDict({k: v for k, v in 
                   sorted(permuted_score.items(), key=lambda item: np.mean(item[1]), reverse=True)})   
    
    return sorted_scores


def modified_precision(precision, known_skew, new_skew): 
    """
    Modify the success ratio according to equation (3) from 
    Lampert and Gancarski (2014). 
    """
    precision[precision<1e-5] = 1e-5
    term1 = new_skew / (1.0-new_skew)
    term2 = ((1/precision) - 1.0)
    
    denom = known_skew + ((1-known_skew)*term1*term2)
    
    return known_skew / denom 
    
def calc_sr_min(skew):
    pod = np.linspace(0,1,100)
    sr_min = (skew*pod) / (1-skew+(skew*pod))
    return sr_min 

def _binary_uninterpolated_average_precision(
            y_true, y_score, known_skew, new_skew, pos_label=1, sample_weight=None):
        precision, recall, _ = precision_recall_curve(
            y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
        # Return the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        if known_skew is not None:
            precision = modified_precision(precision, known_skew, new_skew)
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])

def min_aupdc(y_true, pos_label, average, sample_weight=None, known_skew=None, new_skew=None):
    """
    Compute the minimum possible area under the performance 
    diagram curve. Essentially, a vote of NO for all predictions. 
    """
    min_score = np.zeros((len(y_true)))
    average_precision = partial(_binary_uninterpolated_average_precision,
                                known_skew=known_skew,
                                new_skew=new_skew,
                                pos_label=pos_label)
    ap_min = _average_binary_score(average_precision, y_true, min_score,
                                 average, sample_weight=sample_weight)

    return ap_min


def calc_csi(precision, recall):
    """
    Compute the critical success index
    """
    precision[precision<1e-5] = 1e-3
    recall[recall<1e-5] = 1e-3
    
    csi = 1.0 / ((1/precision) + (1/recall) - 1.0)
    
    return csi 

def norm_csi(y_true, y_score, known_skew, pos_label=1, sample_weight=None):
    """
    Compute the normalized modified critical success index. 
    """
    new_skew = np.mean(y_true)
    precision, recall, _ = precision_recall_curve(
            y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
    if known_skew is not None:
        precision = modified_precision(precision, known_skew, new_skew)
    
    csi = calc_csi(precision, recall)
    max_csi = np.max(csi)
    ncsi = (max_csi - known_skew) / (1.0 - known_skew)
    
    return ncsi 
    


def norm_aupdc(y_true, y_score, known_skew, *, average="macro", pos_label=1,
                            sample_weight=None, min_method='random'):
    """
    Compute the normalized modified average precision. Normalization removes 
    the no-skill region either based on skew or random classifier performance. 
    Modification alters success ratio to be consistent with a known skew. 
  
    Parameters:
    -------------------
        y_true, array of (n_samples,)
            Binary, truth labels (0,1)
        y_score, array of (n_samples,)
            Model predictions (either determinstic or probabilistic)
        known_skew, float between 0 and 1 
            Known or reference skew (# of 1 / n_samples) for 
            computing the modified success ratio.
        min_method, 'skew' or 'random'
            If 'skew', then the normalization is based on the minimum AUPDC 
            formula presented in Boyd et al. (2012).
            
            If 'random', then the normalization is based on the 
            minimum AUPDC for a random classifier, which is equal 
            to the known skew. 
    
    
    Boyd, 2012: Unachievable Region in Precision-Recall Space and Its Effect on Empirical Evaluation, ArXiv
    """
    new_skew = np.mean(y_true)

    y_type = type_of_target(y_true)
    if y_type == "multilabel-indicator" and pos_label != 1:
        raise ValueError("Parameter pos_label is fixed to 1 for "
                         "multilabel-indicator y_true. Do not set "
                         "pos_label or set pos_label to 1.")
    elif y_type == "binary":
        # Convert to Python primitive type to avoid NumPy type / Python str
        # comparison. See https://github.com/numpy/numpy/issues/6784
        present_labels = np.unique(y_true).tolist()
        if len(present_labels) == 2 and pos_label not in present_labels:
            raise ValueError(
                f"pos_label={pos_label} is not a valid label. It should be "
                f"one of {present_labels}"
            )
    average_precision = partial(_binary_uninterpolated_average_precision,
                                known_skew=known_skew,
                                new_skew=new_skew,
                                pos_label=pos_label)
    
    ap = _average_binary_score(average_precision, y_true, y_score,
                                 average, sample_weight=sample_weight)
    
    if min_method == 'random':
        ap_min = known_skew 
    elif min_method == 'skew':
        ap_min = min_aupdc(y_true, 
                       pos_label, 
                       average,
                       sample_weight=sample_weight,
                       known_skew=known_skew, 
                       new_skew=new_skew)
    
    naupdc = (ap - ap_min) / (1.0 - ap_min)

    return naupdc

def brier_skill_score(y_values, forecast_probabilities, **kwargs):
    """Computes the brier skill score"""
    climo = np.mean((y_values - np.mean(y_values)) ** 2)
    return 1.0 - brier_score_loss(y_values, forecast_probabilities) / climo


def bss_reliability(targets, predictions):
    """
    Reliability component of BSS. Weighted MSE of the mean forecast probabilities
    and the conditional event frequencies. 
    """
    mean_fcst_probs, event_frequency, indices = reliability_curve(targets, predictions, n_bins=10)
    # Add a zero for the origin (0,0) added to the mean_fcst_probs and event_frequency
    counts = [1e-5]
    for i in indices:
        if i is np.nan:
            counts.append(1e-5)
        else:
            counts.append(len(i[0]))

    mean_fcst_probs[np.isnan(mean_fcst_probs)] = 1e-5
    event_frequency[np.isnan(event_frequency)] = 1e-5

    diff = (mean_fcst_probs-event_frequency)**2
    return np.average(diff, weights=counts)

def reliability_curve(targets, predictions, n_bins=10):
        """
        Generate a reliability (calibration) curve. 
        Bins can be empty for both the mean forecast probabilities 
        and event frequencies and will be replaced with nan values. 
        Unlike the scikit-learn method, this will make sure the output
        shape is consistent with the requested bin count. The output shape
        is (n_bins+1,) as I artifically insert the origin (0,0) so the plot
        looks correct. 
        """
        bin_edges = np.linspace(0,1, n_bins+1)
        bin_indices = np.clip(
                np.digitize(predictions, bin_edges, right=True) - 1, 0, None
                )

        indices = [np.where(bin_indices==i+1)
               if len(np.where(bin_indices==i+1)[0]) > 0 else np.nan for i in range(n_bins) ]

        mean_fcst_probs = [np.nan if i is np.nan else np.mean(predictions[i]) for i in indices]
        event_frequency = [np.nan if i is np.nan else np.sum(targets[i]) / len(i[0]) for i in indices]

        # Adding the origin to the data
        mean_fcst_probs.insert(0,0)
        event_frequency.insert(0,0)
        
        return np.array(mean_fcst_probs), np.array(event_frequency), indices

