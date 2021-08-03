import numpy as np 
from imblearn.under_sampling import RandomUnderSampler

class DateBasedCV:
    """
    CustomCV handles train, test splits for the ML pipeline
    based on date. Additionally, it maintains the skew of the whole dataset
    in each validation fold like stratified K-folds CV. 
    """
    def __init__(self, n_splits, dates, y=None, verbose=False):
        self.n_splits = n_splits 
        self.dates = np.array(dates).astype(float)
        self.y = y
        self.verbose=verbose
        if self.y is not None:
            self.skew = np.mean(self.y)  

    def get_indices_based_on_dates(self, dates, date_set):
        """Return indices based on a set of dates"""
        idx = []
        for d in date_set:
            temp_idx = np.where(dates == float(d))[0]
            idx.extend(temp_idx)
        return idx

    def resample(self, X, y, indices):
        """Resample valid indices"""
        resampler = RandomUnderSampler(random_state=42, replacement=True, sampling_strategy = float(self.skew / (1.-self.skew))) 
        _ = resampler.fit_resample(X, y)
            
        return np.array(indices)[resampler.sample_indices_] 

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iteratioons in the cross-validator"""
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and validation sets."""
        unique_dates = np.unique(self.dates)
        n_dates = len(unique_dates)

        if n_dates % self.n_splits != 0:
            fold_interval = int((n_dates / self.n_splits) + 1)
        else:
            fold_interval = int((n_dates / self.n_splits))

        train_size = fold_interval * (self.n_splits-1)
        valid_size = fold_interval 

        for fold, r in enumerate(range(0, n_dates, fold_interval)):
       
            train_date_indices = (np.arange(train_size) + r) % n_dates
            valid_date_indices = (np.arange(train_size, train_size+valid_size) + r) % n_dates 
            
            this_training_folds_dates = unique_dates[train_date_indices]
            this_validation_folds_dates = unique_dates[valid_date_indices]
            
            train_indices = self.get_indices_based_on_dates(self.dates, date_set=this_training_folds_dates)
            valid_indices = self.get_indices_based_on_dates(self.dates, date_set=this_validation_folds_dates)
      
            #valid_indices = self.resample(X[valid_indices], self.y[valid_indices], valid_indices)

            if self.verbose:
                val = _is_cross_validation_good(train_indices, valid_indices)
                print(f'Is CV is good...{val}') 
                
            yield train_indices, valid_indices

def _is_cross_validation_good(training_indices, validation_indices):
    """
    Checks if training indices or validation indices overlap. 
    For True, there should be no overlap
    """
    a_set = set(training_indices)
    b_set = set(validation_indices)

    if (a_set & b_set):
        return False
    else:
        return True

    

