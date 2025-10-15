import numpy as np 
from imblearn.under_sampling import RandomUnderSampler

class DateBasedCV:
    """
    CustomCV handles train, test splits for the ML pipeline
    based on date. Additionally, it maintains the skew of the whole dataset
    in each validation fold like stratified K-folds CV.
    
    Attributes:
    ---------------------
        n_splits : int 
            Number of cross-validation folds 
        
        dates : list , shape = (n_samples,)
           List of dates for each example in the dataset to be split.
           Could be months, day of the week, literal date (YYYYMMDD).
           Folds will be determined such that dates in the training fold
           do not overlap with the validation fold. 
       
       valid_size : int or float between 0 and 1
           If int, then interpreted as the number of dates to include 
           in each validation dataset. E,g, if valid_size=5 and 
           dates is months, then each validation dataset will 
           include 5 months 
           
           If float, then interpreted as a fraction of the unique dates
           provided in dates. 

    """
    def __init__(self, n_splits, dates, valid_size=None, y=None, verbose=False):
        self.n_splits = n_splits 
        self.dates = np.array(dates).astype(float)
        self.y = y
        self.verbose=verbose
        self.unique_dates = np.unique(dates)
        self.n_dates = len(self.unique_dates)
        
        if valid_size is not None:
            if valid_size < 1:
                self.valid_size = int(valid_size * self.n_dates)
            else:
                self.valid_size = valid_size

            if self.valid_size > self.n_dates:
                raise ValueError('Valid size cannot be larger than the number of unique dates!')   

        if self.n_dates % self.n_splits != 0:
            self.fold_interval = int((self.n_dates / self.n_splits) + 1)
        else:
            self.fold_interval = int((self.n_dates / self.n_splits))
        
        if valid_size is None:
            self.valid_size = self.fold_interval 
        
        
        if self.y is not None:
            self.skew = np.mean(self.y)  
   
        self.train_size = self.n_dates - self.valid_size
    
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
        for fold, r in enumerate(range(0, self.n_dates, self.fold_interval)):
       
            train_date_indices = (np.arange(self.train_size) + r) % self.n_dates
            valid_date_indices = (np.arange(self.train_size, self.train_size+self.valid_size) + r) % self.n_dates 

            this_training_folds_dates = self.unique_dates[train_date_indices]
            this_validation_folds_dates = self.unique_dates[valid_date_indices]
            
            train_indices = self.get_indices_based_on_dates(self.dates, date_set=this_training_folds_dates)
            valid_indices = self.get_indices_based_on_dates(self.dates, date_set=this_validation_folds_dates)
            
            #valid_indices = self.resample(X[valid_indices], self.y[valid_indices], valid_indices)

            if self.verbose:
                val = _is_cross_validation_good(train_indices, valid_indices)
                print(train_date_indices, valid_date_indices)
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

    

