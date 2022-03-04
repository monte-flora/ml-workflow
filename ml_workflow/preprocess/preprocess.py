import numpy as np 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer, MinMaxScaler, OneHotEncoder 
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import  SimpleImputer, IterativeImputer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer

import marshal
from types import FunctionType
from sklearn.base import BaseEstimator, TransformerMixin

class PreProcessPipeline:
    
    def __init__(self, imputer='simple', scaler='minmax',
            pca=None, resample='under', numeric_features=None, categorical_features=None ): 

        self.imputer_arg = imputer
        self.scaler_arg = scaler
        self.pca_arg = pca
        self.resample_arg= resample
        self._categorical_features=categorical_features
        self._numeric_features=numeric_features

    def get_steps(self):
        # Pre-processing order : Imputer, Normalize, PCA, Resample, 
        method_args = [self.imputer_arg, self.scaler_arg, self.pca_arg,  
                       self.resample_arg, ]        
        method_order = ['imputer', 'scaler', 'pca_transform', 'resample', ] 

        numeric_transformers = [ ]
        for arg, method in zip(method_args, method_order):
            if arg is not None:
                func = getattr(self, method)
                numeric_transformers.append(func(arg))

        if self._categorical_features is not None:
            categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        
            transformer = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformers, self._numeric_features),
                    ("cat", categorical_transformer, self._categorical_features),
                    ]
                )
        
            return [("preprocessor", transformer)]

        return numeric_transformers

    def resample(self, method=None):
        """
        Resamples a dataset to 1:1 using the imblearn python package
        by either random over or under sampling. 
        """
        if method == 'under':
            resampler = RandomUnderSampler(random_state=42)
        elif method == 'over':
            resampler = RandomOverSampler(random_state=42)

        return ('resampler', resampler)

    def imputer(self,method=None):
        """
        Imputation transformer for missing values.
        """
        if method == 'simple':
            imputer = SimpleImputer(
                missing_values=np.nan, strategy="median"
            )
        else:
            imputer = IterativeImputer(random_state=0)

        return ('imputer', imputer)

    def pca_transform(self, method=None):
        """
        Peforms Principal Component Analysis on the examples
        """
        # Make an instance of the Model
        pca = PCA( )

        return ('pca', pca) 

    def scaler(self, method=None):
        """
        Scaling a dataset.
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        
        return ('scaler', scaler) 

    
class CorrelationFilter:
    """
    CorrelationFilter checks for correlated predictors in a dataset and removes them.
    """

    def get_all_predictor(self, this_predictor, that_predictor, all_predictor_pairs):
        """
        Pair a predictor with all other predictors in a dataset excluding "that_predictor"

        Args:
        --------------
            this_predictor : str
                the predictor we want paired with other predictors in the dataset
            that_predictor : str
                the predictor we don't want to pair with.
            all_predictors_pairs : pandas object
                dataframe where the indices are 2-tuples of predictor pairs and the values
                are the correlation values.

        """
        all_possible_pairs = list(all_predictor_pairs.index.values)

        predictors_paired_with_this_predictor = [ p for p in all_possible_pairs
                                               if (that_predictor not in p) and (this_predictor in p)
                                               ]

        return predictors_paired_with_this_predictor

    def avg_corr_with_other_predictors(self, all_predictor_pairs, predictors_paired_with_this_predictor):
        """
        """
        cc_vals = [all_predictor_pairs[pair] for pair in predictors_paired_with_this_predictor]

        return np.nanmean(cc_vals)

    def correlation_filtering(self, df, cc_val):
        """
        filter a dataframe by correlation
        2. Determine the pair of predictors with the highest correlation 
        greater than a given threshold

        3. Determine the average correlation between these predictors and
            the remaining features

        4. Remove the predictors with the highest average correlation. 
        """
        corr_matrix = df.corr().abs()
        predictor_names = list(df.columns)

        correlated_pairs = {}
        columns_to_drop = []
        # the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
        # The indices of all_predictors are tuples of (predictor1,predictor2) with the values being 
        # the correlation coefficient between them 
        all_predictor_pairs = (
            corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            .stack()
            .sort_values(ascending=False)
        )
        # We are only concerned with predictor pairs with a correlation above a given threshold
        correlated_predictor_pairs = all_predictor_pairs[all_predictor_pairs > cc_val].index.values

        # Cycle the pairs of correlated predictors
        for pair in correlated_predictor_pairs:
            predictor_A, predictor_B = pair
            # If we already removed a predictor then we don't consider in future pairings
            if predictor_A in columns_to_drop or predictor_B in columns_to_drop:
                continue

            # This is all possible pairs (2-tuples) in dataset not including the predictor itself,
            # the predictor it is being compared against, or any predictors that have already
            # been dropped. Similar for the following set of predictor B. 
            predictors_paired_with_A = self.get_all_predictor(this_predictor=predictor_A,
                                                         that_predictor=predictor_B,
                                                         all_predictor_pairs=all_predictor_pairs
                                                        )

            predictors_paired_with_B = self.get_all_predictor(this_predictor=predictor_B,
                                                         that_predictor=predictor_A,
                                                         all_predictor_pairs=all_predictor_pairs
                                                        )

            # Compute the average correlation of predictor A with every predictor 
            # in the dataset (excluding predictor A and predictor B)
            avg_corr_with_A = self.avg_corr_with_other_predictors(all_predictor_pairs,
                                                             predictors_paired_with_A)

            # Compute the average correlation of predictor B with every predictor 
            # in the dataset. 
            avg_corr_with_B = self.avg_corr_with_other_predictors(all_predictor_pairs,
                                                             predictors_paired_with_B)

            if avg_corr_with_A > avg_corr_with_B:
                columns_to_drop.append(predictor_A)
                correlated_pairs[predictor_A] = (
                    predictor_B,
                    all_predictor_pairs[pair],
                )
            else:
                columns_to_drop.append(predictor_B)
                correlated_pairs[predictor_B] = (
                    predictor_A,
                    all_predictor_pairs[pair],
                )

        return columns_to_drop, correlated_pairs

    def filter_df_by_correlation(self, inp_data, target_var, cc_val=0.8):
        """
        Returns an array or dataframe (based on type(inp_data) adjusted to drop \
            columns with high correlation to one another. Takes second arg corr_val
            that defines the cutoff

        ----------
        inp_data : np.array, pd.DataFrame
            Values to consider
        corr_val : float
            Value [0, 1] on which to base the correlation cutoff
        """
        # Creates Correlation Matrix
        corr_matrix = inp_data.corr()

        # Iterates through Correlation Matrix Table to find correlated columns
        columns_to_drop = []
        n_cols = len(corr_matrix.columns)

        correlated_features = []
        print("Calculating correlations between features...")
        for i in range(n_cols):
            for k in range(i + 1, n_cols):
                val = corr_matrix.iloc[k, i]
                col = corr_matrix.columns[i]
                row = corr_matrix.index[k]
                col_to_target = corr_matrix.loc[col, target_var]
                row_to_target = corr_matrix.loc[row, target_var]
                if abs(val) >= cc_val:
                    # Prints the correlated feature set and the corr val
                    if (
                        abs(col_to_target) > abs(row_to_target)
                        and (row not in columns_to_drop)
                        and (row not in self.EXTRA_VARIABLES + self.TARGETS_VARIABLES )
                    ):
                        # print( f'{col} ({col_to_target:.3f}) | {row} ({row_to_target:.3f}) | {val:.2f}....Dropped {row}')
                        columns_to_drop.append(row)
                        correlated_features.append((row, col))
                    if (
                        abs(row_to_target) > abs(col_to_target)
                        and (col not in columns_to_drop)
                        and (col not in self.EXTRA_VARIABLES + self.TARGETS_VARIABLES)
                    ):
                        # print( f'{col} ({col_to_target:.3f}) | {row} ({row_to_target:.3f}) | {val:.2f}....Dropped {col}')
                        columns_to_drop.append(col)
                        correlated_features.append((row, col))

        # Drops the correlated columns
        print("Dropping {} highly correlated features...".format(len(columns_to_drop)))
        print(len(columns_to_drop) == len(correlated_features))
        df = self.drop_columns(inp_data=inp_data, to_drop=columns_to_drop)

        return df, columns_to_drop, correlated_features

    def drop_columns(self, inp_data, to_drop):
        """
        """
        # Drops the correlated columns
        columns_to_drop = list(set(to_drop))
        inp_data = inp_data.drop(columns=columns_to_drop)
        # Return same type as inp
        return inp_data





