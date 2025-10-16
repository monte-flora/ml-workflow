from typing import Literal, Optional, List

import numpy as np 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer, MinMaxScaler, OrdinalEncoder 
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import  SimpleImputer, IterativeImputer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as IMBPipeline
from sklearn.pipeline import Pipeline as SKPipeline
from joblib import Memory

import marshal
from types import FunctionType
from sklearn.base import BaseEstimator, TransformerMixin
from tempfile import mkdtemp

from dataclasses import dataclass
from typing import Optional, Literal, List
import numpy as np
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as IMBPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


@dataclass
class PreProcessPipeline:
    """
    PreProcessPipeline wraps sklearn estimators with preprocessing: 
    imputation, scaling, PCA, and class resampling.
    
    Args:
        imputer: Method to handle missing data ('simple', 'iterative', or None)
        imputer_kwargs: Custom kwargs for imputer (uses sensible defaults if None)
        scaler: Feature scaling method ('standard', 'robust', 'minmax', or None)
        pca: Whether to apply PCA dimensionality reduction
        resample: Class resampling method ('under', 'over', or None)
        numeric_features: List of numerical feature names
        categorical_features: List of categorical feature names
    """
    imputer: Optional[Literal['simple', 'iterative']] = 'simple'
    imputer_kwargs: Optional[dict] = None
    scaler: Optional[Literal['minmax', 'standard', 'robust']] = 'standard'
    pca: bool = False
    resample: Optional[Literal['under', 'over']] = None
    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None

    # Configuration for different components
    _IMPUTERS = {
        'simple': (SimpleImputer, {'missing_values': np.nan, 'strategy': 'median'}),
        'iterative': (IterativeImputer, {
            'random_state': 0, 'missing_values': np.nan, 'n_nearest_features': 10,
            'initial_strategy': 'mean', 'skip_complete': True, 'min_value': -0.0001,
            'max_value': 10000, 'tol': 0.01, 'max_iter': 5
        })
    }
    
    _SCALERS = {
        'standard': StandardScaler,
        'robust': RobustScaler,
        'minmax': MinMaxScaler
    }
    
    _RESAMPLERS = {
        'under': RandomUnderSampler,
        'over': RandomOverSampler
    }

    def __post_init__(self):
        """Validate resampling requirement."""
        if self.resample is not None:
            try:
                import imblearn
            except ImportError:
                raise ImportError("imbalanced-learn required for resampling")

    def get_pipeline(self, estimator):
        """Build complete pipeline with estimator."""
        steps = self._build_steps()
        steps.append(('model', estimator))
        
        Pipeline = IMBPipeline if self.resample else SKPipeline
        return Pipeline(steps=steps)
    
    def _build_steps(self):
        """Build preprocessing pipeline steps."""
        # Build numeric transformers
        numeric_transformers = []
        
        if self.imputer:
            numeric_transformers.append(self._make_imputer())
        if self.scaler:
            numeric_transformers.append(self._make_scaler())
        if self.pca:
            numeric_transformers.append(('pca', PCA()))
        
        # Handle categorical features if specified
        if self.categorical_features:
            steps = [self._make_column_transformer(numeric_transformers)]
        else:
            steps = numeric_transformers
        
        # Add resampling if specified
        if self.resample:
            steps.append(self._make_resampler())
        
        return steps

    def _make_imputer(self):
        """Create imputer step."""
        imputer_class, default_kwargs = self._IMPUTERS[self.imputer]
        kwargs = self.imputer_kwargs or default_kwargs
        return ('imputer', imputer_class(**kwargs))

    def _make_scaler(self):
        """Create scaler step."""
        scaler_class = self._SCALERS[self.scaler]
        return ('scaler', scaler_class())

    def _make_resampler(self):
        """Create resampler step."""
        resampler_class = self._RESAMPLERS[self.resample]
        return ('resampler', resampler_class(random_state=42))

    def _make_column_transformer(self, numeric_transformers):
        """Create column transformer for mixed numeric/categorical features."""
        # Categorical pipeline: encode then apply numeric transforms
        categorical_pipeline = [
            ('encoder', OrdinalEncoder(
                handle_unknown="use_encoded_value", 
                unknown_value=np.nan
            ))
        ] + numeric_transformers
        
        transformer = ColumnTransformer(
            transformers=[
                ("num", SKPipeline(numeric_transformers), self.numeric_features),
                ("cat", SKPipeline(categorical_pipeline), self.categorical_features),
            ]
        )
        
        return ("preprocessor", transformer)

    
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

    def filter_dataframe(self, X, cc_val):
        """
        filter a dataframe by correlation
        2. Determine the pair of predictors with the highest correlation 
        greater than a given threshold

        3. Determine the average correlation between these predictors and
            the remaining features

        4. Remove the predictors with the highest average correlation. 
        """
        corr_matrix = X.corr().abs()
        predictor_names = list(X.columns)

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

        X_subset = X.drop(columns_to_drop, axis=1)       
                
        return X_subset, columns_to_drop, correlated_pairs

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





