import numpy as np
import random

class Data_Generation():
    
    '-------------------------------- Initialize Class ------------------------------------'
    def __init__(self):
        pass
    
    
    '-------------------------------- Generate Linear Regression data ------------------------------------'
    def _data_generation_linear_regression(self, nrows : float = 300, ncols : float = 20, noise_addition : bool = True) -> np.ndarray:

        """
        Args:
             nrows          : number of rows for data matrix X.
             ncols          : number of cols for data matrix X.
             noise_addition : add noise to output vector y (True/False).

        """

        # Create matrix with uniformly random variables:
        X = np.random.uniform(low = 0, high = 1, size = (nrows, ncols))    

        # Weight vector w and noise:  
        w = np.random.uniform(low = 0, high = 1, size = (ncols, 1))      
        y = np.matmul(X, w)

        # noise addition of selected
        if noise_addition:
            noise = np.random.normal(loc = 0, scale = 1, size = (nrows, 1))   
            y    += noise

        return X, w, y

    
    '-------------------------------- Generate Classifier data ------------------------------------'
    def _data_generation_classifier(self, nrows : float = 100, ncols : float = 10) -> np.ndarray:

        """
        Args:
             nrows          : number of rows for data matrix X.
             ncols          : number of cols for data matrix X.
             noise_addition : add noise to output vector y (True/False).

        """

        # Create matrix with uniformly random variables:
        X = np.random.normal(loc = 0, scale = 1, size = (nrows, ncols))   

        # Weight vector w and noise:  
        w = np.random.normal(loc = 0, scale = 1, size = (ncols, 1))

        # Create output vector y:
        y = np.zeros(nrows)

        for i in range(nrows):
            y[i] = np.sign(np.matmul(X[i, :].T, w))


        return X, w, y
    

    '-------------------------------- Split Data ------------------------------------'
    def _data_splitting(self, X : np.ndarray, y : np.ndarray, mode : str, nrows_train : float, seed  = 0, random = False) -> np.ndarray:

        """
        Args:
             X           : data matrix X.
             y           : output vector y.
             mode        : mode for splitting -> ratio takes a percentage of the available rows and index splits the data untill the given index value.
             nrows_train : in case of mode ratio give a value between 0 and 1 and in case of mode index give an index value between 0 and max nr. of rows of matrix X (or vector y)

        """

        # Initialize train and test sets:
        X_train   = np.zeros((1, X.shape[1]))
        y_train   = np.zeros(1)

        X_test    = np.zeros((1, X.shape[1]))
        y_test    = np.zeros(1)

        # Select total rows of matrix X
        nrows_tot = X.shape[0]

        # Split data set:
        match mode:
            case 'ratio':

                if (nrows_train > 0) & (nrows_train <= 1):
                    
                    if nrows_train*nrows_tot % 1 > 0:
                        print(f'For the given value of nrows_train a remainder has been found of {nrows_train*nrows_tot % 1}. Therefore, the ratio has been adjusted to: {int(nrows_tot*nrows_train)/nrows_tot}.')

                    if random:

                        rng = np.random.default_rng(seed)
                        train_size = int(nrows_tot * nrows_train)

                        # random train indices
                        train_idx = rng.choice(nrows_tot, size=train_size, replace=False)

                        # test indices = complement of train indices
                        test_idx = np.setdiff1d(np.arange(nrows_tot), train_idx)

                        # split
                        X_train, y_train = X[train_idx], y[train_idx]
                        X_test,  y_test  = X[test_idx],  y[test_idx]
                    else:
                        X_train = X[0 : int(nrows_tot*nrows_train), :]
                        y_train = y[0 : int(nrows_tot*nrows_train)]

                        # Generate test set:
                        X_test  = X[int(nrows_tot*nrows_train):, :]
                        y_test  = y[int(nrows_tot*nrows_train):]    


                else:
                    print('The parameter "nrows_train" has to be between 0 and 1 for mode: ratio')


            case 'index':

                if (nrows_train > 0) & (nrows_train <= nrows_tot):

                    # Generate train set:
                    X_train = X[0 : nrows_train, :]
                    y_train = y[0 : nrows_train]

                    # Generate test set:
                    X_test  = X[nrows_train:, :]
                    y_test  = y[nrows_train:]    


                else:
                    print('The parameter "nrows_train" has to be between 0 and max nr. of rows of X for mode: index')

            case _:

                print('Please select either the "ratio" or "columns" mode.')


        return X_train, y_train, X_test, y_test

    '-------------------------------- Addition of multicolinearities ------------------------------------'
    def _multicolinearities_addition(self, X : np.ndarray, N_multicolinear_columns : int = 200) -> np.ndarray:

        """
        Args:
             X                       : data matrix X.
             N_multicolinear_columns : Number of multicolinear columns to be generated.

        """

        # Initialize multicolinear column matrix:
        multicolinear_columns   = np.zeros((X.shape[0], N_multicolinear_columns))

        for n in range(N_multicolinear_columns):

            # initialize multicolinear column:
            multicolinear_column = np.zeros((1, X.shape[0]))

            # Create a multicolinear column by making a linear combination of existing columns of X:
            for _ in range(random.randint(2, 10)):

                # Select random column:
                i = random.randint(0, X.shape[1] - 1)

                # Select random magnitude:
                k = random.uniform(-1, 1) 

                # Generate multicolinear column:
                multicolinear_column += k*np.array(X[:, i])

            # Add Gaussian Noise:
            multicolinear_column += np.random.normal(loc = 0, scale = 1, size = (1, X.shape[0]))

            # Fill multicolinear column matrix:
            multicolinear_columns[:, n] = multicolinear_column 

        return multicolinear_columns
    
    '-------------------------------- Addition of superifical columns ------------------------------------'
    def _superficial_addition(self, nrows : float = 300, ncols : float = 20) -> np.ndarray:

        """
        Args:
             nrows          : number of rows for data matrix X.
             ncols          : number of cols for data matrix X.

        """

        # Create matrix with uniformly random variables:
        superficial_columns = np.random.normal(loc = 0, scale = 1, size = (nrows, ncols))   

        return superficial_columns
    
    '-------------------------------- Dataset to make it hard for OLS ------------------------------------'
    def make_ols_tricky_separable(
        self,
        n_pos=120,                 # majority (+1)
        n_neg=40,                  # minority (−1)
        ncols=20,                  # feature dimension
        margin=0.1,                # small margin  means close along separator
        scale_pos=1.0,             # variance scale for +1 class
        scale_neg=2.0,             # variance scale for −1 class 
        outlier_frac=0.05,         # fraction of each class as far outliers (same side)
        outlier_multiplier=12.0,   # how far to push outliers along their side
        seed=None
    ):
        """
        Notes:
            - Data are strictly linearly separable (we enforce a tiny positive margin).
            - OLS performs poorly due to: small margin, imbalance, variance mismatch, and far outliers.
            - Logistic / hinge focus on margin and are less swayed by far, easy points.
        """
        
        rng = np.random.default_rng(seed)

        # Choose a random separating direction w (unit vector), bias b = 0
        w = rng.normal(size=ncols)
        w /= (np.linalg.norm(w) + 1e-12)

        # Place class means near the boundary along w to make the margin small
        d = margin * min(scale_pos, scale_neg)
        mu_pos = (+0.5 * d) * w
        mu_neg = (-0.5 * d) * w

        # Sample from Gaussians with different variances (imbalance via n_pos vs n_neg)
        X_pos = mu_pos + rng.normal(scale=scale_pos, size=(n_pos, ncols))
        X_neg = mu_neg + rng.normal(scale=scale_neg, size=(n_neg, ncols))

        # Add outliers on the *correct* side 
        def add_outliers(X, y_sign, frac, mult):
            if frac <= 0: return X
            k = max(1, int(len(X) * frac))
            idx = rng.choice(len(X), size=k, replace=False)
            # push along the class side: +w for +1, -w for -1
            X[idx] = X[idx] + (mult * y_sign) * w
            return X

        X_pos = add_outliers(X_pos, +1, outlier_frac, outlier_multiplier)
        X_neg = add_outliers(X_neg, -1, outlier_frac, outlier_multiplier)

        #  Stack and label
        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.ones(n_pos, dtype=float), -np.ones(n_neg, dtype=float)])

        # Ensure strict linear separability with a tiny positive margin epsilon
        # If any sample violates y*(w·x) > 0, nudge minimally along y*w.
        proj = (X @ w) * y
        eps = 1e-3
        need = eps - proj
        mask = need > 0
        if np.any(mask):
            # Move only along w, just enough to satisfy y*(w·x) >= eps
            X[mask] += (need[mask].reshape(-1, 1) * (y[mask].reshape(-1, 1) * w))

        return X, y, w