'-------------------------------- Load in regular libraries ------------------------------------'
import numpy as np
import matplotlib.pyplot as plt

'-------------------------------- Load in custom libraries ------------------------------------'
from src.Direct_Solvers import DirectSolver
from src.Matrix_Properties import MatrixProperties

'-------------------------------- Initialize custom libraries ------------------------------------'
MAP = MatrixProperties()
DS  = DirectSolver()

'-------------------------------- Load custom functions ------------------------------------'
def _data_generation(nrows : float = 300, ncols : float = 20, noise_addition : bool = True) -> np.ndarray:
    
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

def _data_splitting(X : np.ndarray, y : np.ndarray, mode : str, nrows_train : float) -> np.ndarray:
    
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
                    
                # Generate train set:
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


" ---------------------------------------- Generate & Test Data ---------------------------------------------- "
# Generate data:
X, w, y = _data_generation(nrows = 10, ncols = 5, noise_addition = False)

# Split data:
X_train, y_train, X_test, y_test = _data_splitting(X = X, y = y, mode = 'ratio', nrows_train = 0.8)

# Compute condition number:
kappa = MAP._condition_number(np.matmul(X_train.transpose(), X_train))

# Compute rank:
rank = MAP._rank(np.matmul(X_train.transpose(), X_train))


" ---------------------------------------- Solve ---------------------------------------------- "
mode = 'LU'

# Solve with LU Factorization:
L, U, q, w = DS._solver(np.matmul(X_train.transpose(), X_train), np.matmul(X_train.transpose(), y_train), mode = mode)

# Compute cost function:
J_train = np.linalg.norm(np.matmul(X_train, w) - y_train)
J_test  = np.linalg.norm(np.matmul(X_test, w) - y_test)

print(f'J_train(w) = {J_train}')
print(f'J_test(w) = {J_test}')