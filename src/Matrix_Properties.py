import numpy as np

class MatrixProperties():
    
    " ------------------------ Initialize Class ------------------------"
    def __init__(self):
        pass
    
    " ------------------------ Compute Condition Number ------------------------"
    def _condition_number(self, X : np.ndarray) -> float:

        """
        Args:
             X : data matrix X.

        """

        # np.matmul(X.transpose(), X)
        eigenvalues, eigenvectors = np.linalg.eig(X)

        # condition number:
        kappa = np.sqrt(max(eigenvalues))/np.sqrt(min(eigenvalues))

        print(f'Condition number of the matrix is: {kappa}')

        return kappa
    
    " ------------------------ Compute Matrix Rank ------------------------"
    def _rank(self, X : np.ndarray) -> float:
        
        """
        Args:
             X : data matrix X.

        """
        
        rank = np.linalg.matrix_rank(X)
        
        if X.shape[0] == rank:
            print(f'Matrix has full rank, with rank: {rank}.')
            
        else:
            print(f'Matrix does not have full rank, with rank of: {rank}.')