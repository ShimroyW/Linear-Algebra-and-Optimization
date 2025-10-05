import numpy as np

class DirectSolver():
    
    " ------------------------ Initialize Class ------------------------"
    def __init__(self):
        pass
    
    
    " ----------------------- Cholesky Factorization -------------------------"
    def _cholesky_decomposition(self, X : np.ndarray) -> np.ndarray:
        
        """
        Args:
             X : data matrix X.

        """

        C = np.zeros_like(X)

        for k in range(X.shape[0]):

            C[k, k] = np.sqrt(X[k, k] - np.sum(C[k, :k]**2))

            for i in range(k + 1, X.shape[0]):
                C[i, k] = (1/C[k, k])*(X[i, k] - np.sum(C[i, :k]*C[k, :k]))
                
        return C, C.transpose()    
    
        
    " ----------------------- LU Factorization -------------------------"
    def _LU_factorization(self, X : np.ndarray) -> np.ndarray:

        """
        Args:
             X : data matrix X.

        """

        # Check of matrix is square
        if not X.shape[0] == X.shape[1]:
            raise ValueError('Input matrix must be square')

        # Initialize L-Matrix:
        L = np.zeros_like(X)
        np.fill_diagonal(L,1)

        # Initialize U-matrix:
        U = np.copy(X)

        for i in range(X.shape[0] - 1):
            for j in range(i + 1, X.shape[0]):

                L[j, i]  = U[j, i]/U[i, i] 
                U[j, i:] = U[j, i:] - L[j, i]*U[i, i:]
                U[j, i]  = 0

        return L, U
    
    
    " ----------------------- QR Factorization -------------------------"
    def _QR_factorization(self, X : np.ndarray) -> np.ndarray:
        
        """
        Args:
             X : data matrix X.

        """

        # Initialize matrices:
        Q = np.zeros((X.shape[0], X.shape[1]))
        R = np.zeros((X.shape[1], X.shape[1]))


        # Create orthogonal matrix by Gram-Schmidt process:
        for i in range(X.shape[1]):
            r = np.array(X[:, i]) # -> Use copy/create array! Since this is a reference to a slice!

            for j in range(i):
                r -= np.dot(r, Q[:, j])*Q[:, j]

            Q[:, i] = r/np.linalg.norm(r)

        # Create upper-triangular matrix R:
        for i in range(X.shape[1]):
            for j in range(i + 1):
                R[j, i] = np.dot(X[:, i], Q[:, j])        

        return Q, R        

    
    " ----------------------- Forward Substitution -------------------------"
    def _forward_substitution(self, L : np.ndarray, y : np.ndarray) -> np.ndarray:

        """
        Args:
             L : lower triangular matrix L.
             y : output vector y

        """

        q = np.copy(y)

        for i in range(y.shape[0]):  

            for j in range(i):
                q[i] = q[i] - (L[i, j]*q[j])

            q[i] = q[i] / L[i, i]

        return q
    
    
    " ----------------------- Backward Substitution -------------------------"
    def _backward_substitution(self, U : np.ndarray, q : np.ndarray) -> np.ndarray:

        """
        Args:
             U : upper triangular matrix L.
             y : output vector y

        """

        w = np.zeros_like(q)

        for i in range(w.shape[0], 0, -1):
            w[i - 1] = (q[i - 1] - np.dot(U[i - 1, i:], w[i:]))/U[i - 1, i - 1]

        return w
    
    
    " ----------------------- Solver -------------------------"
    def _solver(self, X : np.ndarray, y : np.ndarray, mode : str = 'LU') -> np.ndarray:    
        
        match mode:
            
            case 'LU':
                L, U = self._LU_factorization(X)
            
            case 'Cholesky':
                L, U = self._cholesky_decomposition(X)
                        
        # Forward substitution:
        q = self._forward_substitution(L, y)
        
        # Backward substitution:
        w = self._backward_substitution(U, q)
        
        return L, U, q, w