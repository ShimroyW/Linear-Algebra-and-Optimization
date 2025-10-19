import numpy as np
from abc import ABC, abstractmethod

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
    
'-------------------------------- Load custom classes ------------------------------------'
class Models(DirectSolver):
    
    '-------------------------------- Initialize Class ------------------------------------'
    def __init__(self):
        pass
    
    " ---------------------------------------- Ordinary Least Squears Method ---------------------------------------------- "
    def _ordinary_least_squares(self, X_train : np.ndarray, X_test : np.ndarray, y_train : np.ndarray, 
                                y_test : np.ndarray, mode : str = 'Cholesky', 
                                print_statement : bool = False) -> np.ndarray:

        """
        Args:
             mode            : Mode for (direct) Matrix solver.
             X_train         : Train set for data matrix X.
             X_test          : Test set for data matrix X.
             y_train         : Output vector y for trainset.
             y_test          : Output vector y for testset.
             print_statement : Print Loss function output.

        """

        X_ols = np.matmul(X_train.transpose(), X_train)

        # Solve with LU Factorization:
        L, U, q, w = self._solver(X_ols, np.matmul(X_train.transpose(), y_train), mode = mode)

        # Compute cost function:
        J_train = (1/2)*np.linalg.norm(np.matmul(X_train, w) - y_train)**2
        J_test  = (1/2)*np.linalg.norm(np.matmul(X_test, w) - y_test)**2

        # Print output:
        if print_statement:
            print(f'J_train(w) = {J_train}')
            print(f'J_test(w) = {J_test}')

        return J_train, J_test
    
    
    " ---------------------------------------- Ridge Regression Method ---------------------------------------------- "
    def _ridge_regression(self, X_train : np.ndarray, X_test : np.ndarray, y_train : np.ndarray, 
                          y_test : np.ndarray, mode : str = 'Cholesky', lam : float = 1, 
                          print_statement : bool = False) -> np.ndarray:

        """
        Args:
             mode            : Mode for (direct) Matrix solver.
             X_train         : Train set for data matrix X.
             X_test          : Test set for data matrix X.
             y_train         : Output vector y for trainset.
             y_test          : Output vector y for testset.
             print_statement : Print Loss function output.
             lam             : Lambda value.

        """

        X_ridge = np.matmul(X_train.transpose(), X_train) + lam*np.identity(X_train.shape[1])

        # Solve with LU Factorization:
        L, U, q, w = self._solver(X_ridge, np.matmul(X_train.transpose(), y_train), mode = mode)

        # Compute cost function:
        J_train = (1/2)*np.linalg.norm(np.matmul(X_train, w) - y_train)**2 + (lam/2)*np.linalg.norm(w)**2
        J_test  = (1/2)*np.linalg.norm(np.matmul(X_test, w) - y_test)**2 + (lam/2)*np.linalg.norm(w)**2

        # Print output:
        if print_statement:
            print(f'J_train(w) = {J_train}')
            print(f'J_test(w) = {J_test}')

        return J_train, J_test
    
    
    " ---------------------------------------- Hinge Loss Classification ---------------------------------------------- "
    def _hinge_loss_classification(self, X_train : np.ndarray, X_test : np.ndarray, y_train : np.ndarray, 
                                   y_test : np.ndarray, mode : str = 'Cholesky', lam : float = 1, 
                                   print_statement : bool = False) -> np.ndarray:

        """
        Args:
             mode            : Mode for (direct) Matrix solver.
             X_train         : Train set for data matrix X.
             X_test          : Test set for data matrix X.
             y_train         : Output vector y for trainset.
             y_test          : Output vector y for testset.
             print_statement : Print Loss function output.
             lam             : Lambda value.

        """

        X_hinge_loss = np.matmul(X_train.transpose(), X_train) + lam*np.identity(X_train.shape[1])

        # Solve with LU Factorization:
        L, U, q, w = self._solver(X_hinge_loss, np.matmul(X_train.transpose(), y_train), mode = mode)

        # Compute cost function:
        H_train = 0
        H_test  = 0

        for i in range(X_train.shape[0]):
            x_col    = X_train[i, :]
            s        = y_train[i]*np.dot(w.T, X_train[i, :])

            H_train += max(0, 1-s)

        J_train = H_train + (lam/2)*np.linalg.norm(w)**2   

        for i in range(X_test.shape[0]):
            x_col  = X_test[i, :]
            s      = y_test[i]*np.dot(w.T, X_test[i, :])

            H_test += max(0, 1-s)

        J_test = H_test + (lam/2)*np.linalg.norm(w)**2   

        # Print output:
        if print_statement:
            print(f'J_train(w) = {J_train}')
            print(f'J_test(w) = {J_test}')

        return J_train, J_test
    
    " ---------------------------------------- Logistic Classification ---------------------------------------------- "
    def _logistic_classification(self, X_train : np.ndarray, X_test : np.ndarray, y_train : np.ndarray, 
                                 y_test : np.ndarray, mode : str = 'Cholesky', lam : float = 1, 
                                 print_statement : bool = False) -> np.ndarray:

        """
        Args:
             mode            : Mode for (direct) Matrix solver.
             X_train         : Train set for data matrix X.
             X_test          : Test set for data matrix X.
             y_train         : Output vector y for trainset.
             y_test          : Output vector y for testset.
             print_statement : Print Loss function output.
             lam             : Lambda value.

        """

        X_logistic = np.matmul(X_train.transpose(), X_train) + lam*np.identity(X_train.shape[1])

        # Solve with LU Factorization:
        L, U, q, w = self._solver(X_logistic, np.matmul(X_train.transpose(), y_train), mode = mode)

        # Compute cost function:
        L_train = 0
        L_test  = 0

        for i in range(X_train.shape[0]):
            x_col    = X_train[i, :]
            s        = y_train[i]*np.dot(w.T, X_train[i, :])

            L_train += np.log(1 + np.exp(-s))

        J_train = L_train + (lam/2)*np.linalg.norm(w)**2   

        for i in range(X_test.shape[0]):
            x_col   = X_test[i, :]
            s       = y_test[i]*np.dot(w.T, X_test[i, :])

            L_test += np.log(1 + np.exp(-s))

        J_test = L_test + (lam/2)*np.linalg.norm(w)**2   

        # Print output:
        if print_statement:
            print(f'J_train(w) = {J_train}')
            print(f'J_test(w) = {J_test}')

        return J_train, J_test
    


# class defining common loss interface
class Loss(ABC):

    # common initializer
    def __init__(self, dimension, l, learning_rate):
        self.dimension = dimension
        self.l = l
        self.learning_rate = learning_rate
        self.weight = np.random.uniform(low=0, high=1, size=(dimension, 1))
        self.cache = None
        self.grad = None

    # calculates loss and stores varibales necessary to calculate gradient
    @abstractmethod
    def forward(self):
        pass
    
    # calculates gradient based on stored variables from forward pass
    @abstractmethod
    def backward(self):
        pass

    # updates weights based on calculated gradient
    @abstractmethod
    def step():
        pass



class HingeLossClassification(Loss):
    

    def forward(self, X, y):
        y = y.reshape(-1, 1)               # ensure column vector and not (N,)
        output = X @ self.weight          
        margins = 1 - y * output           
        loss = np.maximum(0, margins)      
        self.cache = (X, y, margins)
        return loss.sum() + 0.5 * self.l * np.linalg.norm(self.weight)**2

    def backward(self):
        X, y, margins = self.cache
        indicator = (margins > 0).astype(float)  
        self.grad = -(X.T @ (indicator * y)) + self.l * self.weight  
        return self.grad

    def step(self):
        self.weight -= self.learning_rate * self.grad
    
class LogisticLossClassification(Loss):

    def forward(self, X, y):
        y = y.reshape(-1, 1)                 # ensure column vector and not (N,)
        output = X @ self.weight             
        logits = -y * output                 
        loss_vec = np.logaddexp(0.0, logits) 
        self.cache = (X, y, logits)
        return loss_vec.sum() + 0.5 * self.l * np.linalg.norm(self.weight)**2

    def backward(self):
        X, y, logits = self.cache
        probs = 1.0 / (1.0 + np.exp(-logits))            
        self.grad = -(X.T @ (y * probs)) + self.l * self.weight 
        return self.grad

    def step(self):
        self.weight -= self.learning_rate * self.grad

