import numpy as np
from abc import ABC, abstractmethod
# import the direct solver
from src.Direct_Solvers import DirectSolver 
    
'-------------------------------- Load custom classes ------------------------------------'
class Models(DirectSolver):
    
    '-------------------------------- Initialize Class ------------------------------------'
    def __init__(self):
        # Initialize the parent DirectSolver class
        super().__init__()
    
    " ---------------------------------------- Ordinary Least Squares Method ---------------------------------------------- "
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
        # Add bias (ones) to the training and test data
        X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

        A = X_train_b.T @ X_train_b
        b = X_train_b.T @ y_train
        
        L, U, q, w = self._solver(A, b, mode=mode)
        # cost func
        J_train = (1/2)*np.linalg.norm(X_train_b @ w - y_train)**2
        J_test  = (1/2)*np.linalg.norm(X_test_b @ w - y_test)**2

        if print_statement:
            print(f'OLS J_train(w) = {J_train}')
            print(f'OLS J_test(w) = {J_test}')

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
        # Add bias (ones) to the training and test data
        X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
        
        # Regularization matrix, the identity matrix scaled by lambda.
        regularization_matrix = lam * np.identity(X_train_b.shape[1])
        
        # Set the first element to 0 to NOT regularize the bias term.
        regularization_matrix[0, 0] = 0
        
        # Ridge solution using the normal equation.
        A = X_train_b.T @ X_train_b + regularization_matrix
        b = X_train_b.T @ y_train
        L, U, q, w = self._solver(A, b, mode=mode)
        
        # weights w include the bias term w[0] and feature weights w[1:]
        weights_only = w[1:]

        J_train = (1/2)*np.linalg.norm(X_train_b @ w - y_train)**2 + (lam/2)*np.linalg.norm(weights_only)**2
        J_test  = (1/2)*np.linalg.norm(X_test_b @ w - y_test)**2 + (lam/2)*np.linalg.norm(weights_only)**2

        if print_statement:
            print(f'Ridge J_train(w) = {J_train}')
            print(f'Ridge J_test(w) = {J_test}')

        return J_train, J_test
    
    " ---------------------------------------- Hinge-Loss Classifier Training ---------------------------------------------- "
    def _train_hinge_loss(self, X_train : np.ndarray, y_train : np.ndarray, 
                          X_test : np.ndarray, y_test : np.ndarray, 
                          lam : float, learning_rate: float, train_steps: int):
        dimension = X_train.shape[1]
        model = HingeLossClassification(dimension, lam, learning_rate)
        
        loss_history = []
        test_loss_history = []
        
        for _ in range(train_steps):
            # full training step: forward, backward, and parameter update
            train_loss = model.forward(X_train, y_train)
            model.backward()
            model.step()
            
            # create a temporary HingeLoss object for test loss w/o corrupting cache for backprop
            temp_model = HingeLossClassification(dimension, lam, learning_rate)
            temp_model.weight = model.weight # Use current weights for eval
            test_loss = temp_model.forward(X_test, y_test)
            
            loss_history.append(train_loss)
            test_loss_history.append(test_loss)
            
        return model, loss_history, test_loss_history

    " ---------------------------------------- Log-Loss Classifier Training ---------------------------------------------- "
    def _train_logistic_loss(self, X_train : np.ndarray, y_train : np.ndarray, 
                             X_test : np.ndarray, y_test : np.ndarray, 
                             lam : float, learning_rate: float, train_steps: int):
        dimension = X_train.shape[1]
        model = LogisticLossClassification(dimension, lam, learning_rate)
        
        loss_history = []
        test_loss_history = []
        
        for _ in range(train_steps):
            # full training step: forward, backward, and parameter update
            train_loss = model.forward(X_train, y_train)
            model.backward()
            model.step()
            
            # calculate test loss for monitoring w/o corrupting the cache
            temp_model = LogisticLossClassification(dimension, lam, learning_rate)
            temp_model.weight = model.weight # Use current weights for eval
            test_loss = temp_model.forward(X_test, y_test)
            
            loss_history.append(train_loss)
            test_loss_history.append(test_loss)
            
        return model, loss_history, test_loss_history

# class defining common loss interface
class Loss(ABC):
    def __init__(self, dimension, l, learning_rate):
        self.dimension = dimension
        self.l = l
        self.learning_rate = learning_rate
        # Initialize weights w/ a bias term.
        self.weight = np.random.uniform(low=-0.1, high=0.1, size=(dimension + 1, 1))
        self.cache = None
        self.grad = None

    @abstractmethod
    def forward(self, X, y):
        pass
    
    @abstractmethod
    def backward(self):
        pass

    def step(self):
        self.weight -= self.learning_rate * self.grad

class HingeLossClassification(Loss):
    def forward(self, X, y):
        # Add bias term to X.
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y = y.reshape(-1, 1) # ensure column vector and not (N,)
        output = X_b @ self.weight
        margins = 1 - y * output
        loss = np.maximum(0, margins)
        # Regularize just the weights, not bias.
        regularization = 0.5 * self.l * np.sum(self.weight[1:]**2) 
        self.cache = (X_b, y, margins)
        return (loss.sum() / X.shape[0]) + regularization

    def backward(self):
        X_b, y, margins = self.cache
        indicator = (margins > 0).astype(float)
        grad_loss = -(X_b.T @ (indicator * y)) / X_b.shape[0]
        reg_grad = self.l * self.weight
        reg_grad[0] = 0 # Don't regularize the bias term.
        self.grad = grad_loss + reg_grad
        return self.grad
    
class LogisticLossClassification(Loss):
    def forward(self, X, y):
        # Add bias term to X
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y = y.reshape(-1, 1) # ensure column vector and not (N,)
        output = X_b @ self.weight
        logits = -y * output
        loss_vec = np.logaddexp(0.0, logits)
        # Regularize just the weights
        regularization = 0.5 * self.l * np.sum(self.weight[1:]**2) 
        self.cache = (X_b, y, output)
        return (loss_vec.sum() / X.shape[0]) + regularization

    def backward(self):
        X_b, y, output = self.cache
        probs = 1.0 / (1.0 + np.exp(y * output))
        grad_loss = -(X_b.T @ (y * probs)) / X_b.shape[0]
        reg_grad = self.l * self.weight
        reg_grad[0] = 0 # Don't regularize the bias term
        self.grad = grad_loss + reg_grad
        return self.grad