import numpy as np
from abc import ABC, abstractmethod
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from numpy.testing import assert_allclose

from src.Direct_Solvers import DirectSolver

'-------------------------------- Load custom classes ------------------------------------'
class Models(DirectSolver):
    
    '-------------------------------- Initialize Class ------------------------------------'
    def __init__(self):
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
        X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

        A = X_train_b.T @ X_train_b
        b = X_train_b.T @ y_train
        
        _, _, _, w = self._solver(A, b, mode=mode)

        J_train = (1/2)*np.linalg.norm(X_train_b @ w - y_train)**2
        J_test  = (1/2)*np.linalg.norm(X_test_b @ w - y_test)**2

        if print_statement:
            print(f'OLS J_train(w) = {J_train}')
            print(f'OLS J_test(w) = {J_test}')

        return w, J_train, J_test
    
    # This method now returns the weights 'w' for verification.
    def _ridge_regression(self, X_train : np.ndarray, X_test : np.ndarray, y_train : np.ndarray, 
                          y_test : np.ndarray, mode : str = 'Cholesky', lam : float = 1, 
                          print_statement : bool = False) -> tuple[np.ndarray, float, float]:
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
        X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
        
        regularization_matrix = lam * np.identity(X_train_b.shape[1])
        regularization_matrix[0, 0] = 0
        
        A = X_train_b.T @ X_train_b + regularization_matrix
        b = X_train_b.T @ y_train
        _, _, _, w = self._solver(A, b, mode=mode)
        
        weights_only = w[1:]

        J_train = (1/2)*np.linalg.norm(X_train_b @ w - y_train)**2 + (lam/2)*np.linalg.norm(weights_only)**2
        J_test  = (1/2)*np.linalg.norm(X_test_b @ w - y_test)**2 + (lam/2)*np.linalg.norm(weights_only)**2

        if print_statement:
            print(f'Ridge J_train(w) = {J_train}')
            print(f'Ridge J_test(w) = {J_test}')

        return w, J_train, J_test
    
    def _train_hinge_loss(self, X_train : np.ndarray, y_train : np.ndarray, 
                          X_test : np.ndarray, y_test : np.ndarray, 
                          lam : float, learning_rate: float, train_steps: int):
        dimension = X_train.shape[1]
        model = HingeLossClassification(dimension, lam, learning_rate)
        
        loss_history = []
        test_loss_history = []
        
        for _ in range(train_steps):
            train_loss = model.forward(X_train, y_train)
            model.backward()
            model.step()
            
            temp_model = HingeLossClassification(dimension, lam, learning_rate)
            temp_model.weight = model.weight
            test_loss = temp_model.forward(X_test, y_test)
            
            loss_history.append(train_loss)
            test_loss_history.append(test_loss)
            
        return model, loss_history, test_loss_history

    def _train_logistic_loss(self, X_train : np.ndarray, y_train : np.ndarray, 
                             X_test : np.ndarray, y_test : np.ndarray, 
                             lam : float, learning_rate: float, train_steps: int):
        dimension = X_train.shape[1]
        model = LogisticLossClassification(dimension, lam, learning_rate)
        
        loss_history = []
        test_loss_history = []
        
        for _ in range(train_steps):
            train_loss = model.forward(X_train, y_train)
            model.backward()
            model.step()
            
            temp_model = LogisticLossClassification(dimension, lam, learning_rate)
            temp_model.weight = model.weight
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
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y = y.reshape(-1, 1)
        output = X_b @ self.weight
        margins = 1 - y * output
        loss = np.maximum(0, margins)
        regularization = 0.5 * self.l * np.sum(self.weight[1:]**2) 
        self.cache = (X_b, y, margins)
        return (loss.sum() / X.shape[0]) + regularization

    def backward(self):
        X_b, y, margins = self.cache
        indicator = (margins > 0).astype(float)
        grad_loss = -(X_b.T @ (indicator * y)) / X_b.shape[0]
        reg_grad = self.l * self.weight
        reg_grad[0] = 0
        self.grad = grad_loss + reg_grad
        return self.grad
    
class LogisticLossClassification(Loss):
    def forward(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y = y.reshape(-1, 1)
        output = X_b @ self.weight
        logits = -y * output
        loss_vec = np.logaddexp(0.0, logits)
        regularization = 0.5 * self.l * np.sum(self.weight[1:]**2) 
        self.cache = (X_b, y, output)
        return (loss_vec.sum() / X.shape[0]) + regularization

    def backward(self):
        X_b, y, output = self.cache
        probs = 1.0 / (1.0 + np.exp(y * output))
        grad_loss = -(X_b.T @ (y * probs)) / X_b.shape[0]
        reg_grad = self.l * self.weight
        reg_grad[0] = 0
        self.grad = grad_loss + reg_grad
        return self.grad

# ====================================================================================
# Verification Code (This part is AI-Generated/Assisted)
# ====================================================================================

def verify_ridge_regression():
    print("--- Verifying Ridge Regression ---")
    X, y, _ = make_regression(n_samples=100, n_features=10, n_informative=5, noise=10, coef=True, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # MODIFIED: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lam = 1.0

    # MODIFIED: Your implementation call
    my_models_instance = Models()
    my_weights, _, _ = my_models_instance._ridge_regression(X_train, X_test, y_train, y_test, lam=lam)
    
    # Scikit-learn's implementation
    sklearn_model = Ridge(alpha=lam, fit_intercept=True)
    # MODIFIED: Fit on training data only
    sklearn_model.fit(X_train, y_train)
    
    sklearn_weights = np.concatenate(([sklearn_model.intercept_], sklearn_model.coef_))

    print("Your Ridge Weights:", my_weights)
    print("Sklearn Ridge Weights:", sklearn_weights)
    
    try:
        assert_allclose(my_weights, sklearn_weights, rtol=1e-5, atol=1e-5)
        print("✅ Ridge Regression implementation is correct.\n")
    except AssertionError as e:
        print("❌ Ridge Regression implementation is NOT correct.")
        print(e)


def verify_hinge_loss():
    print("--- Verifying Hinge Loss (Linear SVM) ---")
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=0, random_state=42)
    y[y == 0] = -1
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # MODIFIED: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lam = 0.1
    learning_rate = 0.01
    train_steps = 1000

    # MODIFIED: Your implementation call
    my_models_instance = Models()
    my_hinge_model, _, _ = my_models_instance._train_hinge_loss(
        X_train, y_train, X_test, y_test, lam, learning_rate, train_steps
    )
    my_weights = my_hinge_model.weight.flatten()

    # Scikit-learn's implementation
    sklearn_svm = SGDClassifier(loss='hinge', penalty='l2', alpha=lam, 
                                learning_rate='constant', eta0=learning_rate, 
                                fit_intercept=True, max_iter=train_steps, tol=None, random_state=42)
    # MODIFIED: Fit on training data only
    sklearn_svm.fit(X_train, y_train)
    sklearn_weights = np.concatenate((sklearn_svm.intercept_, sklearn_svm.coef_.flatten()))
    
    print("Your Hinge Loss Weights:", my_weights)
    print("Sklearn SGD Hinge Weights:", sklearn_weights)
    
    try:
        # NOTE: Tolerances are higher due to random weight initialization and SGD nature
        assert_allclose(my_weights, sklearn_weights, rtol=0.1, atol=0.1)
        print("✅ Hinge Loss implementation is correct.\n")
    except AssertionError as e:
        print("❌ Hinge Loss implementation seems incorrect.")
        print(e)


def verify_logistic_loss():
    print("--- Verifying Logistic Loss ---")
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=0, random_state=42)
    y[y == 0] = -1
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # MODIFIED: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lam = 0.1
    learning_rate = 0.01
    train_steps = 1000

    # MODIFIED: Your implementation call
    my_models_instance = Models()
    my_logistic_model, _, _ = my_models_instance._train_logistic_loss(
        X_train, y_train, X_test, y_test, lam, learning_rate, train_steps
    )
    my_weights = my_logistic_model.weight.flatten()

    # Scikit-learn's implementation
    sklearn_log_reg = SGDClassifier(loss='log_loss', penalty='l2', alpha=lam, 
                                    learning_rate='constant', eta0=learning_rate,
                                    fit_intercept=True, max_iter=train_steps, tol=None, random_state=42)
    # MODIFIED: Fit on training data only
    sklearn_log_reg.fit(X_train, y_train)
    sklearn_weights = np.concatenate((sklearn_log_reg.intercept_, sklearn_log_reg.coef_.flatten()))

    print("Your Logistic Loss Weights:", my_weights)
    print("Sklearn SGD Logistic Weights:", sklearn_weights)
    
    try:
        # NOTE: Tolerances are higher due to random weight initialization and SGD nature
        assert_allclose(my_weights, sklearn_weights, rtol=0.1, atol=0.1)
        print("✅ Logistic Loss implementation is correct.\n")
    except AssertionError as e:
        print("❌ Logistic Loss implementation seems incorrect.")
        print(e)


if __name__ == '__main__':
    verify_ridge_regression()
    verify_hinge_loss()
    verify_logistic_loss()