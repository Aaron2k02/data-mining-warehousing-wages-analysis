import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.exceptions import ConvergenceWarning
import warnings

def run_classifier(clf, param_grid, X_train, y_train, X_test, y_test, title, n_splits=5, n_iter_search=10, random_state=123):
    """
    Perform cross-validation, hyperparameter tuning, and evaluation for a given classifier.
    
    Parameters:
    - clf: Classifier to be tuned and evaluated.
    - param_grid: Dictionary of hyperparameters for RandomizedSearchCV.
    - X_train, y_train: Training dataset.
    - X_test, y_test: Test dataset.
    - title: Title for the classifier in the output.
    - n_splits: Number of splits for cross-validation (default: 5).
    - n_iter_search: Number of parameter combinations for RandomizedSearchCV (default: 10).
    - random_state: Random state for reproducibility (default: 123).
    
    Returns:
    - best_estimator: Best classifier after tuning.
    """
    # Ignore warnings from MLP convergence issues
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    
    # -----------------------------------------------------
    # Cross-Validation Setup
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    gs = RandomizedSearchCV(
        clf,
        param_distributions=param_grid,
        n_iter=n_iter_search,
        cv=cv,
        scoring='accuracy',
        return_train_score=True,
        random_state=random_state
    )

    # -----------------------------------------------------
    # Perform Cross-Validation and Hyperparameter Tuning
    gs.fit(X_train, y_train)
    
    print(f"\n--- RandomizedSearchCV ({title}) ---")
    print(gs)

    print(f"\n--- Cross-Validation Results ({title}) ---")
    print("The best parameters are:", gs.best_params_)
    print("Mean cross-validation accuracy: %.2f%%" % (gs.best_score_ * 100))

    # -----------------------------------------------------
    # Evaluate Model on Test and Validation Sets
    print("\n--- Test Results ---")

    # Predict on Test Set
    y_test_pred = gs.best_estimator_.predict(X_test)

    # Test Set Metrics
    print('Accuracy: %.2f%%' % (accuracy_score(y_test, y_test_pred) * 100))
    print('Precision: %.2f%%' % (precision_score(y_test, y_test_pred, average='weighted') * 100))
    print('Recall: %.2f%%' % (recall_score(y_test, y_test_pred, average='weighted') * 100))

    # Return the best estimator for further use
    return gs.best_estimator_