from benchopt import safe_import_context, BaseSolver
from benchopt.stopping_criterion import SingleRunCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'logreg_l2'

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    parameters = {
        'C': [1e-1, 1, 10],
    }

    stopping_criterion = SingleRunCriterion()

    def set_objective(self, X_train, y_train, preprocessor):
        """Get the data to be passed to fit the solver.

        Parameters
        ----------
        X_train, y_train : ndarrays, (n_samples, n_features) and (n_samples,)
            The training data and labels, that will be used to fit the model.
        preprocessor : sklearn transformer
            A transformer to preprocess the data before fitting the model.
            This part should be used to construct a `sklearn.Pipeline`.
        """
        self.X_train, self.y_train = X_train, y_train
        self.model = make_pipeline(
            preprocessor,
            LogisticRegression(C=self.C, max_iter=1000)
        )

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        """Return the fitted model to be evaluated."""
        return dict(model=self.model)
