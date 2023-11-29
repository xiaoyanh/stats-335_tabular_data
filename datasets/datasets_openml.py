from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder as OHE

    import openml


# Fetch dataset from openML through their ID:
# for instance see: https://www.openml.org/search?type=data&id=44091
DATASETS = {
    'california': 44090,
    'wine': 44091,
}


class Dataset(BaseDataset):

    name = 'openml'

    install_cmd = 'conda'
    requirements = ["pip:openml"]

    parameters = {
        "dataset": list(DATASETS),
    }

    def get_data(self):
        """Get the data to be used to evaluate the ML algorithms.

        Returns
        -------
        X, y : ndarrays, (n_samples, n_features) and (n_samples,)
            The full data and labels, that will be split into train and test
            in the objective.
        preprocessor : sklearn transformer
            A transformer to preprocess the data before fitting the model.
            This part will be passed to the solver as is, and will be used to
            construct a `sklearn.Pipeline`.
        """
        dataset = openml.datasets.get_dataset(
            self.dataset, download_data=True, download_qualities=False,
            download_features_meta_data=True
        )
        X, y, cat_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )

        # Create a preprocessor that will one-hot encode categorical variables.
        # Here, the preprocessor can be adapted to the specific of the loaded
        # dataset, but it should be compatible with the sklearn.Pipeline API.
        size = X.shape[1]
        preprocessor = ColumnTransformer(
            [
                ("one_hot", OHE(categories="auto", handle_unknown="ignore"),
                 [X.columns[i] for i in range(size) if cat_indicator[i]]),
                ("numerical", "passthrough",
                 [X.columns[i] for i in range(size) if not cat_indicator[i]],)
            ]
        )

        return dict(
            X=X,
            y=y,
            preprocessor=preprocessor
        )
