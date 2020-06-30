from active_learning.BaseOracle import BaseOracle


class FakeExperimentOracle(BaseOracle):
    def get_labeled_samples(self, query_indices, data_storage):
        return data_storage.get_true_label(query_indices)
