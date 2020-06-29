from active_learning.BaseOracle import BaseOracle


class FakeExperimentOracle(BaseOracle):
    def get_labeled_samples(self, query_indices, data_storage):
        return data_storage.get_df().loc[query_indices]["true_label"]
