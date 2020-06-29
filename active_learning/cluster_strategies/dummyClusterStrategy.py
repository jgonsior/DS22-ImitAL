from .baseClusterStrategy import BaseClusterStrategy


class DummyClusterStrategy(BaseClusterStrategy):
    def get_cluster_indices(self, **kwargs):
        return []
        #  return self.data_storage.get_df()["cluster"].unique()
