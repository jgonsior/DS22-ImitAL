import math
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
import random
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .experiment_setup_lib import log_it

# refactoring: dataset wird ein pandas dataframe:
# id, feature_columns, label (-1 heißt gibt's noch nie, kann auch weak sein), true_label, dataset (train, test, val?), label_source


class DataStorage:
    def __init__(
        self,
        RANDOM_SEED,
        DATASET_NAME,
        DATASETS_PATH,
        START_SET_SIZE,
        TEST_FRACTION,
        hyper_parameters,
    ):
        if RANDOM_SEED != -1:
            np.random.seed(RANDOM_SEED)
            random.seed(RANDOM_SEED)
            self.RANDOM_SEED = RANDOM_SEED

        log_it("Loading " + DATASET_NAME)

        if DATASET_NAME == "dwtc":
            self._load_dwtc(DATASETS_PATH)
        elif DATASET_NAME == "synthetic":
            self._load_synthetic()
        else:
            self._load_alc(DATASET_NAME, DATASETS_PATH)

        self.label_encoder = LabelEncoder()
        self.df["true_label"] = self.label_encoder.fit_transform(self.df["true_label"])

        # feature normalization
        scaler = RobustScaler()
        self.df[self.feature_columns] = scaler.fit_transform(
            self.df[self.feature_columns]
        )

        # scale back to [0,1]
        scaler = MinMaxScaler()
        self.df[self.feature_columns] = scaler.fit_transform(
            self.df[self.feature_columns]
        )

        #  X_temp = pd.DataFrame(X_temp, dtype=float)
        #  Y_temp = pd.DataFrame(Y_temp, dtype=int)
        #
        #  X_temp = X_temp.apply(pd.to_numeric, downcast="float", errors="ignore")
        #  Y_temp = Y_temp.apply(pd.to_numeric, downcast="integer", errors="ignore")
        self.df["dataset"] = ["train"] * self.amount_of_training_samples + ["test"] * (
            len(self.df) - self.amount_of_training_samples
        )
        """ 
        1. get start_set from X_labeled
        2. if X_unlabeled is None : experiment!
            2.1 if X_test: rest von X_labeled wird X_train_unlabeled
            2.2 if X_test is none: split rest von X_labeled in X_train_unlabeled und X_test
           else (kein experiment):
           X_unlabeled wird X_unlabeled, rest von X_labeled wird X_train_unlabeled_
        
        """

        # separate X_labeled into start_set and labeled _rest
        if START_SET_SIZE < len(self.get_df("train")["label"] != -1):
            # randomly select as much samples as needed from start set size

            # check if the minimum amount of labeled data is present in the start set size
            labels_not_in_start_set = set(range(0, len(self.label_encoder.classes_)))
            all_label_in_start_set = False

            for Y in self.get_Y("train"):
                if Y in labels_not_in_start_set:
                    labels_not_in_start_set.remove(Y)
                if len(labels_not_in_start_set) == 0:
                    all_label_in_start_set = True
                    break
            print(self.get_df("train"))

            if not all_label_in_start_set:
                #  if len(self.get_df("train")['label'] != None) == 0:
                #      print("Please specify at least one labeled example of each class")
                #      exit(-1)

                # move more data here from the classes not present
                for label in labels_not_in_start_set:
                    # select a random sample of this sample which is NOT yet labeled
                    potential_label_candidates = self.get_df(
                        "train", {"true_label": label, "label": -1}
                    )
                    print("label" + str(label))
                    print(potential_label_candidates)
                    if len(potential_label_candidates) == 0:
                        print(
                            "Please specify at least one labeled example of each class"
                        )
                        exit(-1)

                    potential_label_candidates.iloc[0]["label"] = label
            print(self.get_df("train"))
        if START_SET_SIZE == len(X_labeled):
            X_labeled_rest = None
            self.X_train_labeled = X_labeled
            Y_labeled_rest = None
            self.Y_train_labeled = Y_labeled

        else:
            (
                X_labeled_rest,
                self.X_train_labeled,
                Y_labeled_rest,
                self.Y_train_labeled,
            ) = train_test_split(X_labeled, Y_labeled, test_size=START_SET_SIZE)

        if X_unlabeled is not None:
            self.X_train_unlabeled = X_unlabeled
            self.Y_train_unlabeled = pd.DataFrame(
                columns=Y_labeled_rest.columns, dtype=int
            )

            self.X_test = X_labeled_rest
            self.Y_test = Y_labeled_rest
        else:
            # experiment setting!
            # create some fake unlabeled data

            if X_test is not None:
                self.X_train_unlabeled = X_labeled_rest
                self.Y_train_unlabeled = Y_labeled_rest
                self.X_test = X_test
                self.Y_test = Y_test
            else:
                # further split labeled rest for train_test
                (
                    self.X_train_unlabeled,
                    self.X_test,
                    self.Y_train_unlabeled,
                    self.Y_test,
                ) = train_test_split(X_labeled_rest, Y_labeled_rest, TEST_FRACTION)

        Y_train_labeled_set = set(self.Y_train_labeled[0].to_numpy())

        self._print_data_segmentation()

        self.X_train_unlabeled_cluster_indices = {}

        # remove the labeled data from X_train_labeled and merge it with the unlabeled data
        # while preserving the labels
        # and storing the indics of the labeled data
        # so that the first iteration can be a "fake iteration zero" of the AL cycle
        # (metrics will than automatically be calculated for this one too)
        self.prepare_fake_iteration_zero()
        log_it(self.X_train_labeled.shape)
        self.label_encoder = label_encoder

        log_it("Loaded " + DATASET_NAME)

    def _load_dwtc(self, DATASETS_PATH):
        self.df = pd.read_csv(DATASETS_PATH + "/dwtc/aft.csv", index_col="id")

        # shuffle df
        self.df = self.df.sample(frac=1, random_state=self.RANDOM_SEED).reset_index(
            drop=True
        )

        self.feature_columns = self.df.columns.to_list()
        self.feature_columns.remove("CLASS")
        self.df.rename({"CLASS": "true_label"}, axis="columns", inplace=True)

        self.amount_of_training_samples = int(len(self.df) * 0.5)
        self.df["label"] = -1

    def _load_synthetic(self, kwargs):
        self.X_data, self.Y_temp = make_classification(**kwargs)
        self.df = pd.DataFrame(self.X_data)

        # replace labels with strings
        Y_temp = Y_temp.astype("str")
        for i in range(0, kwargs["n_classes"]):
            np.place(Y_temp, Y_temp == str(i), chr(65 + i))

        # feature_columns fehlt

        self.df["true_label"] = Y_temp
        self.df["label"] = -1

    def _load_alc(self, DATASET_NAME, DATASETS_PATH):
        df = pd.read_csv(
            DATASETS_PATH + "/al_challenge/" + DATASET_NAME + ".data",
            header=None,
            sep=" ",
        )
        # feature_columns fehlt

        # shuffle df
        df = df.sample(frac=1, random_state=RANDOM_SEED)

        df = df.replace([np.inf, -np.inf], -1)
        df = df.fillna(0)

        labels = pd.read_csv(
            DATASETS_PATH + "/al_challenge/" + DATASET_NAME + ".label", header=None
        )

        labels = labels.replace([-1], "A")
        labels = labels.replace([1], "B")
        df["true_label"] = labels[0]
        #  Y_temp = labels[0].to_numpy()
        train_indices = {
            "ibn_sina": 10361,
            "hiva": 21339,
            "nova": 9733,
            "orange": 25000,
            "sylva": 72626,
            "zebra": 30744,
        }
        self.amount_of_training_samples = train_indices[DATASET_NAME]
        self.df["label"] = -1

        self.df = df

    def get_X(self, dataset=None):
        if dataset is None:
            return self.df[self.feature_columns]
        else:
            return self.df[self.df["dataset"] == dataset][self.feature_columns]

    def get_Y(self, dataset=None):
        if dataset is None:
            return self.df["label"]
        else:
            return self.df[self.df["dataset"] == dataset]["label"]

    def get_df(self, dataset=None, column_conditions=None):
        if dataset is None:
            df = self.df
        else:
            df = self.df[self.df["dataset"] == dataset]

        if column_conditions != None:
            selection = True
            for k, v in column_conditions.items():
                selection &= df[k] == v
            df = df.loc[selection]

        return df

    def prepare_fake_iteration_zero(self):
        # fake iteration zero where we add the given ground truth labels all at once
        original_X_train_labeled = self.X_train_labeled
        original_Y_train_labeled = self.Y_train_labeled

        self.X_train_labeled = pd.DataFrame(
            columns=original_X_train_labeled.columns, dtype=float
        )

        if self.X_train_labeled is not None:
            self.Y_train_labeled = pd.DataFrame(
                columns=original_Y_train_labeled.columns, dtype=int
            )

        # this one is a bit tricky:
        # we merge both back together here -> but solely for the purpose of using them as the first oracle query down below
        self.X_train_unlabeled = pd.concat(
            [original_X_train_labeled, self.X_train_unlabeled]
        )

        self.Y_train_unlabeled = pd.concat(
            [original_Y_train_labeled, self.Y_train_unlabeled]
        )
        self.ground_truth_indices = original_X_train_labeled.index.tolist()

        self.Y_train_strong_labels = pd.DataFrame.copy(original_Y_train_labeled)

    def _print_data_segmentation(self):
        len_train_labeled = len(self.X_train_labeled)
        len_train_unlabeled = len(self.X_train_unlabeled)
        #  len_test = len(self.X_test)

        len_total = len_train_unlabeled + len_train_labeled  # + len_test

        log_it(
            "size of train  labeled set: %i = %1.2f"
            % (len_train_labeled, len_train_labeled / len_total)
        )
        log_it(
            "size of train unlabeled set: %i = %1.2f"
            % (len_train_unlabeled, len_train_unlabeled / len_total)
        )

    def _move_queries_from_a_to_b(self, X_query, Y_query, query_indices, a, b):
        a_X, a_Y = a
        b_X, b_Y = b
        # move new queries from unlabeled to labeled dataset
        b_X = b_X.append(X_query)
        a_X = a_X.drop(query_indices)
        self.X_train_labeled = self.X_train_labeled.append(X_query)
        self.X_train_unlabeled = self.X_train_unlabeled.drop(query_indices)

        try:
            self.Y_train_strong_labels = self.Y_train_strong_labels.append(
                self.Y_train_unlabeled.loc[query_indices]
            )
        except KeyError:
            # in a non experiment setting an error will be thrown because self.Y_train_unlabeled of course doesn't contains the labels
            for query_index in query_indices:
                self.Y_train_strong_labels.loc[query_index] = [-1]

        self.Y_train_labeled = self.Y_train_labeled.append(Y_query)
        self.Y_train_unlabeled = self.Y_train_unlabeled.drop(
            query_indices, errors="ignore"
        )

        # remove indices from all clusters in unlabeled and add to labeled
        for cluster_id in self.X_train_unlabeled_cluster_indices.keys():
            list_to_be_removed_and_appended = []
            for indice in query_indices:
                if indice in self.X_train_unlabeled_cluster_indices[cluster_id]:
                    list_to_be_removed_and_appended.append(indice)

            # don't change a list you're iterating over!
            for indice in list_to_be_removed_and_appended:
                self.X_train_unlabeled_cluster_indices[cluster_id].remove(indice)
                self.X_train_labeled_cluster_indices[cluster_id].append(indice)

        # remove possible empty clusters
        self.X_train_unlabeled_cluster_indices = {
            k: v
            for k, v in self.X_train_unlabeled_cluster_indices.items()
            if len(v) != 0
        }

    def move_labeled_queries(self, X_query, Y_query, query_indices):
        self._move_queries_from_a_to_b(
            X_query,
            Y_query,
            query_indices,
            a=(self.X_train_labeled, self.Y_train_labeled),
            b=(self.X_train_unlabeled, self.Y_train_unlabeled),
        )
