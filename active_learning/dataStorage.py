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
            df = self._load_dwtc(DATASETS_PATH)
        elif DATASET_NAME == "synthetic":
            df = self._load_synthetic()
        else:
            df = self._load_alc(DATASET_NAME, DATASETS_PATH)
        self.label_encoder = LabelEncoder()
        df["label"] = self.label_encoder.fit_transform(df["label"])

        # feature normalization
        scaler = RobustScaler()
        df[self.feature_columns] = scaler.fit_transform(df[self.feature_columns])

        # scale back to [0,1]
        scaler = MinMaxScaler()
        df[self.feature_columns] = scaler.fit_transform(df[self.feature_columns])

        # split dataframe into test, train_labeled, train_unlabeled
        self.test_X = df[self.amount_of_training_samples :].copy()
        self.test_Y = pd.DataFrame(
            data=self.test_X["label"], columns=["label"], index=self.test_X.index
        )
        del self.test_X["label"]

        train_data = df[: self.amount_of_training_samples].copy()
        train_labeled_data = pd.DataFrame(data=None, columns=train_data.columns)
        self.train_labeled_X = train_labeled_data
        self.train_labeled_Y = pd.DataFrame(
            data=None, columns=["label"], index=self.train_labeled_X.index
        )
        del self.train_labeled_X["label"]

        self.train_unlabeled_X = train_data
        self.train_unlabeled_Y = pd.DataFrame(
            data=train_data["label"], columns=["label"], index=train_data.index
        )
        del self.train_unlabeled_X["label"]

        """ 
        1. get start_set from X_labeled
        2. if X_unlabeled is None : experiment!
            2.1 if X_test: rest von X_labeled wird X_train_unlabeled
            2.2 if X_test is none: split rest von X_labeled in X_train_unlabeled und X_test
           else (kein experiment):
           X_unlabeled wird X_unlabeled, rest von X_labeled wird X_train_unlabeled_
        
        """
        # separate X_labeled into start_set and labeled _rest
        if START_SET_SIZE >= len(self.train_labeled_Y):

            # check if the minimum amount of labeled data is present in the start set size
            labels_not_in_start_set = set(range(0, len(self.label_encoder.classes_)))
            all_label_in_start_set = False

            for Y in self.train_labeled_Y:
                if Y in labels_not_in_start_set:
                    labels_not_in_start_set.remove(Y)
                if len(labels_not_in_start_set) == 0:
                    all_label_in_start_set = True
                    break

            if not all_label_in_start_set:
                #  if len(self.train_labeled_data) == 0:
                #      print("Please specify at least one labeled example of each class")
                #      exit(-1)

                # move more data here from the classes not present
                for label in labels_not_in_start_set:
                    # select a random sample which is NOT yet labeled
                    selected_index = (
                        self.train_unlabeled_Y[self.train_unlabeled_Y["label"] == label]
                        .iloc[0:1]
                        .index
                    )
                    #  print(
                    #      pd.DataFrame(data=label, columns="label", index=selected_index),
                    #  )
                    self._append_samples_to_labeled(
                        selected_index,
                        pd.DataFrame(
                            data=label, columns=["label"], index=selected_index
                        ),
                    )

        len_train_labeled = len(self.train_labeled_Y)
        len_train_unlabeled = len(self.train_unlabeled_Y)
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

        log_it("Loaded " + DATASET_NAME)

    def _load_dwtc(self, DATASETS_PATH):
        df = pd.read_csv(DATASETS_PATH + "/dwtc/aft.csv", index_col="id")

        # shuffle df
        df = df.sample(frac=1, random_state=self.RANDOM_SEED).reset_index(drop=True)

        self.feature_columns = df.columns.to_list()
        self.feature_columns.remove("CLASS")
        df.rename({"CLASS": "label"}, axis="columns", inplace=True)

        self.amount_of_training_samples = int(len(df) * 0.5)
        return df

    def _load_synthetic(self, kwargs):
        self.X_data, self.Y_temp = make_classification(**kwargs)
        df = pd.DataFrame(self.X_data)

        # replace labels with strings
        Y_temp = Y_temp.astype("str")
        for i in range(0, kwargs["n_classes"]):
            np.place(Y_temp, Y_temp == str(i), chr(65 + i))

        # feature_columns fehlt

        df["label"] = Y_temp
        return df

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
        df["label"] = labels[0]
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

        return df

    def _append_samples_to_labeled(self, query_indices, Y_query):
        self.train_labeled_X = self.train_labeled_X.append(
            self.train_unlabeled_X.loc[query_indices]
        )
        self.train_labeled_Y = self.train_labeled_Y.append(Y_query)

    def label_samples(self, query_indices, Y_query):
        # remove from train_unlabeled_data and add to train_labeled_data
        self._append_samples_to_labeled(query_indices, Y_query)
        self.train_unlabeled_data = self.train_unlabeled_data.drop(query_indices)

    def get_true_label(self, query_indice):
        return self.train_unlabeled_data.loc[query_indice, "true_label"]
