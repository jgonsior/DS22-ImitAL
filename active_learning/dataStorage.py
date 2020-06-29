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

        self.df.assign(recommendation=np.NaN)

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

            if not all_label_in_start_set:
                #  if len(self.get_df("train")['label'] != None) == 0:
                #      print("Please specify at least one labeled example of each class")
                #      exit(-1)

                # move more data here from the classes not present
                for label in labels_not_in_start_set:
                    # select a random sample of this sample which is NOT yet labeled

                    selected_index = (
                        self.df.loc[
                            (self.df["true_label"] == label)
                            & (self.df["label"] == -1)
                            & (self.df["dataset"] == "train"),
                            #  & (
                            #      self.df[0]
                            #  ),  # <-- in die Maske mit rein, dass ich nur das erste will :)
                            #  "label",
                        ]
                        .iloc[0:1]
                        .index
                    )
                    self.df.loc[selected_index, "label"] = label

        len_train_labeled = len(self.get_df("train", labeled=True))
        len_train_unlabeled = len(self.get_df("train", unlabeled=True))
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

    def get_X(self, *args, **kwargs):
        df = self.get_df(*args, **kwargs)

        return df[self.feature_columns]

    def get_Y(self, *args, **kwargs):
        df = self.get_df(*args, **kwargs)
        return df["label"]

    def get_df(self, dataset=None, mask=None, labeled=False, unlabeled=False):
        if dataset is None:
            df = self.df
        else:
            df = self.df[self.df["dataset"] == dataset]

        if labeled:
            df = df.loc[df["label"] != -1]

        if unlabeled:
            df = df.loc[df["label"] == -1]

        if mask != None:
            selection = True
            for k, v in mask.items():
                selection &= df[k] == v
            df = df.loc[selection]

        return df

    def label_samples(self, query_indices, Y_query):
        self.df.loc[query_indices]["label"] = Y_query
