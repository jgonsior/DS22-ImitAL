import datetime
import random
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from tqdm import tqdm
import ml_datagen
import multiprocessing as mp


def exp_1_random_ml_datagen(random_state: int, path: str = None):
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = random.randrange(100, 5001, 50)
    n_features = random.randrange(2, 61, 1)
    if int(n_features / 3) < 1:
        n_redundant = 0
        n_repeated = 0
    else:
        n_redundant = random.randrange(0, int(n_features / 3), 1)
        n_repeated = random.randrange(0, int(n_features / 3), 1)
    n_informative = n_features - n_redundant - n_repeated
    n_classes = random.randrange(2, 21, 1)
    n_clusters_per_class = random.randrange(1, 11, 1)
    mov_vectors = np.random.rand(20, n_informative)

    shape = "cubes"

    dic = {
        random_state: {
            "shape": shape,
            "n_samples": n_samples,
            "__n_features": n_features,
            "m_rel": n_informative,
            "m_red": n_redundant,
            "m_irr": n_repeated,
            "n_classes": n_classes,
            "n_clusters_per_class": n_clusters_per_class,
            "mov_vectors": mov_vectors,
            "random_state": rs,
        }
    }

    dataset, labels, _ = ml_datagen.generate(
        shapes=shape,
        m_rel=n_informative,
        m_irr=n_repeated,
        m_red=n_redundant,
        n_classes=n_classes,
        n_samples=n_samples,
        singlelabel=True,
        random_state=rs,
        n_clusters_per_class=n_clusters_per_class,
        mov_vectors=mov_vectors,
    )

    return dataset, labels, dic


def exp_2_random_ml_datagen(random_state: int, path: str = None):
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = random.randrange(100, 5001, 50)
    n_features = random.randrange(2, 61, 1)
    if int(n_features / 3) < 1:
        n_redundant = 0
        n_repeated = 0
    else:
        n_redundant = random.randrange(0, int(n_features / 3), 1)
        n_repeated = random.randrange(0, int(n_features / 3), 1)
    n_informative = n_features - n_redundant - n_repeated
    n_classes = random.randrange(2, 21, 1)
    n_clusters_per_class = random.randrange(1, 11, 1)
    mov_vectors = np.random.rand(20, n_informative)

    shape = "mix"

    dic = {
        random_state: {
            "shape": shape,
            "n_samples": n_samples,
            "__n_features": n_features,
            "m_rel": n_informative,
            "m_red": n_redundant,
            "m_irr": n_repeated,
            "n_classes": n_classes,
            "n_clusters_per_class": n_clusters_per_class,
            "mov_vectors": mov_vectors,
            "random_state": rs,
        }
    }

    dataset, labels, _ = ml_datagen.generate(
        shapes=shape,
        m_rel=n_informative,
        m_irr=n_repeated,
        m_red=n_redundant,
        n_classes=n_classes,
        n_samples=n_samples,
        singlelabel=True,
        random_state=rs,
        n_clusters_per_class=n_clusters_per_class,
        mov_vectors=mov_vectors,
    )

    return dataset, labels, dic


def exp_3_random_ml_datagen(random_state: int, path: str = None):
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = random.randrange(100, 5001, 50)
    n_features = random.randrange(2, 61, 1)
    if int(n_features / 3) < 1:
        n_redundant = 0
        n_repeated = 0
    else:
        n_redundant = random.randrange(0, int(n_features / 3), 1)
        n_repeated = random.randrange(0, int(n_features / 3), 1)
    n_informative = n_features - n_redundant - n_repeated
    n_classes = random.randrange(2, 21, 1)
    n_clusters_per_class = random.randrange(1, 11, 1)
    mov_vectors = np.random.rand(20, n_informative)

    shape = random.choice(["spheres", "cubes", "moons", "mix"])

    dic = {
        random_state: {
            "shape": shape,
            "n_samples": n_samples,
            "__n_features": n_features,
            "m_rel": n_informative,
            "m_red": n_redundant,
            "m_irr": n_repeated,
            "n_classes": n_classes,
            "n_clusters_per_class": n_clusters_per_class,
            "mov_vectors": mov_vectors,
            "random_state": rs,
        }
    }

    dataset, labels, _ = ml_datagen.generate(
        shapes=shape,
        m_rel=n_informative,
        m_irr=n_repeated,
        m_red=n_redundant,
        n_classes=n_classes,
        n_samples=n_samples,
        singlelabel=True,
        random_state=rs,
        n_clusters_per_class=n_clusters_per_class,
        mov_vectors=mov_vectors,
    )

    return dataset, labels, dic


def exp_4_random_ml_datagen(random_state: int, path: str = None):
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = random.randrange(100, 5001, 50)
    n_features = random.randrange(2, 61, 1)
    if int(n_features / 3) < 1:
        n_redundant = 0
        n_repeated = 0
    else:
        n_redundant = random.randrange(0, int(n_features / 3), 1)
        n_repeated = random.randrange(0, int(n_features / 3), 1)
    n_informative = n_features - n_redundant - n_repeated
    n_classes = random.randrange(2, 21, 1)
    n_clusters_per_class = random.randrange(1, 11, 1)
    mov_vectors = np.random.rand(20, n_informative)

    shape = [("cubes", 0.6), ("spheres", 0.2), ("moons", 0.2)]

    dic = {
        random_state: {
            "shape": shape,
            "n_samples": n_samples,
            "__n_features": n_features,
            "m_rel": n_informative,
            "m_red": n_redundant,
            "m_irr": n_repeated,
            "n_classes": n_classes,
            "n_clusters_per_class": n_clusters_per_class,
            "mov_vectors": mov_vectors,
            "random_state": rs,
        }
    }

    dataset, labels, _ = ml_datagen.generate(
        shapes=shape,
        m_rel=n_informative,
        m_irr=n_repeated,
        m_red=n_redundant,
        n_classes=n_classes,
        n_samples=n_samples,
        singlelabel=True,
        random_state=rs,
        n_clusters_per_class=n_clusters_per_class,
        mov_vectors=mov_vectors,
    )

    return dataset, labels, dic


def exp_5_random_ml_datagen(random_state: int, path: str = None):
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = random.randrange(100, 5001, 50)
    n_features = random.randrange(2, 61, 1)
    if int(n_features / 3) < 1:
        n_redundant = 0
        n_repeated = 0
    else:
        n_redundant = random.randrange(0, int(n_features / 3), 1)
        n_repeated = random.randrange(0, int(n_features / 3), 1)
    n_informative = n_features - n_redundant - n_repeated
    n_classes = random.randrange(2, 21, 1)
    n_clusters_per_class = random.randrange(1, 11, 1)
    n_random_points = random.random() * 0.2
    mov_vectors = np.random.rand(20, n_informative)

    shape = "mix"

    dic = {
        random_state: {
            "shape": shape,
            "n_samples": n_samples,
            "__n_features": n_features,
            "m_rel": n_informative,
            "m_red": n_redundant,
            "m_irr": n_repeated,
            "n_classes": n_classes,
            "n_clusters_per_class": n_clusters_per_class,
            "mov_vectors": mov_vectors,
            "random_state": rs,
        }
    }

    dataset, labels, _ = ml_datagen.generate(
        shapes=shape,
        m_rel=n_informative,
        m_irr=n_repeated,
        m_red=n_redundant,
        n_classes=n_classes,
        n_samples=n_samples,
        singlelabel=True,
        random_state=rs,
        n_clusters_per_class=n_clusters_per_class,
        random_points=n_random_points,
        mov_vectors=mov_vectors,
    )

    return dataset, labels, dic


def exp_6_random_ml_datagen(random_state: int, path: str = None):
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = int(np.random.normal(1500, 600, 1)[0])
    n_samples = n_samples if n_samples > 100 else 100
    n_informative = int(np.random.normal(20, 10, 1)[0])
    n_informative = n_informative if n_informative > 3 else 3
    n_redundant = random.randrange(0, 11, 1)
    n_repeated = random.randrange(0, 11, 1)
    n_features = n_informative + n_redundant + n_repeated
    n_classes = random.randrange(2, 15, 1)
    n_clusters_per_class = random.randrange(1, 6, 1)

    shape = [("cubes", 0.6), ("spheres", 0.2), ("moons", 0.2)]

    dic = {
        random_state: {
            "shape": shape,
            "n_samples": n_samples,
            "__n_features": n_features,
            "m_rel": n_informative,
            "m_red": n_redundant,
            "m_irr": n_repeated,
            "n_classes": n_classes,
            "n_clusters_per_class": n_clusters_per_class,
            "random_state": rs,
        }
    }

    dataset, labels, _ = ml_datagen.generate(
        shapes=shape,
        m_rel=n_informative,
        m_irr=n_repeated,
        m_red=n_redundant,
        n_classes=n_classes,
        n_samples=n_samples,
        singlelabel=True,
        random_state=rs,
        n_clusters_per_class=n_clusters_per_class,
        mov_vectors="random",
    )

    return dataset, labels, dic


def exp_7_random_ml_datagen(random_state: int, path: str = None):
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = int(np.random.normal(1500, 600, 1)[0])
    n_samples = n_samples if n_samples > 100 else 100
    n_informative = int(np.random.normal(20, 10, 1)[0])
    n_informative = n_informative if n_informative > 3 else 3
    n_redundant = random.randrange(0, 11, 1)
    n_repeated = random.randrange(0, 11, 1)
    n_features = n_informative + n_redundant + n_repeated
    n_classes = random.randrange(2, 15, 1)
    n_clusters_per_class = random.randrange(1, 6, 1)

    n_categorical_variables = random.randrange(0, 11, 1)
    n_categorical_variables = (
        n_categorical_variables
        if n_categorical_variables <= n_informative
        else n_informative
    )
    categorical_variables = []
    for i in range(n_categorical_variables):
        v = int(np.random.normal(4, 6, 1)[0])
        v = v if v > 1 else 2
        categorical_variables.append(v)

    shape = "mix"

    dic = {
        random_state: {
            "shape": shape,
            "n_samples": n_samples,
            "__n_features": n_features,
            "m_rel": n_informative,
            "m_red": n_redundant,
            "m_irr": n_repeated,
            "n_classes": n_classes,
            "n_clusters_per_class": n_clusters_per_class,
            "random_state": rs,
        }
    }

    dataset, labels, _ = ml_datagen.generate(
        shapes=shape,
        m_rel=n_informative,
        m_irr=n_repeated,
        m_red=n_redundant,
        n_classes=n_classes,
        n_samples=n_samples,
        singlelabel=True,
        random_state=rs,
        n_clusters_per_class=n_clusters_per_class,
        mov_vectors="random",
        categorical_variabels=categorical_variables,
    )

    return dataset, labels, dic


def exp_8_random_ml_datagen(random_state: int, path: str = None):
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = int(np.random.normal(1500, 600, 1)[0])
    n_samples = n_samples if n_samples > 100 else 100
    n_informative = int(np.random.normal(20, 10, 1)[0])
    n_informative = n_informative if n_informative > 3 else 3
    n_redundant = random.randrange(0, 11, 1)
    n_repeated = random.randrange(0, 11, 1)
    n_features = n_informative + n_redundant + n_repeated
    n_classes = random.randrange(2, 15, 1)
    n_clusters_per_class = random.randrange(1, 6, 1)

    n_categorical_variables = random.randrange(0, 11, 1)
    n_categorical_variables = (
        n_categorical_variables
        if n_categorical_variables <= n_informative
        else n_informative
    )
    categorical_variables = []
    for i in range(n_categorical_variables):
        v = int(np.random.normal(4, 6, 1)[0])
        v = v if v > 1 else 2
        categorical_variables.append(v)

    shape = "mix"
    dic = {
        random_state: {
            "shape": shape,
            "n_samples": n_samples,
            "__n_features": n_features,
            "m_rel": n_informative,
            "m_red": n_redundant,
            "m_irr": n_repeated,
            "n_classes": n_classes,
            "n_clusters_per_class": n_clusters_per_class,
            "random_state": rs,
        }
    }

    if random.random() >= 0.5:
        dataset, labels, _ = ml_datagen.generate(
            shapes=shape,
            m_rel=n_informative,
            m_irr=n_repeated,
            m_red=n_redundant,
            n_classes=n_classes,
            n_samples=n_samples,
            singlelabel=True,
            random_state=rs,
            n_clusters_per_class=n_clusters_per_class,
            mov_vectors="random",
            categorical_variabels=categorical_variables,
        )
    else:
        dataset, labels = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            n_clusters_per_class=n_clusters_per_class,
            random_state=rs,
        )

    return dataset, labels, dic


def exp_9_random_ml_datagen(random_state: int, path: str = None):
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = random.randrange(100, 5001, 50)
    n_features = random.randrange(2, 61, 1)
    if int(n_features / 3) < 1:
        n_redundant = 0
        n_repeated = 0
    else:
        n_redundant = random.randrange(0, int(n_features / 3), 1)
        n_repeated = random.randrange(0, int(n_features / 3), 1)
    n_informative = n_features - n_redundant - n_repeated
    n_classes = random.randrange(2, 21, 1)
    n_clusters_per_class = random.randrange(1, 11, 1)
    n_random_points = random.random() * 0.2
    mov_vectors = np.random.rand(20, n_informative)

    n_categorical_variables = random.randrange(0, 11, 1)
    n_categorical_variables = (
        n_categorical_variables
        if n_categorical_variables <= n_informative
        else n_informative
    )
    categorical_variables = []
    for i in range(n_categorical_variables):
        v = int(np.random.normal(4, 6, 1)[0])
        v = v if v > 1 else 2
        categorical_variables.append(v)

    shape = "mix"

    dic = {
        random_state: {
            "shape": shape,
            "n_samples": n_samples,
            "__n_features": n_features,
            "m_rel": n_informative,
            "m_red": n_redundant,
            "m_irr": n_repeated,
            "n_classes": n_classes,
            "n_clusters_per_class": n_clusters_per_class,
            "mov_vectors": mov_vectors,
            "random_state": rs,
        }
    }

    dataset, labels, _ = ml_datagen.generate(
        shapes=shape,
        m_rel=n_informative,
        m_irr=n_repeated,
        m_red=n_redundant,
        n_classes=n_classes,
        n_samples=n_samples,
        singlelabel=True,
        random_state=rs,
        n_clusters_per_class=n_clusters_per_class,
        random_points=n_random_points,
        mov_vectors=mov_vectors,
        categorical_variabels=categorical_variables,
    )

    return dataset, labels, dic


def random_sklearn(random_state: int, path: str = None):
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = random.randrange(100, 5001, 50)
    n_features = random.randrange(2, 61, 1)
    if int(n_features / 3) < 1:
        n_redundant = 0
        n_repeated = 0
    else:
        n_redundant = random.randrange(0, int(n_features / 3), 1)
        n_repeated = random.randrange(0, int(n_features / 3), 1)
    n_informative = n_features - n_redundant - n_repeated
    n_classes = random.randrange(2, 21, 1)
    n_clusters_per_class = random.randrange(1, 11, 1)

    while n_classes * n_clusters_per_class > 2 ** n_informative:
        if n_clusters_per_class > 1:
            n_clusters_per_class -= 1
        elif n_classes > 2:
            n_classes -= 1
        else:
            n_informative += 1

    dic = {
        random_state: {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_informative": n_informative,
            "n_redundant": n_redundant,
            "n_repeated": n_repeated,
            "n_classes": n_classes,
            "n_clusters_per_class": n_clusters_per_class,
            "random_state": rs,
        }
    }

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        random_state=rs,
    )
    return X, y, dic


def example(number: int):
    if number == 0:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for i in tqdm(range(1000)):
            dataset, labels = random_sklearn(i, ts + "/ds.txt")
    elif number == 1:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for i in tqdm(range(1000)):
            dataset, labels = exp_1_random_ml_datagen(i, ts + "/ds.txt")
    elif number == 2:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for i in tqdm(range(1000)):
            dataset, labels = exp_2_random_ml_datagen(i, ts + "/ds.txt")
    elif number == 3:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for i in tqdm(range(1000)):
            dataset, labels = exp_3_random_ml_datagen(i, ts + "/ds.txt")
    elif number == 4:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for i in tqdm(range(1000)):
            dataset, labels = exp_4_random_ml_datagen(i, ts + "/ds.txt")
    elif number == 5:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for i in tqdm(range(1000)):
            dataset, labels = exp_5_random_ml_datagen(i, ts + "/ds.txt")
    elif number == 6:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for i in tqdm(range(1000)):
            dataset, labels = exp_6_random_ml_datagen(i, ts + "/ds.txt")
    elif number == 7:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for i in tqdm(range(1000)):
            dataset, labels = exp_7_random_ml_datagen(i, ts + "/ds.txt")
    elif number == 8:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for i in tqdm(range(1000)):
            dataset, labels = exp_8_random_ml_datagen(i, ts + "/ds.txt")


if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count() - 4)
    pool.map(example, range(6, 9))

    # ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #
    # for i in tqdm(range(1000)):
    #     dataset, labels = exp_2_random_ml_datagen(i, ts + "/ds.txt")
