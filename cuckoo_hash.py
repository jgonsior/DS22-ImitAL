from collections import Counter, defaultdict
import enum
from math import floor
from typing import Any, DefaultDict, Optional, Union, List
from tabulate import tabulate
import random


class Node:
    def __init__(self, key: int, value: Any):
        self.key = key
        self.value = value

    def __str__(self):
        return "(" + str(self.key) + ", " + str(self.value) + ")"


for seed in range(0, 9781):
    # print(seed)
    random.seed(seed)

    class CuckooHashing:
        maxLoop = 50

        def __init__(self, bucket_size: int) -> None:
            self.bucket_size = bucket_size
            self.current_size = 0
            self.bucket_1: List[Optional[Node]] = [None] * self.bucket_size
            self.bucket_2: List[Optional[Node]] = [None] * self.bucket_size

            self.hash_func_params = [
                random.randint(2, self.bucket_size) for _ in range(0, 4)
            ]

            # solution:
            self.rehashing_counter = 0
            self.evicted_nodes_counter: DefaultDict = defaultdict(int)

        # h1(x) = x mod 17 + x mod 3
        def hash_func_1(self, key: int) -> int:
            return (
                (key % self.hash_func_params[0]) + (key % self.hash_func_params[1])
            ) % self.bucket_size

        # h2(x) = x mod 19 + x mod 2
        def hash_func_2(self, key: int) -> int:
            return (
                (key % self.hash_func_params[2]) - (key % self.hash_func_params[3])
            ) % self.bucket_size

        def insert(self, key: int, value: Any, iteration_level=0) -> bool:
            # we do not have to insert duplicates
            if self.retrieve(key) != None:
                return False

            for _ in range(self.maxLoop):
                if self.bucket_1[self.hash_func_1(key)] == None:
                    self.bucket_1[self.hash_func_1(key)] = Node(key, value)
                    return True

                # cuckoo
                tmp: Node = self.bucket_1[self.hash_func_1(key)]
                self.bucket_1[self.hash_func_1(key)] = Node(key, value)
                key = tmp.key
                value = tmp.value
                self.evicted_nodes_counter[value] += 1

                if self.bucket_2[self.hash_func_2(key)] == None:
                    self.bucket_2[self.hash_func_2(key)] = Node(key, value)
                    return True

                # cuckoo
                tmp: Node = self.bucket_2[self.hash_func_2(key)]
                self.bucket_2[self.hash_func_2(key)] = Node(key, value)
                key = tmp.key
                value = tmp.value
                self.evicted_nodes_counter[value] += 1

            self.rehash_tables(iteration_level=iteration_level)
            self.insert(key, value, iteration_level=iteration_level)
            return True

        def rehash_tables(self, iteration_level=0) -> None:
            # self.bucket_size *= 2
            # lösung
            # print("\t rehash ", iteration_level)
            self.rehashing_counter += 1

            # create new hash functions
            self.hash_func_params = [
                random.randint(2, self.bucket_size) for i in range(0, 4)
            ]

            # collect already existing nodes
            nodes: List[Node] = []
            for i in range(0, self.bucket_size):
                if self.bucket_1[i] is not None:
                    nodes.append(self.bucket_1[i])

                if self.bucket_2[i] is not None:
                    nodes.append(self.bucket_2[i])

            self.bucket_1 = [None] * self.bucket_size
            self.bucket_2 = [None] * self.bucket_size

            for node in nodes:
                self.insert(node.key, node.value, iteration_level=iteration_level + 1)

        def retrieve(self, key: int) -> Union[None, Any]:
            h1 = self.hash_func_1(key)
            h2 = self.hash_func_2(key)

            # check both positions
            value_at_h1 = self.bucket_1[h1]
            value_at_h2 = self.bucket_2[h2]

            if value_at_h1 != None and value_at_h1.key == key:
                return value_at_h1.value
            elif value_at_h2 != None and value_at_h2.key == key:
                return value_at_h2.value
            else:
                return None

        def print_hash_tables(self) -> None:
            # for i, (b1, b2) in enumerate(zip(self.bucket_1, self.bucket_2)):
            #    print("{} {} {} ".format(i, b1, b2))
            print(
                tabulate(
                    [
                        ["Pos " + str(i) for i in range(0, self.bucket_size)],
                        self.bucket_1,
                        self.bucket_2,
                    ],
                    headers="firstrow",
                    tablefmt="grid",
                )
            )

    language_hash_tables = CuckooHashing(16)

    values_to_store = [
        "Ada",
        "Basic",
        "C",
        "C#",
        "C++",
        "D",
        "Eiffel",
        "F#",
        "Erlang",
        "Fortran",
        "Go",
        "Haskell",
        "Java",
        "Javascript",
        "Kotlin",
        "Lisp",
        "MATLAB",
        "Pascal",
        "Perl",
        "PHP",
        "Prolog",
        "Python",
        "Ruby",
        "Scala",
        "Smalltalk",
        "SQL",
        "Swift",
        "TypeScript",
    ]

    for i, value in enumerate(values_to_store):
        # lösung:
        # print("Insert " + value)
        language_hash_tables.insert(i, value)
        # language_hash_tables.print_hash_tables()

    if sum(x is not None for x in language_hash_tables.bucket_2) > 5:

        if language_hash_tables.rehashing_counter > 800:
            print(seed)  # print(seed)
            # print(language_hash_tables.bucket_2)

            language_hash_tables.print_hash_tables()

            print(
                "Wie oft wurden die hash funktionen ausgetauscht?",
                language_hash_tables.rehashing_counter,
            )

            print(
                "Welches Element wird am öftesten evicted?",
                Counter(language_hash_tables.evicted_nodes_counter).most_common(100),
            )

            # test that our hashing algorithm worked

            for i, programming_language in enumerate(values_to_store):
                assert language_hash_tables.retrieve(i) == programming_language


            irgendeine verständnisfrage a la "wofür benötigt man den hash überhaupt?"
            --> welcher codeschnippsel kann verwendet werden, um zu überprüfen, dass die hashtabelle funktioniert?


            ODER: implementiere eine einfache Löschfunktion -> lösche dann nach dem einfügen von den ersten 10 element 3 und 4 -> welches Element steht am Ende im bucket 2 an stelle 5??
            --> dabei kann ja nicht soo viel anders gemacht werden (beim löschen, oder???!!!!)
            (den code sollen sie dann bitteschön selber schreiben)
