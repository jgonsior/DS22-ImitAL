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


# for seed in range(0, 9781):
for seed in range(27, 28):
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

        def delete(self, key) -> None:
            h1 = self.hash_func_1(key)
            h2 = self.hash_func_2(key)

            # check both positions
            value_at_h1 = self.bucket_1[h1]
            value_at_h2 = self.bucket_2[h2]

            if value_at_h1 != None and value_at_h1.key == key:
                self.bucket_1[h1] = None
            elif value_at_h2 != None and value_at_h2.key == key:
                self.bucket_2[h2] = None

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

            for i, lang in enumerate(values_to_store):
                print(
                    "{:>3} - {:>10} {:>2} {:>2}".format(
                        i,
                        lang,
                        language_hash_tables.hash_func_1(i),
                        language_hash_tables.hash_func_2(i),
                    )
                )

            # php ruby javascript
            # delete stuff
            language_hash_tables.delete(8)
            language_hash_tables.delete(21)
            language_hash_tables.delete(27)
            language_hash_tables.print_hash_tables()

            language_hash_tables.insert(8, "Erlang")
            language_hash_tables.insert(27, "TypeScript")
            language_hash_tables.insert(21, "Python")
            language_hash_tables.print_hash_tables()
            # test that our hashing algorithm worked

        # verständnisfrage:
        # a) (einfachster test, ganz klassisch)
        for i, programming_language in enumerate(values_to_store):
            assert language_hash_tables.retrieve(i) == programming_language

        # b)
        values_bucket_1 = [
            n.value for n in language_hash_tables.bucket_1 if n is not None
        ]
        values_bucket_2 = [
            n.value for n in language_hash_tables.bucket_2 if n is not None
        ]
        for programming_language in values_to_store:
            assert (
                programming_language in values_bucket_1
                or programming_language in values_bucket_2
            )

        # c) -> hash_1 und hash_2 sind unten vertauscht, und da Ada zufälligerweise für hash1 und hash2 dieselben werte enthält funktioniert der code trotzdem
        key_to_test = 0
        value_to_test = "Ada"
        hash_1 = language_hash_tables.hash_func_1(key_to_test)
        hash_2 = language_hash_tables.hash_func_2(key_to_test)
        assert (
            language_hash_tables.bucket_1[hash_2].value == value_to_test
            or language_hash_tables.bucket_2[hash_1].value == value_to_test
        )

        # d)
        for i, node in enumerate(language_hash_tables.bucket_1):
            if node is None:
                pass
            else:
                assert i == language_hash_tables.hash_func_1(node.key)

        for i, node in enumerate(language_hash_tables.bucket_2):
            if node is None:
                pass
            else:
                assert i == language_hash_tables.hash_func_2(node.key)
