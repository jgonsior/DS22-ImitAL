from collections import Counter, defaultdict
from typing import Any, DefaultDict, Optional, Union, List
from tabulate import tabulate
import random

# adapted version of https://github.com/DeborahHier/CuckooHash
class Node:
    def __init__(self, key: int, value: Any):
        self.key = key
        self.value = value
        self.data = value

    def __str__(self):
        return "(" + str(self.key) + ", " + str(self.value) + ")"


for rs in range(0, 10):
    # rs = 0
    random.seed(rs)

    class CuckooHashing:
        static_rehashing_counter: int
        maxLoop = 50

        def __init__(self, max_hash_size: int) -> None:
            self.max_hash_size = max_hash_size
            self.bucket_size = 2 * max_hash_size
            self.current_size = 0
            self.bucket_1: List[Optional[Node]] = [None] * self.bucket_size
            self.bucket_2: List[Optional[Node]] = [None] * self.bucket_size

            self.hash_parameter1 = 3
            self.hash_parameter2 = 2

            # solution:
            self.rehashing_counter = 0
            self.evicted_nodes_counter: DefaultDict = defaultdict(int)

        # h1(x) = x mod 17 + x mod 3
        def hash_func_1(self, key: int) -> int:
            return ((key % 17) + (key % self.hash_parameter1)) % self.bucket_size

        # h2(x) = x mod 19 + x mod 2
        def hash_func_2(self, key: int) -> int:
            return ((key % 19) + (key % self.hash_parameter2)) % self.bucket_size

        def insert(self, key: int, value: Any, iteration_level=0) -> bool:

            # we do not have to insert duplicates
            if self.retrieve(key) != None:
                return False

            node: Optional[Node] = Node(key, value)

            pos1 = self.hash_func_1(key)
            pos2 = self.hash_func_2(key)

            # start loop with first bucket
            current_position = pos1
            current_bucket = self.bucket_1

            for _ in range(self.maxLoop):
                # easy: bucket is empty at h1/h2, just insert it
                if current_bucket[current_position] == None:
                    current_bucket[current_position] = node
                    return True

                # apparently the position is occupied -> try to move the value which is already in place at this position to the other bucket
                evicted_node = current_bucket[current_position]

                # lösung
                self.evicted_nodes_counter[evicted_node.value] += 1

                current_bucket[
                    current_position
                ] = node  # store our node at the position we want it to be
                node = (
                    evicted_node  # try to insert the evicted node at the other position
                )

                # in case we've checked bucket 1 before, we check now bucket 2 in the next iterationa
                if current_position == pos1:
                    pos1 = self.hash_func_1(node.key)
                    pos2 = self.hash_func_2(node.key)
                    current_position = pos2
                    current_bucket = self.bucket_2
                else:
                    pos1 = self.hash_func_1(node.key)
                    pos2 = self.hash_func_2(node.key)
                    current_position = pos1
                    current_bucket = self.bucket_1

            # if we made it here we have to rehash everything an then just try to insert the evicted node again
            self.rehash_buckets(iteration_level=iteration_level)

            self.insert(node.key, node.value, iteration_level=iteration_level)
            return True

        def rehash_buckets(self, iteration_level=0) -> None:
            # lösung
            print("\t rehash ", iteration_level)
            self.rehashing_counter += 1
            CuckooHashing.static_rehashing_counter += 1

            # create new hash functions
            self.hash_parameter1 += random.randint(1, 100)
            self.hash_parameter2 += random.randint(1, 100)

            tmp = CuckooHashing(self.max_hash_size)

            for i in range(self.bucket_size):
                n1 = self.bucket_1[i]
                n2 = self.bucket_2[i]

                if n1 != None:
                    tmp.insert(n1.key, n1.value, iteration_level=iteration_level + 1)

                if n2 != None:
                    tmp.insert(n2.key, n2.value, iteration_level=iteration_level + 1)

            self.bucket_1 = tmp.bucket_1
            self.bucket_2 = tmp.bucket_2

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

        def print_buckets(self) -> None:
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

    test_hash_map = CuckooHashing(8)
    CuckooHashing.static_rehashing_counter = 0

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
        print("Insert " + value)
        test_hash_map.insert(i, value)

    test_hash_map.print_buckets()

    print("Wie oft wurde sondiert? (ohne Rekursion)", test_hash_map.rehashing_counter)
    print(
        "Wie oft wurde sondiert? (mit Rekursion)",
        test_hash_map.static_rehashing_counter,
    )

    print(
        "Welches Element wird am öftesten evicted?",
        Counter(test_hash_map.evicted_nodes_counter).most_common(100),
    )
    # exit(-1)
