with open("graphs_aborted.txt") as f:
    for line in f:
        id, sols = line.split(": ")
        sols_list = [str(y) for y in sorted([int(x) for x in sols.split(",")])]
        # print(id)
        # print(sols_list)
        print(id + ":\t" + ",".join(sols_list))