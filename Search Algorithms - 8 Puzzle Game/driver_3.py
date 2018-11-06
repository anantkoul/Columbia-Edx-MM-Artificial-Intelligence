import queue as Q
import time as t
import resource
import sys
import os
import math
import psutil

#### SKELETON CODE ####

goal_state = (0,1,2,3,4,5,6,7,8)

## The Class that Represents the Puzzle

class PuzzleState(object):

    """docstring for PuzzleState"""

    def __init__(self, config, n, parent=None, action="Initial", cost=0):

        if n*n != len(config) or n < 2:

            raise Exception("the length of config is not correct!")

        self.n = n

        self.cost = cost


        self.parent = parent

        self.action = action

        self.dimension = n

        self.config = config

        self.children = []

        for i, item in enumerate(self.config):

            if item == 0:

                self.blank_row = int(i / self.n)

                self.blank_col = i % self.n

                break

    def display(self):

        for i in range(self.n):

            line = []

            offset = i * self.n

            for j in range(self.n):

                line.append(self.config[offset + j])

            print (line)

    def move_left(self):

        if self.blank_col == 0:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index - 1

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)

    def move_right(self):

        if self.blank_col == self.n - 1:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index + 1

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)

    def move_up(self):

        if self.blank_row == 0:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index - self.n

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):

        if self.blank_row == self.n - 1:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index + self.n

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def expand(self):

        """expand the node"""

        # add child nodes in order of UDLR

        if len(self.children) == 0:

            up_child = self.move_up()

            if up_child is not None:

                self.children.append(up_child)

            down_child = self.move_down()

            if down_child is not None:

                self.children.append(down_child)

            left_child = self.move_left()

            if left_child is not None:

                self.children.append(left_child)

            right_child = self.move_right()

            if right_child is not None:

                self.children.append(right_child)

        return self.children



# Function that Writes to output.txt

### Students need to change the method to have the corresponding parameters

def writeOutput(path_to_goal, cost_of_path, nodes_expanded, search_depth, max_search_depth, running_time, max_ram_usage):
    with open("output.txt", "w") as file:
        file.write("path_to_goal: {}\n".format(path_to_goal))
        file.write("cost_of_path: {}\n".format(cost_of_path))
        file.write("nodes_expanded: {}\n".format(nodes_expanded))
        file.write("search_depth: {}\n".format(search_depth))
        file.write("max_search_depth: {}\n".format(max_search_depth))
        file.write("running_time: {}\n".format(running_time))
        file.write("max_ram_usage: {}\n".format(max_ram_usage))

        #print ("cost_of_path: {}\n".format(cost_of_path))
        #print ("nodes_expanded: {}\n".format(nodes_expanded))
        #print ("search_depth: {}\n".format(search_depth))
        #print ("max_search_depth: {}\n".format(max_search_depth))
        #print ("running_time: {}\n".format(running_time))
        #print ("max_ram_usage: {}\n".format(max_ram_usage))

def bfs_search(initial_state):

    """BFS search"""

    start_time = Mem_time_cal()

    frontier = Q.Queue()
    out = set()
    node = 0
    depth = 0
    out.add(initial_state.config)

    while not goalTest(initial_state.config):

        curr_state = initial_state.expand()
        if curr_state[0].cost > depth:
            depth = curr_state[0].cost

        for index in range(len(curr_state)):
            temp = curr_state[index].config
            if temp not in out:
                out.add(temp)
                frontier.put(curr_state[index])

        initial_state  = frontier.get()
        node = node + 1

    cost_of_path = initial_state.cost
    search_depth = initial_state.cost
    path_to_goal = [initial_state.action]
    state = initial_state.parent

    while initial_state.action != "Initial":
        path_to_goal.append(initial_state.action)
        initial_state = initial_state.parent

    end_time = Mem_time_cal()
    writeOutput(path_to_goal[::-1], cost_of_path, node, search_depth, depth, end_time[0]- start_time[0], end_time[1]-start_time[1])

def dfs_search(initial_state):

    """DFS search"""

    start_time = Mem_time_cal()

    frontier = []
    out = set()
    node = 0
    depth = 0
    out.add(initial_state.config)

    while not goalTest(initial_state.config):

        curr_state = initial_state.expand()[::-1]

        for index in range(len(curr_state)):
            temp = curr_state[index].config
            if temp not in out:
                out.add(temp)
                frontier.append(curr_state[index])

        initial_state  = frontier.pop()
        node = node + 1

        if initial_state.cost > depth:
            depth = initial_state.cost

    cost_of_path = initial_state.cost
    search_depth = initial_state.cost
    path_to_goal = [initial_state.action]
    state = initial_state.parent

    while initial_state.action != "Initial":
        path_to_goal.append(initial_state.action)
        initial_state = initial_state.parent

    end_time = Mem_time_cal()
    writeOutput(path_to_goal[::-1], cost_of_path, node, search_depth, depth, end_time[0]- start_time[0], end_time[1]-start_time[1])


def A_star_search(initial_state):

    """A * search"""

    start_time = Mem_time_cal()

    frontier = {}
    out = set()
    node = 0
    depth = 0
    out.add(initial_state.config)

    while not goalTest(initial_state.config):


        curr_state = initial_state.expand()[::-1]

        if curr_state[0].cost > depth:
            depth = curr_state[0].cost

        for index in range(len(curr_state)):
            temp = curr_state[index].config
            if temp not in out:
                n = 8
                p = 0
                out.add(temp)
                p = calculate_manhattan_dist(index, curr_state, n, p)

                p = p + curr_state[index].cost
                if p not in frontier:
                    frontier[p] = Q.Queue()
                    frontier[p].put(curr_state[index])
                else:
                    frontier[p].put(curr_state[index])

        frontier_min = min(frontier)
        initial_state = frontier[frontier_min].get()

        if frontier[frontier_min].empty():
            frontier.pop(frontier_min)

        node = node + 1


    cost_of_path = initial_state.cost
    search_depth = initial_state.cost
    path_to_goal = [initial_state.action]
    state = initial_state.parent


    while initial_state.action != "Initial":
        path_to_goal.append(initial_state.action)
        initial_state = initial_state.parent

    end_time = Mem_time_cal()
    writeOutput(path_to_goal[::-1], cost_of_path, node, search_depth, depth, end_time[0]- start_time[0], end_time[1]-start_time[1])


def Mem_time_cal():
    running_time = t.time()
    mem_usage = psutil.Process(os.getpid())
    return [running_time, mem_usage.memory_info().rss]

def calculate_manhattan_dist(idx, curr_state, n, m):
    for z in range(n):
        m += abs(int(curr_state[idx].config[z]/3) - int(z/3)) + abs(curr_state[idx].config[z]%3 - z%3)

    return m

def goalTest(puzzle_state):
    if puzzle_state == goal_state:
        return True
    else:
        return False


# Main Function that reads in Input and Runs corresponding Algorithm


def main():

    sm = sys.argv[1].lower()

    begin_state = sys.argv[2].split(",")

    begin_state = tuple(map(int, begin_state))

    size = int(math.sqrt(len(begin_state)))

    hard_state = PuzzleState(begin_state, size)

    if sm == "bfs":

        bfs_search(hard_state)

    elif sm == "dfs":

        dfs_search(hard_state)

    elif sm == "ast":

        A_star_search(hard_state)

    else:

        print("Enter valid command arguments !")

if __name__ == '__main__':

    main()
