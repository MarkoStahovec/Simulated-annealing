from random import uniform
from random import shuffle
from random import randrange
from math import exp
from math import sqrt
import os.path
import copy


class Monk:
    def __init__(self, fitness, complete_rakes, correct_finish, random_decisions, border_tiles):
        self.fitness = fitness
        self.complete_rakes = complete_rakes
        self.correct_finish = correct_finish
        self.random_decisions = random_decisions
        self.border_tiles = border_tiles


ROTATIONS = 4  # no. of rotations in a variable to reduce magical numbers in code


# prints out a given garden with a given headline
def print_map(matrix, msg_type):
    if str(msg_type) == "0":
        print("-----------------------------------------------")
        print("\n***************** INITIAL_MAP *****************\n")
        print("-----------------------------------------------")
    elif str(msg_type) == "1":
        print("-----------------------------------------------")
        print("\n***************** FINAL_MAP *****************\n")
        print("-----------------------------------------------")
    rows = len(matrix[0])
    cols = len(matrix)
    print("")
    for i in range(0, cols):
        for j in range(0, rows):
            print(matrix[i][j], end='\t')
        print("")
    print("")


# prints out algorithm stats after annealing is over
def print_algorithm_stats(monk, fin_garden, max_fitness, it):
    print_map(fin_garden, 1)
    print(f"\nNO. OF ITERATIONS: {it}")
    print(f"FINAL FITNESS: {monk.fitness}")
    print(f"MAX FITNESS: {max_fitness}")
    print(f"COMPLETE RAKES: {monk.complete_rakes}")
    if monk.correct_finish > 0:
        print(f"FINISHED CORRECTLY: YES")
    else:
        print(f"FINISHED CORRECTLY: NO")
    return


# loads garden from a file
def load_map(file):
    gard = []
    with open(file, "r") as f:
        for line in f:
            item_list = list(line.strip().split())
            gard.append(item_list)

    parse_map(gard)
    return gard


# parses zeroes as integers, since load_map reads numbers from a file as characters
def parse_map(arr):
    for x in range(0, len(arr)):
        for y in range(0, len(arr[0])):
            if arr[x][y].isnumeric():
                arr[x][y] = int(arr[x][y])


# generates a random map of given dimensions and desired density of rocks
def generate_map(x, y, rock_density):
    gard = []

    for i in range(0, x):
        row = []
        idx = 0
        while idx < y:
            square = uniform(0, 1)
            rock1_prob = uniform(0, 0.038 * rock_density)
            rock2_prob = uniform(0, 0.015 * rock_density)
            if rock2_prob > square:
                row.append("X")
                idx += 1
                if idx <= y - 1:
                    row.append("X")
            elif rock1_prob > square:
                row.append("X")
            elif square > rock1_prob:
                row.append(0)
            idx += 1

        gard.append(row)

    return gard


# cleans garden from the last move, when monk gets stuck inside, teleporting him back in time shamefully
def clean_garden(monk, gard, number):
    for i in range(0, len(gard)):
        for j in range(0, len(gard[0])):
            if gard[i][j] == number:
                gard[i][j] = 0
                monk.fitness -= 1

    monk.complete_rakes -= 1  # decrement monk's complete rakes since this function removes that particular move
    return


# placeholder function for setting up a map input containing multiple decision-making for user
def map_wrapper():
    g = []
    while True:  # endless loop for incorrect input handling
        board_choice = input("Load map from file? [y/n]: ")
        if board_choice == "y":

            filename = input('File name (ex. \'map1\'): ')
            filename = "maps/" + filename + ".txt"
            if os.path.isfile(filename):  # if the path to file exists
                g = load_map(filename)
                break
            else:
                continue

        elif board_choice == "n":
            x_size = int(input("Choose no. of columns: "))
            if x_size > 20:
                x_size = 20
            elif x_size < 4:
                x_size = 4

            y_size = int(input("Choose no. of rows: "))
            if y_size > 20:
                y_size = 20
            elif y_size < 4:
                y_size = 4

            rock_density = int(input("Select rock density [0-5]: "))
            if rock_density > 5:
                rock_density = 5
            elif rock_density < 0:
                rock_density = 0

            g = generate_map(x_size, y_size, rock_density)
            break

    return g


# placeholder function for different annealing settings
def annealing_settings_wrapper():
    while True:
        manual_settings = input("Set annealing settings manually? [y/n]: ")
        if manual_settings == "y":
            temp = int(input("Choose temperature [2-100]: "))
            if temp > 100:  # correct ambigiuous values
                temp = 100
            elif temp < 2:
                temp = 2

            threshold_extender = int(input("Choose threshold extender (default is 4) [1-100]: "))
            if threshold_extender > 100:  # correct ambigiuous values
                threshold_extender = 100
            elif threshold_extender < 1:
                threshold_extender = 1

            return temp, threshold_extender

        elif manual_settings == "n":  # return default values if no is chosen
            return 30, 4


# calculate and return all border tiles of possible entries with directions
def get_border_tiles(garden, dimensions):
    border_tiles = []
    for i in range(0, dimensions[0]):
        for j in range(0, dimensions[1]):
            if (i == 0 or j == 0 or (i == dimensions[0] - 1) or (j == dimensions[1] - 1)) and (garden[i][j] != "X"):
                # border_tiles.append([i, j])
                # -> 0
                # v  1
                # <- 2
                # ^  3

                if i == 0 and j == 0:  # all cases for corner entrances, where are 2 possible directions instead of 1
                    border_tiles.append([i, j, 0])
                    border_tiles.append([i, j, 1])
                elif i == 0 and (j == dimensions[1] - 1):
                    border_tiles.append([i, j, 1])
                    border_tiles.append([i, j, 2])
                elif (i == dimensions[0] - 1) and (j == dimensions[1] - 1):
                    border_tiles.append([i, j, 2])
                    border_tiles.append([i, j, 3])
                elif (i == dimensions[0] - 1) and j == 0:
                    border_tiles.append([i, j, 0])
                    border_tiles.append([i, j, 3])

                elif i == 0:
                    border_tiles.append([i, j, 1])
                elif j == 0:
                    border_tiles.append([i, j, 0])
                elif i == dimensions[0] - 1:
                    border_tiles.append([i, j, 3])
                elif j == dimensions[1] - 1:
                    border_tiles.append([i, j, 2])

    shuffle(border_tiles)  # shuffle them up before sending out to the first monk
    return border_tiles


# creates an array of random decisions for monk, when he hits a barrier
# is necessary to ensure, that decisions are same for each new monk created, so we get a strict neighbour
def get_random_decisions(garden, dimensions):
    rock_counter = 3  # init to 3 to ensure, that there are at least some decisions when running into raked tiles
    random_decisions = []
    for r in range(0, dimensions[0]):  # count all the rocks
        for c in range(0, dimensions[1]):
            if garden[r][c] == "X":
                rock_counter += 1

    for i in range(0, rock_counter):
        random_decisions.append(round(uniform(0, 1), 2))

    return random_decisions


# returns maximum possible fitness for a given garden by substracting number of rocks from dimensions of a garden
def get_max_fitness(garden, dimensions):
    max_fitness = dimensions[0] * dimensions[1]
    for i in range(0, dimensions[0]):
        for j in range(0, dimensions[1]):
            if garden[i][j] == "X":
                max_fitness -= 1

    return max_fitness


# returns new coordinates when moving forward
def get_forward_position(old_position):
    new_position = old_position.copy()
    if old_position[2] == 0:
        new_position[1] += 1
    elif old_position[2] == 1:
        new_position[0] += 1
    elif old_position[2] == 2:
        new_position[1] -= 1
    elif old_position[2] == 3:
        new_position[0] -= 1

    return new_position


# returns new coordinates when sidestep is necessary
def get_side_position(old_position, side):
    new_position = old_position.copy()
    if side == "left":
        new_position[2] = ((old_position[2] - 1) % ROTATIONS)
    elif side == "right":
        new_position[2] = ((old_position[2] + 1) % ROTATIONS)
    if new_position[2] == 0:
        new_position[1] += 1
    elif new_position[2] == 1:
        new_position[0] += 1
    elif new_position[2] == 2:
        new_position[1] -= 1
    elif new_position[2] == 3:
        new_position[0] -= 1

    return new_position


# returns boolean for a given coordinates, whether they are in bounds of a given garden
def is_in_bounds(garden, x, y):
    if (x > len(garden) - 1) or (y > len(garden[0]) - 1) or x < 0 or y < 0:
        return False
    return True


# returns boolean for a given coordinates, whether the given tile is rakeable (there is no rock nor it hasnt been raked)
def is_tile_rakeable(garden, x, y):
    if garden[x][y] != 0:
        return False
    else:
        return True


# main raking function, where monk rakes one path from a given starting position
def rake_garden(garden, position, turn, monk):
    tiles_raked = 0
    if not is_tile_rakeable(garden, position[0], position[1]):  # check to ensure entry is valid
        return turn, tiles_raked

    turn += 1  # variable that holds number of rake, which is being drawn into the garden
    while True:
        if not is_in_bounds(garden, position[0], position[1]):
            if tiles_raked == 0:  # if no tile was raked, decrement turn since it didnt do any work
                turn -= 1
            break

        garden[position[0]][position[1]] = turn
        tiles_raked += 1

        forward_position = get_forward_position(position)  # get forward position and check whether it is valid
        if is_in_bounds(garden, forward_position[0], forward_position[1]):
            if is_tile_rakeable(garden, forward_position[0], forward_position[1]):
                position = forward_position
                continue  # if the forward position is valid, choose it and continue further
        else:
            break

        is_right_correct = is_left_correct = False
        right_position = get_side_position(position, "right")  # get right position and check whether it is valid
        if is_in_bounds(garden, right_position[0], right_position[1]):
            if is_tile_rakeable(garden, right_position[0], right_position[1]):
                is_right_correct = True

        left_position = get_side_position(position, "left")  # get left position and check whether it is valid
        if is_in_bounds(garden, left_position[0], left_position[1]):
            if is_tile_rakeable(garden, left_position[0], left_position[1]):
                is_left_correct = True

        if not is_right_correct and not is_left_correct:  # decision tree for sidestepping
            # this condition determines, whether the monk made a successful rake or he is stuck
            if not is_in_bounds(garden, left_position[0], left_position[1]):
                return turn, tiles_raked
            elif not is_in_bounds(garden, right_position[0], right_position[1]):
                return turn, tiles_raked
            else:
                return -1, tiles_raked
        elif is_right_correct and is_left_correct:  # if both sides are valid, use predefined decisions to decide
            if monk.random_decisions[0] > 0.5:
                position = right_position
            else:
                position = left_position
            monk.random_decisions.append(monk.random_decisions.pop(0))  # move used decision to the end
            continue
        elif is_right_correct and not is_left_correct:
            position = right_position
            continue
        elif not is_right_correct and is_left_correct:
            position = left_position
            continue

    return turn, tiles_raked  # return current number of the path and amount of raked tiles


# this is a placeholder function for one solution of a particular garden and monk stats
def rake_wrapper(start_garden, new_monk, max_fitness):
    new_garden = [x[:] for x in start_garden]  # init new garden identical to the starting one

    turn = 0  # this variable holds number of lines made by the monk
    for tile in new_monk.border_tiles:  # cycle through all possible entries and try raking
        turn, tiles_raked = rake_garden(new_garden, tile, turn, new_monk)
        new_monk.fitness += tiles_raked

        if turn != -1:  # if monk left the garden correctly
            new_monk.complete_rakes = turn
        if turn == -1:  # if monk is stuck inside
            new_monk.complete_rakes += 1
            new_monk.correct_finish = 0
            clean_garden(new_monk, new_garden, new_monk.complete_rakes)  # remove the line which got him stuck
            return new_garden

    if new_monk.fitness == max_fitness:
        new_monk.correct_finish = 1

    return new_garden


# main algorithm function
def simulated_annealing(main_garden, dimensions):
    initial_temperature, th_extender = annealing_settings_wrapper()  # get annealing settings
    threshold = ((initial_temperature ** 0.1) / initial_temperature) / th_extender  # calculate lowest possible temp
    th_extender = int(sqrt(th_extender))
    max_fitness = get_max_fitness(main_garden, dimensions)

    print_map(main_garden, 0)  # print initial map for the user

    starter_tiles = get_border_tiles(main_garden, dimensions)
    random_decisions = get_random_decisions(main_garden, dimensions)

    best_monk = Monk(0, 0, 0, random_decisions, starter_tiles)  # init best_monk, this var will hold the best solution
    best_garden = rake_wrapper(main_garden, best_monk, max_fitness)
    current_garden = copy.deepcopy(best_garden)
    current_monk = copy.deepcopy(best_monk)  # this var holds the currect previous monk

    temperature = initial_temperature
    i = 1
    while temperature > threshold:
        # make a copy of a current monk and alter it slightly, so it is still just a neighbor
        new_monk = Monk(0, 0, 0, current_monk.random_decisions.copy(), current_monk.border_tiles.copy())
        # two parameters that are alterable
        # try to alter decisions, then just swap order of two border_tiles
        if threshold > uniform(0, max(new_monk.random_decisions) + threshold):
            idx = randrange(0, len(new_monk.random_decisions))
            new_monk.random_decisions[idx] = round((new_monk.random_decisions[idx] + 0.5) % 1, 2)

        else:  # here is a swap of two entry tiles
            while True:
                idx1 = randrange(0, len(new_monk.border_tiles))
                idx2 = randrange(0, len(new_monk.border_tiles))
                if idx1 != idx2:
                    break

            new_monk.border_tiles[idx1], new_monk.border_tiles[idx2] = \
                new_monk.border_tiles[idx2], new_monk.border_tiles[idx1]

        new_garden = rake_wrapper(main_garden, new_monk, max_fitness)  # get new garden with new monk

        # difference parameter that compares new monks' results with best found so far
        best_diff = new_monk.fitness + new_monk.correct_finish - best_monk.fitness - best_monk.correct_finish

        if best_diff > 0:  # if better solution was found then the best one so far
            best_monk = copy.deepcopy(new_monk)
            best_garden = copy.deepcopy(new_garden)
        elif best_diff == 0:
            if new_monk.complete_rakes < best_monk.complete_rakes:
                best_monk = copy.deepcopy(new_monk)
                best_garden = copy.deepcopy(new_garden)

        curr_diff = new_monk.fitness + new_monk.correct_finish - current_monk.fitness - current_monk.correct_finish

        if curr_diff == 0:  # if better solution then previous was found
            if new_monk.complete_rakes < current_monk.complete_rakes:
                current_monk = copy.deepcopy(new_monk)
                current_garden = copy.deepcopy(new_garden)
        elif curr_diff > 0 or exp((new_monk.fitness + new_monk.correct_finish - current_monk.fitness -
                                   current_monk.correct_finish) / temperature) >= uniform(0, 1):
            current_monk = copy.deepcopy(new_monk)
            current_garden = copy.deepcopy(new_garden)

        if i % th_extender == 0:
            temperature = temperature * (0.9992 ** (i ** threshold))  # cool down the temperature
        i += 1

    print_algorithm_stats(best_monk, best_garden, max_fitness, i)

    return best_monk


if __name__ == "__main__":
    print("-----------------------------------------------")
    print("\n* Zen's garden (Simulated annealing)\n* Author: Marko Stahovec\n")
    print("-----------------------------------------------")
    main_garden = map_wrapper()
    dimensions = [len(main_garden), len(main_garden[0])]
    simulated_annealing(main_garden, dimensions)

    """
    initial_temperature = 25
    temperature = initial_temperature
    threshold = ((initial_temperature ** 0.1) / initial_temperature) / 10
    for i in range(1, 100000):
        temperature = temperature * (0.9992**(i**threshold))
        print(i, ": ", end="")
        print(temperature)
        if temperature < threshold:
            break
    print(threshold)
    """

    print("\nend")
