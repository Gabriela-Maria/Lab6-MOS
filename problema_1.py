import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import time

N_GEN = 250
N_POP = 250
mutate_rate = 0.1
CNG_LEN = 5


def simulate(maze, commands, start=(1, 0), start_dir=0, show=False):
    # Directions: Up, Right, Down, Left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    x, y = start
    dir_idx = start_dir
    steps = 0

    for cmd in commands:
        if show:
            print(x, y)
        if cmd == "L":
            dir_idx = (dir_idx - 1) % 4
        elif cmd == "R":
            dir_idx = (dir_idx + 1) % 4

        dx, dy = directions[dir_idx]
        nx, ny = x + dx, y + dy
        if show:
            print("> ", nx, ny)
        if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and maze[nx][ny] == 0:
            x, y = nx, ny
            if show:
                print("DS")
            steps += 1
        else:
            return (x, y), 1, steps

    return (x, y), 0, steps


def fitness(indiv, maze):
    end_pos = (len(maze) - 2, len(maze[0]) - 1)
    (x, y), penalties, steps = simulate(maze, indiv)

    dist = abs(end_pos[0] - x) + abs(end_pos[1] - y)

    score = -dist * 5 - penalties * 800 + steps * 25

    if (x, y) == end_pos:
        score += 1e8

    return score


def generarGen():
    return random.choice(["S"] * 3 + ["L"] + ["R"])


def cromosoma(longitudInicial):
    return [generarGen() for _ in range(longitudInicial)]


def agregarCromosoma(cromosoma, aAgregar):
    return cromosoma + [generarGen() for _ in range(aAgregar)]


def crossover(indiv1, indiv2):
    return indiv1[: len(indiv1) - CNG_LEN] + indiv2[-CNG_LEN:]


def mutate(indiv, stagIndex):
    for i in range(len(indiv) - int(1.5 * CNG_LEN) - stagIndex, len(indiv)):
        if random.random() < 0.6:
            indiv[i] = generarGen()

    return indiv


def select(population, maze):
    tournament = random.sample(population, 5)
    return max(tournament, key=lambda indiv: fitness(indiv, maze))


def imprimir(poblacion, maze):
    best = poblacion[0]
    path_maze = maze.copy()
    x, y = 1, 0
    dir_idx = 0
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    path_maze[x][y] = 2

    for cmd in best:
        if cmd == "L":
            dir_idx = (dir_idx - 1) % 4
        elif cmd == "R":
            dir_idx = (dir_idx + 1) % 4

        dx, dy = directions[dir_idx]
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and maze[nx][ny] == 0:
            x, y = nx, ny
            path_maze[x][y] = 2

    display_maze(path_maze)


def isFitnessStagnant(fit):
    if len(fit) > 1:
        baseline = fit[len(fit) - 1]
    cou = 0

    for i in reversed(range(0, 250)):
        if i not in fit:
            break

        if fit[i] != baseline:
            break

        cou += 1

    if cou > 10 and cou < 15:
        return CNG_LEN * 1

    if cou >= 15 and cou < 20:
        return CNG_LEN * 2

    if cou >= 20:
        return CNG_LEN * 3

    return 0


def genetico(maze):
    start = time.time()

    lenIndiv = CNG_LEN
    poblacion = [cromosoma(lenIndiv) for _ in range(N_POP)]
    poblacion.sort(key=lambda indiv: fitness(indiv, maze), reverse=True)
    fit = {}
    mejor = [-1e9, []]

    for gen in range(250):
        stagIndex = isFitnessStagnant(fit)

        elite = [poblacion[0], poblacion[1], poblacion[2]]
        nueva_gen = []

        for _ in range(10):
            for indiv in elite:
                nueva_gen.append(
                    mutate(agregarCromosoma(indiv, lenIndiv - len(indiv)), stagIndex)
                )

        poblacion = [agregarCromosoma(_, lenIndiv - len(indiv)) for _ in poblacion]

        for i in range(N_POP):
            p1 = select(poblacion, maze)
            p2 = select(poblacion, maze)

            nueva_gen.append(mutate(crossover(p1, p2), stagIndex))

        for indiv in poblacion:
            if random.random() < mutate_rate:
                nueva_gen.append(mutate(indiv, stagIndex))

        poblacion = nueva_gen
        poblacion.sort(key=lambda indiv: fitness(indiv, maze), reverse=True)
        poblacion = poblacion[:N_POP]

        fitnessGeneracion = fitness(poblacion[0], maze)

        print("GEN #", gen, ":", fitnessGeneracion)  # , #poblacion[0])

        (x, y), penalties, steps = simulate(maze, poblacion[0], (1, 0), 0, False)

        fit[gen] = fitness(poblacion[0], maze)

        if fitnessGeneracion > mejor[0]:
            mejor[0] = fitnessGeneracion
            mejor[1] = poblacion[0]

        # print(x, y)
        # print(steps, lenIndiv)
        if steps == lenIndiv:
            lenIndiv += CNG_LEN

        if fitnessGeneracion > 1e7:
            break

    end = time.time()

    print("Tiempo total:", int(end - start), "segundos")

    imprimir([mejor[1]], maze)


def display_maze(maze):
    cmap = ListedColormap(["white", "black", "green", "black"])
    plt.figure(figsize=(6, 6))
    plt.pcolor(maze[::-1], cmap=cmap, edgecolors="k", linewidths=2)
    plt.gca().set_aspect("equal")
    plt.xticks([])
    plt.yticks([])
    plt.title("Maze")
    plt.show()


start = time.time()

maze = np.loadtxt("./maze_case_heavy.txt", dtype=int)
genetico(maze)