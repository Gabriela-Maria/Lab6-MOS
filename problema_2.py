import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import time

N_GEN = 250
N_POP = 250
mutate_rate = 0.1
CNG_LEN = 5

def create_maze(dim, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    maze = np.ones((dim * 2 + 1, dim * 2 + 1), dtype=int)

    x, y = (0, 0)
    maze[2 * x + 1, 2 * y + 1] = 0
    stack = [(x, y)]

    while stack:
        x, y = stack[-1]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < dim and 0 <= ny < dim and maze[2 * nx + 1, 2 * ny + 1] == 1:
                maze[2 * nx + 1, 2 * ny + 1] = 0
                maze[2 * x + 1 + dx, 2 * y + 1 + dy] = 0
                stack.append((nx, ny))
                break
        else:
            stack.pop()

    maze[1, 0] = 0  # Entrada
    maze[-2, -1] = 0  # Salida
    return maze


def percibir(maze, x, y, dir_idx):
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def hay_pared(dx, dy):
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]):
            return maze[nx][ny] == 1
        return True

    left = directions[(dir_idx - 1) % 4]
    front = directions[dir_idx]
    right = directions[(dir_idx + 1) % 4]

    percepcion = []
    for vec in [left, front, right]:  # nuevo orden, estable
        percepcion.append("P" if hay_pared(*vec) else "L")

    return "_".join(percepcion)

# en el anterior era simulate(maze, commands, start=(1, 0), start_dir=0, show=False):
def simulate_mealy(maze, individuo, start=(1, 0), start_dir=1, max_steps=300, show=False):
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
    x, y = start
    dir_idx = start_dir  # comienza mirando hacia la derecha
    steps = 0
    estado = "A"

    for _ in range(max_steps):
        percepcion = percibir(maze, x, y, dir_idx)

        if (estado, percepcion) not in individuo:
            return (x, y), 1, steps  # no sabe qué hacer

        accion, nuevo_estado = individuo[(estado, percepcion)]
        estado = nuevo_estado

        if show:
            print(f"[{estado}] Percepción: {percepcion} → Acción: {accion}")

        if accion == "L":
            dir_idx = (dir_idx - 1) % 4
        elif accion == "R":
            dir_idx = (dir_idx + 1) % 4
        # "S" o avanzar recto
        dx, dy = directions[dir_idx]
        nx, ny = x + dx, y + dy

        if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and maze[nx][ny] == 0:
            x, y = nx, ny
            steps += 1

            if (x, y) == (len(maze) - 2, len(maze[0]) - 1):
                return (x, y), 0, steps  # llegó exitosamente
        else:
            steps += 1  # cuenta intento fallido pero continúa
            continue

    return (x, y), 1, steps  # no llegó a la meta


def fitness(indiv, mazes):
    total_score = 0

    for maze in mazes:
        end_pos = (len(maze) - 2, len(maze[0]) - 1)
        (x, y), penalties, steps = simulate_mealy(maze, indiv)

        dist = abs(end_pos[0] - x) + abs(end_pos[1] - y)

        score = -dist * 5 - penalties * 800 + steps * 25

        if (x, y) == end_pos:
            score += 1e8

        total_score += score

    return total_score / len(mazes)

def fitness_por_maze(indiv, maze):
    end_pos = (len(maze) - 2, len(maze[0]) - 1)
    (x, y), penalties, steps = simulate_mealy(maze, indiv)
    dist = abs(end_pos[0] - x) + abs(end_pos[1] - y)
    score = -dist * 5 - penalties * 800 + steps * 25
    if (x, y) == end_pos:
        score += 1e8
    return score


def generarGen():
    return random.choice(["S"] * 3 + ["L"] + ["R"])


# en el anterior era cromosoma(longitudInicial):
def generar_individuo_mealy():
    inputs = [
        "L_L_L", "P_L_L", "L_P_L", "L_L_P",
        "P_P_L", "P_L_P", "L_P_P", "P_P_P"
    ]
    estados = ["A", "B", "C"]  # puedes ajustar según la complejidad deseada
    acciones = ["S", "L", "R"]

    individuo = {}
    for estado in estados:
        for percepcion in inputs:
            accion = random.choice(acciones)
            nuevo_estado = random.choice(estados)
            individuo[(estado, percepcion)] = (accion, nuevo_estado)

    return individuo



def agregarCromosoma(cromosoma, aAgregar):
    return cromosoma + [generarGen() for _ in range(aAgregar)]


def crossover_mealy(ind1, ind2):
    hijo = {}
    for clave in ind1:
        if random.random() < 0.5:
            hijo[clave] = ind1[clave]
        else:
            hijo[clave] = ind2[clave]
    return hijo



def mutate_mealy(individuo, mutation_rate=0.1):
    acciones = ["S", "L", "R"]
    estados = ["A", "B", "C"]
    nuevo = {}

    for clave, (accion, estado_siguiente) in individuo.items():
        if random.random() < mutation_rate:
            nueva_accion = random.choice(acciones)
            nuevo_estado = random.choice(estados)
            nuevo[clave] = (nueva_accion, nuevo_estado)
        else:
            nuevo[clave] = (accion, estado_siguiente)

    return nuevo



def select(population, mazes):
    tournament = random.sample(population, 5)
    return max(tournament, key=lambda indiv: fitness(indiv, mazes))


def imprimir_mealy(indiv, maze):
    path_maze = maze.copy()
    x, y = 1, 0
    dir_idx = 1  # comienza mirando a la derecha
    estado = "A"
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    path_maze[x][y] = 2  # marca el inicio

    for _ in range(300):
        percepcion = percibir(maze, x, y, dir_idx)
        if (estado, percepcion) not in indiv:
            break

        accion, estado = indiv[(estado, percepcion)]

        if accion == "L":
            dir_idx = (dir_idx - 1) % 4
        elif accion == "R":
            dir_idx = (dir_idx + 1) % 4
        # si es "S", sigue recto

        dx, dy = directions[dir_idx]
        nx, ny = x + dx, y + dy

        if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and maze[nx][ny] == 0:
            x, y = nx, ny
            path_maze[x][y] = 2  # marca el camino
            if (x, y) == (len(maze) - 2, len(maze[0]) - 1):
                break  # llegó a la salida
        else:
            continue  # no se mueve pero sigue intentando

    display_maze(path_maze)


def imprimir_en_todos_los_mazes(indiv, mazes):
    for i, maze in enumerate(mazes):
        print(f"\n🧭 Maze #{i + 1}")
        (x, y), penal, steps = simulate_mealy(maze, indiv, (1, 0), 1, show=False)
        print(f"Resultado: posición final=({x}, {y}), penalización={penal}, pasos={steps}")
        imprimir_mealy(indiv, maze)


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


def genetico(mazes):
    start = time.time()

    lenIndiv = CNG_LEN  # ya no se usa directamente pero puedes dejarlo para luego
    poblacion = [generar_individuo_mealy() for _ in range(N_POP)]
    poblacion.sort(key=lambda indiv: fitness(indiv, mazes), reverse=True)
    fit = {}
    mejor = [-1e9, []]

    for gen in range(N_GEN):
        stagIndex = isFitnessStagnant(fit)

        elite = [poblacion[0], poblacion[1], poblacion[2]]
        nueva_gen = []

        # Guarda élite tal cual para no destruir lo poco que aprendieron
        nueva_gen.extend(elite)



        # Generar descendencia por selección + cruce + mutación (falta adaptar operadores)
        for _ in range(N_POP):
            p1 = select(poblacion, mazes)
            p2 = select(poblacion, mazes)
            hijo = crossover_mealy(p1, p2)
            hijo = mutate_mealy(hijo, mutation_rate=0.1)

            nueva_gen.append(hijo)

        poblacion = nueva_gen
        poblacion.sort(key=lambda indiv: fitness(indiv, mazes), reverse=True)
        poblacion = poblacion[:N_POP]

        fitnessGeneracion = fitness(poblacion[0], mazes)
        print(f"\nGEN #{gen}")
        for i, maze in enumerate(mazes):
            score = fitness_por_maze(poblacion[0], maze)
            print(f"  Maze #{i+1}: {score}")

        (x, y), penalties, steps = simulate_mealy(mazes[0], poblacion[0], (1, 0), 1, False)
        fit[gen] = fitnessGeneracion

        if fitnessGeneracion > mejor[0]:
            mejor[0] = fitnessGeneracion
            mejor[1] = poblacion[0]

        if fitnessGeneracion > 1e7:
            break

    end = time.time()
    print("Tiempo total:", int(end - start), "segundos")
    evaluar_en_todos_los_mazes(mejor[1], mazes)

    imprimir_en_todos_los_mazes(mejor[1], mazes)

def evaluar_en_todos_los_mazes(individuo, mazes):
    for i, maze in enumerate(mazes):
        print(f"\n📍 Maze #{i+1}")
        (x, y), penal, steps = simulate_mealy(maze, individuo, (1, 0), 1, show=False)
        print(f"Resultado final: ({x}, {y}), penalización: {penal}, pasos: {steps}")
        imprimir_mealy(individuo, maze)


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

K = 10
K_MAZES = [create_maze(7, seed=i) for i in range(K)]  # Semilla fija para reproducibilidad

genetico(K_MAZES)