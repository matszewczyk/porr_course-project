import numpy as np
# Zmieniam znak przy 20 na koncu równania (z +20 na -20)

def func1(x):
    x = np.array(x)
    i = np.array(range(1, len(x) + 1))
    result = 1 / 40 * np.sum(x ** 2) + 1 - np.prod(np.cos(x / i))

    return result

def func2(x):
    x = np.array(x)
    n = len(x)
    result_1 = 20 * np.exp(-0.2 * np.sqrt(1/n * np.sum(x**2)))
    result_2 = - np.exp(1/n * np.sum(np.cos(2*np.pi*x))) - 20 + np.e
    return result_1 + result_2


def initial_pop_generation(x, n, start_wth_zeros=False, min_value_n=None,
                           max_value_n=None):
    '''
    x : int
        parzysta liczba osobników
    n : int
        parzysta liczba genów
    start_wth_zeros : bool
        True - początkowe cechy osobnika równ zeru
        False - losowo dobrane cechy początkowe
    '''
    population = []
    if start_wth_zeros:
        for i in range(x):
            individual = np.array([0 for i in range(0, n)])
            population.append(individual)
    else:
        if (min_value_n == None or max_value_n == None):
            raise Exception('Set min&max value for n or set start_with_zeros to True')

        for i in range(x):
            individual = np.array([np.random.randint(min_value_n, max_value_n) for i in range(0, n)])
            population.append(individual)

    return population

def crossing(indiv1, indiv2,n):
    '''
    indiv1

    indiv2
    '''
    #losowanie locus podzialu
    locus = np.random.randint(0, n)
    embryo1 = np.concatenate((indiv1[:locus],indiv2[locus:]))
    embryo2 = np.concatenate((indiv2[:locus],indiv1[locus:]))

    return embryo1, embryo2


def mutate(individual, p_mutate):
    '''
    individuals : [int]
        list of the genes of the individual
    '''

    for i in range(len(individual)):
        r = np.random.rand()
        if r < p_mutate / 2:
            individual[i] -= 1
        elif r < p_mutate:
            individual[i] += 1

    return individual


def optimize(population, x, n, function, min_or_max, end_optim, p_mutate=0.2):
    '''
    population : list of numpy arrays
        zbiór wszystkich osobników
    '''
    best_individual_scores = []
    iteration = 0
    iteration_dict = {}
    score_dict = {}

    while True:
        iteration += 1
        # krzyzowanie i mutacja
        embryons = []
        pop_score_dict = {}
        for i in range(0, x, 2):
            crossing_result = crossing(population[i], population[i + 1], n)
            for embryon in crossing_result:
                population.append(mutate(embryon, p_mutate))
        scores = []
        for individual in population:
            score = function(individual)
            scores.append(score)
            pop_score_dict[tuple(individual)] = float(score)

        if min_or_max == 'max':
            reverse = True
        elif min_or_max == 'min':
            reverse = False
        else:
            raise Exception('min_or_max parameter invalid. Type "min" or "max".')

        population_score = sorted(zip(population, scores), key=lambda x: x[1], reverse=reverse)
        best_individual_gen = population_score[0][0]
        best_individual_score = population_score[0][1]

        best_individual_scores.append(best_individual_score)

        sorted_population, _ = zip(*population_score)
        population = list(sorted_population[:x])

        print(f'Iteration: {iteration} | Best score: {best_individual_score}')
        iteration_dict[int(f'{iteration}')] = best_individual_score
        score_dict[int(f'{iteration}')] = pop_score_dict

        # zakończ jeśli nie ma poprawy w 100 ostatnich iteracjach (?)
        if len(set(best_individual_scores[-end_optim:])) == 1 and iteration > 10:
            break

    return best_individual_gen, best_individual_score, iteration_dict, score_dict


x = 50

# Liczba genów osobnika
n = 2

# Minimalna wartość genu
min_value_n = -40

# Maksymalna wartość genu
max_value_n = 40

# Inicjowanie osobnika zerami (w przypadku tych zadan nie ma sensu)
start_wth_zeros = False

# Pradopodobieństwo mutacji
p_mutate = 0.2

# Funkcja oceny
function = func1

# Cel (min/max)
min_or_max='min'

# Zakończ, jeśli wynik najlepszego osobnika nie zmienił się od ... iteracji.
end_optim=50

population = initial_pop_generation(x,n,start_wth_zeros,min_value_n,max_value_n)

best_gen, best_score, iteration_dict, score_dict = optimize(population,x,n,function, min_or_max,end_optim,p_mutate)

print(f'Najlepszy osobnik: {best_gen}\n')
print(f'Wynik dla najlepszego osobnika: {best_score}')


import matplotlib.pylab as plt

for n in range(1, len(list(score_dict.keys())) + 1):
    x = list()
    y = list()
    z = list()
    plt.title('Rozkład populacji na poziomicach funkcji celu: \n'
              + 'Iteracja: ' + str(n))
    plt.xlabel('Wartość x1')
    plt.ylabel('Wartość x2')
    for i in range(0, len(list(score_dict[n].keys()))):
        dict_iter_n = score_dict[n].keys()
        x.append(int(list(dict_iter_n)[i][0]))
        y.append(int(list(dict_iter_n)[i][1]))
        z.append(float(list(score_dict[n].values())[i]))
    print(x)
    print(y)
    print(z)
    plt.tricontour(x, y, z)
    plt.pause(0.50)
    if  n < len(list(score_dict.keys())):
        print(n)
        plt.cla()
    else:
        plt.show()

