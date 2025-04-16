from deap import tools, algorithms
import numpy as np

def assign_spea2_fitness(population):
    """
    Compute SPEA2 fitness for a population.
    The fitness is based on strength, raw fitness, and density.
    """
    N = len(population)
    strengths = np.zeros(N)
    raw_fitness = np.zeros(N)

    # Calculate strength
    for i in range(N):
        for j in range(N):
            if tools.emo.isDominated(population[j].fitness.values, population[i].fitness.values):
                strengths[i] += 1

    # Calculate raw fitness
    for i in range(N):
        for j in range(N):
            if tools.emo.isDominated(population[i].fitness.values, population[j].fitness.values):
                raw_fitness[i] += strengths[j]

    # Calculate density
    distances = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            distances[i, j] = np.linalg.norm(np.array(population[i].fitness.values) - np.array(population[j].fitness.values))
        distances[i] = np.sort(distances[i])
    k = int(np.sqrt(N))
    densities = 1.0 / (distances[:, k] + 2.0)

    # SPEA2 scalar fitness
    for i in range(N):
        population[i].spea2_fitness = raw_fitness[i] + densities[i]

def run_spea2(toolbox, population_size=300, generations=50, cxpb=0.8, mutpb=0.2, archive_size=100):
    """
    Run SPEA2 algorithm for multi-objective optimization.
    """
    population = toolbox.population(n=population_size)
    archive = []

    # Evaluate initial population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    for gen in range(generations):
        print(f"SPEA2: Generation {gen + 1}/{generations}")

        combined_population = population + archive
        assign_spea2_fitness(combined_population)

        combined_population.sort(key=lambda ind: ind.spea2_fitness)
        archive = combined_population[:archive_size]

        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)

        population = tools.selBest(archive + offspring, k=population_size)

    pareto_front = tools.sortNondominated(archive, len(archive), first_front_only=True)[0]
    return pareto_front
