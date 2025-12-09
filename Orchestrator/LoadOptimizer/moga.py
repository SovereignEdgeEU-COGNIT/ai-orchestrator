from deap import tools, algorithms

def run_moga(toolbox, population_size=300, generations=50, cxpb=0.8, mutpb=0.2):
    """
    Run MOGA algorithm for multi-objective optimization.

    Parameters:
    - toolbox: DEAP toolbox with problem-specific setup.
    - population_size: Number of individuals in the population.
    - generations: Number of generations.
    - cxpb: Crossover probability.
    - mutpb: Mutation probability.

    Returns:
    - Pareto front of the final generation.
    """
    # Initialize population
    population = toolbox.population(n=population_size)

    # Evaluate initial population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    for gen in range(generations):
        print(f"MOGA: Generation {gen + 1}/{generations}")

        # Compute scalar fitness as a weighted sum for selection purposes
        weights = [1 / len(ind.fitness.values)] * len(ind.fitness.values)  # Equal weights by default
        scalar_fitness = [
            sum(weight * value for weight, value in zip(weights, ind.fitness.values))
            for ind in population
        ]

        # Assign scalar fitness for selection purposes
        for ind, scalar_fit in zip(population, scalar_fitness):
            ind.scalar_fitness = scalar_fit  # Temporary scalar fitness attribute

        # Sort individuals by scalar fitness
        population.sort(key=lambda ind: ind.scalar_fitness)

        # Perform variation: Crossover and mutation
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)

        # Evaluate offspring
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)

        # Combine population and offspring
        population = tools.selBest(offspring + population, k=population_size)

    # Extract Pareto front
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    return pareto_front
