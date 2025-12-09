from deap import tools, algorithms

def run_nsga_ii(toolbox, population_size=300, generations=50, cxpb=0.8, mutpb=0.2):
    """
    Run NSGA-II algorithm for multi-objective optimization.

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
    print("first stage")
    # Evaluate initial population
    for ind in population:
        print(ind)
        ind.fitness.values = toolbox.evaluate(ind)
    print("second stage")
    for gen in range(generations):
        print(f"NSGA-II: Generation {gen + 1}/{generations}")

        # Variation: Crossover and mutation
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)

        # Evaluate offspring
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)

        # Combine population and offspring
        combined_population = population + offspring

        # Select the next generation using non-dominated sorting and crowding distance
        population = toolbox.select(combined_population, k=population_size)

    # Extract Pareto front
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    return pareto_front
