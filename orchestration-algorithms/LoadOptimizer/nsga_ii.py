import pandas as pd
from deap import base, creator, tools, algorithms
import numpy as np

# Load your dataset
def load_data(file_path):
    # Assuming your dataset is a CSV file
    df = pd.read_csv(file_path)
    return df

# Define the problem as minimizing interference and maximizing green energy
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # Minimize interference, maximize green energy
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.random)  # Random initialization
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=4)  # Assuming 4 decision variables
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation function using the dataset
def evaluate(individual, data):
    """
    Evaluate the fitness of an individual.
    Individual parameters:
        - individual[0]: CPU allocation
        - individual[1]: Memory allocation
        - individual[2]: Green energy utilization weight
        - individual[3]: Interference weight
    """
    cpu, memory, green_energy_weight, interference_weight = individual

    # Example: Compute objectives based on weighted sums (replace with your logic)
    green_energy_score = np.sum(data['green_energy'] * green_energy_weight)
    interference_score = np.sum(data['interference'] * interference_weight)
    
    # Objectives: Minimize interference, maximize green energy
    obj1 = interference_score  # Minimize
    obj2 = green_energy_score  # Maximize
    return obj1, obj2

# Register evaluation with data dependency
def evaluate_with_data(ind):
    return evaluate(ind, data)

toolbox.register("evaluate", evaluate_with_data)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# NSGA-II Algorithm
def main(file_path):
    global data  # Make data accessible in evaluate function
    data = load_data(file_path)  # Load dataset
    
    # Normalize dataset if necessary
    data['green_energy'] = (data['green_energy'] - data['green_energy'].min()) / (data['green_energy'].max() - data['green_energy'].min())
    data['interference'] = (data['interference'] - data['interference'].min()) / (data['interference'].max() - data['interference'].min())

    population = toolbox.population(n=100)
    NGEN = 50
    CXPB, MUTPB = 0.9, 0.1

    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, CXPB, MUTPB)
        fits = list(map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, len(population))

    # Extract Pareto front solutions
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    return pareto_front

if __name__ == "__main__":
    # Example: Replace 'your_dataset.csv' with your actual dataset path
    dataset_path = "your_dataset.csv"
    pareto_front = main(dataset_path)

    # Print or save Pareto front solutions
    for solution in pareto_front:
        print("Solution:", solution, "Fitness:", solution.fitness.values)
from deap import base, creator, tools, algorithms

# NSGA-II: Non-dominated Sorting Genetic Algorithm II
# A popular multi-objective optimization algorithm using Pareto dominance.
def run_nsga_ii(toolbox, population_size=300, generations=100, cxpb=0.7, mutpb=0.3):
    """
    Run the NSGA-II algorithm to find a Pareto front for multi-objective optimization.

    Parameters:
    - toolbox: DEAP toolbox with problem-specific setup.
    - population_size: Number of individuals in the population.
    - generations: Number of generations to evolve.
    - cxpb: Crossover probability.
    - mutpb: Mutation probability.

    Returns:
    - Pareto front containing the best solutions across objectives.
    """
    population = toolbox.population(n=population_size)  # Initialize population
    for gen in range(generations):
        print(f"NSGA-II: Generation {gen + 1}/{generations}")  # Track progress
        # Apply crossover and mutation
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
        
        # Evaluate offspring
        fits = map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        # Select next generation based on non-dominated sorting
        population = toolbox.select(offspring, len(population))
    
    # Extract Pareto front from the final population
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    return pareto_front
