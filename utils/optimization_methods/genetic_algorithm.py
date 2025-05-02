import numpy as np
import pandas as pd

# Constants for Genetic Algorithm
TOTAL_SHELF_SPACE = 1000  # Total shelf capacity
LP_MAX_BUFFER = 1.2       # Multiplier for max allocation based on recent sales
LP_MIN_BUFFER = 0.2       # Multiplier for min allocation based on recent sales
LP_MIN_STOCK_MIN = 5      # Minimum units per SKU

GA_POPULATION_SIZE = 50    # Number of individuals in each generation
GA_GENERATIONS = 100      # Total generations to evolve
GA_MUTATION_RATE = 0.1     # Probability of mutation per offspring

def optimize_ga(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize shelf space allocation using a simple genetic algorithm (GA).

    Steps:
      1. Initialize a population of random allocations summing to TOTAL_SHELF_SPACE.
      2. Evaluate fitness (total profit minus penalties for violations).
      3. Select parents by tournament selection (choose best of 3 randomly picked).
      4. Produce offspring via single-point crossover and random mutations.
      5. Repeat over GA_GENERATIONS, then select the fittest individual.

    Returns a DataFrame with columns:
      ['Product_Name', 'Category', 'Allocated_Space', 'Profit_Per_Unit']
    """
    # Convert DataFrame columns to numpy arrays for performance
    profits = df['Profit_Per_Unit'].to_numpy()
    sales = df['Sales_Last_30_Days'].to_numpy()
    n = len(df)

    def create_individual() -> np.ndarray:
        """
        Create a random allocation vector that sums to TOTAL_SHELF_SPACE.
        """
        weights = np.random.rand(n)
        allocation = (weights / weights.sum() * TOTAL_SHELF_SPACE).astype(int)
        return allocation

    def fitness(individual: np.ndarray) -> float:
        """
        Calculate fitness: sum(profit*allocation) minus penalties.
        Penalties applied for:
          - Exceeding total shelf space.
          - Allocations above sales * LP_MAX_BUFFER.
          - Allocations below LP_MIN_STOCK_MIN or sales * LP_MIN_BUFFER.
        """
        total_profit = individual.dot(profits)
        penalty = 0.0
        total_alloc = individual.sum()
        # Over-capacity penalty
        if total_alloc > TOTAL_SHELF_SPACE:
            penalty += (total_alloc - TOTAL_SHELF_SPACE) * 100
        # Bounds violation penalties
        over = individual > sales * LP_MAX_BUFFER
        under = individual < np.minimum(LP_MIN_STOCK_MIN, sales * LP_MIN_BUFFER)
        penalty += (over.sum() + under.sum()) * 100
        return total_profit - penalty

    # 1. Initialization
    population = [create_individual() for _ in range(GA_POPULATION_SIZE)]

    # Evolution process
    for _ in range(GA_GENERATIONS):
        # Evaluate fitness score for each individual
        scores = np.array([fitness(ind) for ind in population])
        parents = []
        # Tournament selection to pick parents
        for _ in range(GA_POPULATION_SIZE):
            contenders = np.random.choice(GA_POPULATION_SIZE, 3, replace=False)
            best = contenders[np.argmax(scores[contenders])]
            parents.append(population[best])

        # Crossover and mutation to create next generation
        new_population = []
        for i in range(0, GA_POPULATION_SIZE, 2):
            p1, p2 = parents[i], parents[i+1]
            # Single-point crossover
            crossover_point = np.random.randint(n)
            child1 = np.concatenate([p1[:crossover_point], p2[crossover_point:]])
            child2 = np.concatenate([p2[:crossover_point], p1[crossover_point:]])
            # Random mutation
            if np.random.rand() < GA_MUTATION_RATE:
                idx = np.random.randint(n)
                child1[idx] = np.random.randint(TOTAL_SHELF_SPACE)
            if np.random.rand() < GA_MUTATION_RATE:
                idx = np.random.randint(n)
                child2[idx] = np.random.randint(TOTAL_SHELF_SPACE)
            new_population.extend([child1, child2])
        population = new_population

    # 5. Select the best individual from the final population
    best_alloc = max(population, key=fitness).astype(int)
    # Attach allocation results to DataFrame
    result = df.copy()
    result['Allocated_Space'] = best_alloc
    return result[['Product_Name', 'Category', 'Allocated_Space', 'Profit_Per_Unit']]