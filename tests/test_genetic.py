"""
Tests for genetic algorithm domain.
"""

import pytest
import numpy as np
from morphogen.stdlib.genetic import (
    genetic, Individual, Population, GeneticOperations, parallel_fitness_evaluation
)


class TestIndividualOperations:
    """Test individual operations"""

    def test_alloc_individual(self):
        """Test individual allocation"""
        ind = genetic.alloc_individual(genome_size=10, seed=42)

        assert ind.genome.shape == (10,)
        assert ind.fitness == 0.0
        assert ind.age == 0
        assert ind.id == 0

    def test_individual_deterministic(self):
        """Test deterministic initialization"""
        ind1 = genetic.alloc_individual(10, seed=42)
        ind2 = genetic.alloc_individual(10, seed=42)

        np.testing.assert_array_equal(ind1.genome, ind2.genome)

    def test_individual_copy(self):
        """Test individual copy semantics"""
        ind1 = genetic.alloc_individual(10, seed=42)
        ind2 = ind1.copy()

        np.testing.assert_array_equal(ind1.genome, ind2.genome)

        # Modify copy
        ind2.genome[0] = 999.0
        assert ind1.genome[0] != 999.0


class TestPopulationOperations:
    """Test population operations"""

    def test_alloc_population(self):
        """Test population allocation"""
        pop = genetic.alloc_population(pop_size=20, genome_size=10, seed=42)

        assert pop.size == 20
        assert len(pop.individuals) == 20
        assert pop.generation == 0
        assert all(ind.genome.shape == (10,) for ind in pop.individuals)

    def test_population_deterministic(self):
        """Test deterministic population initialization"""
        pop1 = genetic.alloc_population(10, 5, seed=42)
        pop2 = genetic.alloc_population(10, 5, seed=42)

        for ind1, ind2 in zip(pop1.individuals, pop2.individuals):
            np.testing.assert_array_equal(ind1.genome, ind2.genome)

    def test_population_copy(self):
        """Test population copy semantics"""
        pop1 = genetic.alloc_population(10, 5, seed=42)
        pop2 = pop1.copy()

        assert pop1.size == pop2.size
        assert pop1.generation == pop2.generation

        # Modify copy
        pop2.individuals[0].genome[0] = 999.0
        assert pop1.individuals[0].genome[0] != 999.0


class TestFitnessOperations:
    """Test fitness evaluation"""

    def test_evaluate_fitness(self):
        """Test fitness evaluation"""
        pop = genetic.alloc_population(5, 10, seed=42)

        # Simple fitness: sum of genome
        def fitness_fn(genome):
            return np.sum(genome)

        pop = genetic.evaluate_fitness(pop, fitness_fn)

        # All individuals should have fitness assigned
        assert all(ind.fitness != 0.0 for ind in pop.individuals)

    def test_rank_population(self):
        """Test population ranking"""
        pop = genetic.alloc_population(5, 10, seed=42)

        # Assign random fitnesses
        for i, ind in enumerate(pop.individuals):
            ind.fitness = float(i)

        pop = genetic.rank_population(pop)

        # Should be sorted descending
        fitnesses = [ind.fitness for ind in pop.individuals]
        assert fitnesses == sorted(fitnesses, reverse=True)


class TestSelectionOperations:
    """Test selection operators"""

    def test_tournament_selection(self):
        """Test tournament selection"""
        pop = genetic.alloc_population(10, 5, seed=42)

        # Assign fitnesses (0 to 9)
        for i, ind in enumerate(pop.individuals):
            ind.fitness = float(i)

        # Tournament should favor higher fitness
        selected = genetic.tournament_selection(pop, tournament_size=3, seed=42)

        assert selected.fitness >= 0

    def test_roulette_selection(self):
        """Test roulette selection"""
        pop = genetic.alloc_population(10, 5, seed=42)

        # Assign fitnesses
        for i, ind in enumerate(pop.individuals):
            ind.fitness = float(i)

        selected = genetic.roulette_selection(pop, seed=42)

        assert selected.fitness >= 0

    def test_roulette_selection_negative_fitness(self):
        """Test roulette selection with negative fitness"""
        pop = genetic.alloc_population(5, 3, seed=42)

        # Assign negative fitnesses
        for i, ind in enumerate(pop.individuals):
            ind.fitness = float(i) - 5.0

        # Should handle negative fitness by shifting
        selected = genetic.roulette_selection(pop, seed=42)
        assert selected is not None

    def test_elitism_select(self):
        """Test elitism selection"""
        pop = genetic.alloc_population(10, 5, seed=42)

        # Assign fitnesses
        for i, ind in enumerate(pop.individuals):
            ind.fitness = float(i)

        pop = genetic.rank_population(pop)
        elites = genetic.elitism_select(pop, n_elite=3)

        assert len(elites) == 3
        # Top 3 fitnesses should be [9, 8, 7]
        assert elites[0].fitness == 9.0
        assert elites[1].fitness == 8.0
        assert elites[2].fitness == 7.0


class TestGeneticOperators:
    """Test genetic operators"""

    def test_mutate(self):
        """Test mutation"""
        genome = np.ones(10)
        mutated = genetic.mutate(genome, mutation_rate=1.0, mutation_scale=0.1, seed=42)

        # Should be different
        assert not np.allclose(mutated, genome)
        # Original should be unchanged
        np.testing.assert_array_equal(genome, np.ones(10))

    def test_mutate_rate(self):
        """Test mutation rate"""
        genome = np.ones(100)
        mutated = genetic.mutate(genome, mutation_rate=0.1, mutation_scale=10.0, seed=42)

        # Count how many genes changed significantly
        changed = np.sum(np.abs(mutated - genome) > 0.01)
        # Should be around 10 ± 5
        assert 5 <= changed <= 20

    def test_crossover_uniform(self):
        """Test uniform crossover"""
        genome1 = np.ones(10)
        genome2 = np.zeros(10)

        child1, child2 = genetic.crossover_uniform(genome1, genome2, seed=42)

        # Children should be mix of 0s and 1s
        assert np.all((child1 == 0) | (child1 == 1))
        assert np.all((child2 == 0) | (child2 == 1))
        # Should be complementary
        np.testing.assert_array_almost_equal(child1 + child2, np.ones(10))

    def test_crossover_single_point(self):
        """Test single-point crossover"""
        genome1 = np.ones(10)
        genome2 = np.zeros(10)

        child1, child2 = genetic.crossover_single_point(genome1, genome2, seed=42)

        # Each child should have contiguous runs
        assert np.all((child1 == 0) | (child1 == 1))

    def test_crossover_blend(self):
        """Test blend crossover"""
        genome1 = np.ones(10) * 2.0
        genome2 = np.ones(10) * 4.0

        child1, child2 = genetic.crossover_blend(genome1, genome2, alpha=0.5)

        # Should be average (3.0)
        np.testing.assert_array_almost_equal(child1, np.ones(10) * 3.0)
        np.testing.assert_array_almost_equal(child2, np.ones(10) * 3.0)


class TestBreedingOperations:
    """Test breeding (crossover + mutation)"""

    def test_breed(self):
        """Test breeding two parents"""
        parent1 = genetic.alloc_individual(10, seed=1)
        parent2 = genetic.alloc_individual(10, seed=2)

        child1, child2 = genetic.breed(
            parent1, parent2,
            crossover_method='uniform',
            mutation_rate=0.1,
            mutation_scale=0.1,
            seed=42
        )

        assert child1.genome.shape == (10,)
        assert child2.genome.shape == (10,)
        assert child1.fitness == 0.0
        assert child2.fitness == 0.0
        assert child1.age == 0
        assert child2.age == 0


class TestGenerationalEvolution:
    """Test generational evolution"""

    def test_evolve_generation(self):
        """Test one generation of evolution"""
        pop = genetic.alloc_population(20, 10, seed=42)

        # Simple fitness: minimize distance from zeros
        def fitness_fn(genome):
            return -np.sum(genome ** 2)

        pop = genetic.evolve_generation(
            pop,
            fitness_fn=fitness_fn,
            n_elite=2,
            selection_method='tournament',
            crossover_method='uniform',
            mutation_rate=0.1,
            mutation_scale=0.3,
            seed=42
        )

        # Generation should increment
        assert pop.generation == 1
        # Population size should be preserved
        assert pop.size == 20
        # Fitness history should be updated
        assert len(pop.best_fitness_history) == 1
        assert len(pop.mean_fitness_history) == 1

    def test_evolve_generation_elitism(self):
        """Test that elitism preserves best individuals"""
        pop = genetic.alloc_population(10, 5, seed=42)

        # Assign ascending fitness
        for i, ind in enumerate(pop.individuals):
            ind.fitness = float(i)

        pop = genetic.rank_population(pop)

        # Evolve with elitism
        def fitness_fn(genome):
            return 0.0  # Dummy, already evaluated

        new_pop = genetic.evolve_generation(
            pop,
            fitness_fn=fitness_fn,
            n_elite=3,
            seed=42
        )

        # Top 3 should have aged (elites)
        assert new_pop.individuals[0].age == 1
        assert new_pop.individuals[1].age == 1
        assert new_pop.individuals[2].age == 1
        # Rest should be new (age 0)
        assert new_pop.individuals[3].age == 0


class TestEvolutionLoop:
    """Test complete evolution loop"""

    def test_run_evolution(self):
        """Test running complete evolution"""
        pop = genetic.alloc_population(20, 10, seed=42)

        # Fitness: minimize squared distance from zeros
        def fitness_fn(genome):
            return -np.sum(genome ** 2)

        # Track progress
        best_fitnesses = []

        def callback(p):
            best_fitnesses.append(p.best_fitness_history[-1])

        final_pop = genetic.run_evolution(
            pop,
            fitness_fn=fitness_fn,
            n_generations=10,
            callback=callback,
            n_elite=2,
            mutation_rate=0.2,
            mutation_scale=0.5,
            seed=42
        )

        assert final_pop.generation == 10
        assert len(best_fitnesses) == 10
        # Fitness should improve (become less negative)
        assert best_fitnesses[-1] > best_fitnesses[0]

    def test_get_best_individual(self):
        """Test extracting best individual"""
        pop = genetic.alloc_population(10, 5, seed=42)

        # Assign fitnesses
        for i, ind in enumerate(pop.individuals):
            ind.fitness = float(i)

        best = genetic.get_best_individual(pop)
        assert best.fitness == 9.0

    def test_get_diversity(self):
        """Test diversity measurement"""
        pop = genetic.alloc_population(10, 5, seed=42)

        diversity = genetic.get_diversity(pop)

        # Should be positive
        assert diversity > 0

        # Homogeneous population should have low diversity
        for ind in pop.individuals:
            ind.genome = np.ones(5)

        diversity_low = genetic.get_diversity(pop)
        assert diversity_low < 0.01


class TestPresets:
    """Test preset configurations"""

    def test_flappy_bird_evolution(self):
        """Test Flappy Bird evolution preset"""
        pop = genetic.flappy_bird_evolution(pop_size=50, genome_size=49, seed=42)

        assert pop.size == 50
        assert pop.individuals[0].genome.shape == (49,)


class TestIntegration:
    """Integration tests"""

    def test_optimize_sphere_function(self):
        """Test optimizing simple sphere function"""
        # Minimize sum of squared values
        # Global minimum: all zeros, fitness = 0

        pop = genetic.alloc_population(30, 5, seed=42)

        def fitness_fn(genome):
            return -np.sum(genome ** 2)

        # Evolve
        final_pop = genetic.run_evolution(
            pop,
            fitness_fn=fitness_fn,
            n_generations=50,
            n_elite=3,
            mutation_rate=0.2,
            mutation_scale=0.3,
            seed=42
        )

        # Get best individual
        best = genetic.get_best_individual(final_pop)

        # Should be close to zeros
        assert best.fitness > -0.5  # Close to 0
        assert np.linalg.norm(best.genome) < 1.0  # Genome close to zeros

    def test_optimize_rastrigin_function(self):
        """Test optimizing Rastrigin function (harder, multimodal)"""
        def rastrigin(genome):
            A = 10
            n = len(genome)
            return -(A * n + np.sum(genome**2 - A * np.cos(2 * np.pi * genome)))

        pop = genetic.alloc_population(50, 3, seed=42)

        final_pop = genetic.run_evolution(
            pop,
            fitness_fn=rastrigin,
            n_generations=100,
            n_elite=5,
            mutation_rate=0.15,
            mutation_scale=0.5,
            seed=42
        )

        best = genetic.get_best_individual(final_pop)

        # Should find reasonably good solution (global optimum is 0)
        # This is a hard problem, so we just check for improvement
        assert final_pop.best_fitness_history[-1] > final_pop.best_fitness_history[0]

    def test_parallel_evaluation(self):
        """Test parallel fitness evaluation helper"""
        pop = genetic.alloc_population(10, 5, seed=42)

        # Batch fitness function
        def batch_fitness_fn(genomes):
            return [np.sum(g) for g in genomes]

        pop = parallel_fitness_evaluation(pop, batch_fitness_fn)

        # All individuals should have fitness
        assert all(ind.fitness != 0.0 for ind in pop.individuals)


class TestDeterminism:
    """Test deterministic behavior"""

    def test_deterministic_evolution(self):
        """Test that seeded evolution is reproducible"""
        def fitness_fn(genome):
            return -np.sum(genome ** 2)

        # Run 1
        pop1 = genetic.alloc_population(20, 10, seed=42)
        final1 = genetic.run_evolution(
            pop1, fitness_fn, n_generations=10, seed=42
        )

        # Run 2 (same seed)
        pop2 = genetic.alloc_population(20, 10, seed=42)
        final2 = genetic.run_evolution(
            pop2, fitness_fn, n_generations=10, seed=42
        )

        # Results should be identical
        np.testing.assert_array_almost_equal(
            final1.best_fitness_history,
            final2.best_fitness_history
        )
        np.testing.assert_array_almost_equal(
            final1.individuals[0].genome,
            final2.individuals[0].genome
        )
