# https://mlrose.readthedocs.io/en/stable/source/algorithms.html

import dataframe_image as dfi
import matplotlib.pyplot as plt
import mlrose_hiive
import mlrose
import numpy as np
import pandas as pd
import time
import warnings

from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GridSearchCV, train_test_split, validation_curve)   
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
# from yellowbrick.model_selection import LearningCurve, ValidationCurve

warnings.filterwarnings("ignore")
np.random.seed(42)

def run_experiments(experiment=None, all=False):
    if experiment == 'fp':
        run_four_peaks()
    if experiment == 'cp':
        run_continuous_peaks()
    if experiment == 'nn':
        run_nn()
    if experiment == 'om':
        run_one_max()

"""
Four Peaks Optimization Problem
    1. Random Hill Climb Experiment
    2. Simulated Annealing Experiment
    3. Genetic Algorithm Experiment <--- 
    4. MIMIC Algorithm Experiment
"""
def run_four_peaks():
    print("Running Experiments for Four Peaks Optimization Problem")
    print()

    # Define Fitness function and discrete problem object
    fitness = mlrose_hiive.FourPeaks()
    problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True)

    max_attempts = 100
    max_iters = 100
    # RHC
    print("Running Random Hill Climb Experiment")
    start_time = time.time()
    rhc_best_state, rhc_best_fitness, rhc_fitness_curve = mlrose_hiive.random_hill_climb(problem, 
                                                            max_attempts = max_attempts, 
                                                            max_iters=max_iters, 
                                                            curve=True, 
                                                            random_state=42,
                                                            restarts=100)
    end_time = time.time()
    rhc_time = end_time - start_time
    print("Time (s): {}".format(rhc_time))
    print()

    # SA
    print("Running Simulated Annealing Experiment")
    start_time = time.time()
    sa_best_state, sa_best_fitness, sa_fitness_curve = mlrose_hiive.simulated_annealing(
                                                        problem, 
                                                        max_attempts=max_attempts, 
                                                        max_iters=max_attempts, 
                                                        curve=True, 
                                                        random_state=42,
                                                        schedule=mlrose_hiive.GeomDecay(init_temp = 1, decay=0.1, min_temp=1))
    end_time = time.time()
    sa_time = end_time - start_time
    print("Time (s): {}".format(sa_time))
    print()

    # GA
    print("Running Genetic Algorithm Experiment")
    start_time = time.time()
    ga_best_state, ga_best_fitness, ga_fitness_curve = mlrose_hiive.genetic_alg(
                                                        problem, 
                                                        max_attempts=max_attempts, 
                                                        max_iters=max_iters, 
                                                        curve=True, 
                                                        random_state=42,
                                                        pop_size=200,
                                                        mutation_prob=0.2)
    end_time = time.time()
    ga_time = end_time - start_time
    print("Time (s): {}".format(ga_time))
    print()

    # MIMIC
    print("Running MIMIC Algorithm Experiment")
    start_time = time.time()
    mimic_best_state, mimic_best_fitness, mimic_fitness_curve = mlrose_hiive.mimic(
                                                                problem, 
                                                                max_attempts = 100, 
                                                                max_iters = 100,  
                                                                curve = True, 
                                                                random_state = 42,
                                                                keep_pct=0.25)
    end_time = time.time()
    mimic_time = end_time - start_time
    print("Time (s): {}".format(mimic_time))
    print()

    # Plot Iterations vs Fitness
    iterations = range(1, 101)
    plt.plot(iterations, rhc_fitness_curve, label='RHC', color='green')
    plt.plot(iterations, sa_fitness_curve, label='SA', color='red')
    plt.plot(iterations, ga_fitness_curve, label='GA', color='blue')
    plt.plot(iterations, mimic_fitness_curve, label='MIMIC', color='orange')
    plt.legend(loc="best")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.savefig("results/fourpeaks_fitness.png")

    # Plot Time Table
    # https://www.geeksforgeeks.org/creating-a-pandas-dataframe-using-list-of-tuples/
    data = [('RHC', round(rhc_time, 5)), 
            ('SA', round(sa_time, 5)), 
            ('GA', round(ga_time, 5)), 
            ('MIMIC', round(mimic_time, 5))] 
    
    df = pd.DataFrame(data, columns =['Algorithm', 'Time (s)']) 
    dfi.export(df,"results/fourpeaks_times.png")

"""
Continuous Peaks Optimization Problem
    1. Random Hill Climb Experiment
    2. Simulated Annealing Experiment <---
    3. Genetic Algorithm Experiment
    4. MIMIC Algorithm Experiment
"""
def run_continuous_peaks():
    print("Running Experiments for Continuous Peaks Problem")
    print()

    # Define Fitness function and discrete problem object
    fitness = mlrose_hiive.ContinuousPeaks(t_pct=0.2)
    problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True)

    max_attempts = 100
    max_iters = 100

    # RHC
    print("Running Random Hill Climb Experiment")
    start_time = time.time()
    rhc_best_state, rhc_best_fitness, rhc_fitness_curve = mlrose_hiive.random_hill_climb(problem, 
                                                            max_attempts = max_attempts, 
                                                            max_iters=max_iters, 
                                                            curve=True, 
                                                            random_state=42,
                                                            restarts=100)
    end_time = time.time()
    rhc_time = end_time - start_time
    print("Time (s): {}".format(rhc_time))
    print()

    # SA
    print("Running Simulated Annealing Experiment")
    start_time = time.time()
    sa_best_state, sa_best_fitness, sa_fitness_curve = mlrose_hiive.simulated_annealing(
                                                        problem, 
                                                        max_attempts=max_attempts, 
                                                        max_iters=max_iters, 
                                                        curve=True, 
                                                        schedule=mlrose_hiive.GeomDecay(init_temp = 1, decay=0.1, min_temp=1))
    end_time = time.time()
    sa_time = end_time - start_time
    print("Time (s): {}".format(sa_time))
    print()

    # GA
    print("Running Genetic Algorithm Experiment")
    start_time = time.time()
    ga_best_state, ga_best_fitness, ga_fitness_curve = mlrose_hiive.genetic_alg(
                                                        problem, 
                                                        max_attempts=max_attempts, 
                                                        max_iters=max_iters, 
                                                        curve=True, 
                                                        random_state=42,
                                                        pop_size=200)
    end_time = time.time()
    ga_time = end_time - start_time
    print("Time (s): {}".format(ga_time))
    print()

    # MIMIC
    print("Running MIMIC Algorithm Experiment")
    start_time = time.time()
    mimic_best_state, mimic_best_fitness, mimic_fitness_curve = mlrose_hiive.mimic(
                                                                problem, 
                                                                max_attempts = 100, 
                                                                max_iters = 100,  
                                                                curve = True, 
                                                                random_state = 42)
    end_time = time.time()
    mimic_time = end_time - start_time
    print("Time (s): {}".format(mimic_time))
    print()

    # Plot Iterations vs Fitness
    iterations = range(1, 1001)
    plt.plot(iterations, rhc_fitness_curve, label='RHC', color='green')
    plt.plot(iterations, sa_fitness_curve, label='SA', color='red')
    plt.plot(iterations, ga_fitness_curve, label='GA', color='blue')
    #plt.plot(iterations, mimic_fitness_curve, label='MIMIC', color='orange')
    plt.legend(loc="best")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.savefig("results/contpeaks_fitness.png")

    # Plot Time Table
    # https://www.geeksforgeeks.org/creating-a-pandas-dataframe-using-list-of-tuples/
    # data = [('RHC', round(rhc_time, 5)), 
    #         ('SA', round(sa_time, 5)), 
    #         ('GA', round(ga_time, 5)), 
    #         ('MIMIC', round(mimic_time, 5))] 
    data = [('RHC', round(rhc_time, 5)), 
            ('SA', round(sa_time, 5)), 
            ('GA', round(ga_time, 5))] 
    df = pd.DataFrame(data, columns =['Algorithm', 'Time (s)']) 
    dfi.export(df,"results/contpeaks_times.png")


    print("Running Experiments for Flip Flop Problem")
    print()

    # Define Fitness function and discrete problem object
    fitness = mlrose.FlipFlop()
    problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True)

    # RHC
    print("Running Random Hill Climb Experiment")
    start_time = time.time()
    rhc_best_state, rhc_best_fitness, rhc_fitness_curve = mlrose_hiive.random_hill_climb(problem, max_attempts = 100, max_iters=100, curve=True, random_state=42)
    end_time = time.time()
    rhc_time = end_time - start_time
    print("Time (s): {}".format(rhc_time))
    print()

    # SA
    print("Running Simulated Annealing Experiment")
    start_time = time.time()
    sa_best_state, sa_best_fitness, sa_fitness_curve = mlrose_hiive.simulated_annealing(
                                                        problem, 
                                                        max_attempts=100, 
                                                        max_iters=100, 
                                                        curve=True, 
                                                        random_state=42)
    end_time = time.time()
    sa_time = end_time - start_time
    print("Time (s): {}".format(sa_time))
    print()

    # GA
    print("Running Genetic Algorithm Experiment")
    start_time = time.time()
    ga_best_state, ga_best_fitness, ga_fitness_curve = mlrose_hiive.genetic_alg(
                                                        problem, 
                                                        max_attempts=100, 
                                                        max_iters=100, 
                                                        curve=True, 
                                                        random_state=42)
    end_time = time.time()
    ga_time = end_time - start_time
    print("Time (s): {}".format(ga_time))
    print()

    # MIMIC
    print("Running MIMIC Algorithm Experiment")
    start_time = time.time()
    mimic_best_state, mimic_best_fitness, mimic_fitness_curve = mlrose_hiive.mimic(
                                                                problem, 
                                                                max_attempts = 100, 
                                                                max_iters = 100,  
                                                                curve = True, 
                                                                pop_size=500,
                                                                keep_pct=0.1,
                                                                random_state = 42)
    end_time = time.time()
    mimic_time = end_time - start_time
    print("Time (s): {}".format(mimic_time))
    print()

    # Plot Iterations vs Fitness
    iterations = range(1, 101)
    plt.plot(iterations, rhc_fitness_curve, label='RHC', color='green')
    plt.plot(iterations, sa_fitness_curve, label='SA', color='red')
    plt.plot(iterations, ga_fitness_curve, label='GA', color='blue')
    plt.plot(iterations, mimic_fitness_curve, label='MIMIC', color='orange')
    plt.legend(loc="best")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.savefig("results/flipflop_fitness.png")

    # Plot Time Table
    # https://www.geeksforgeeks.org/creating-a-pandas-dataframe-using-list-of-tuples/
    data = [('RHC', round(rhc_time, 5)), 
            ('SA', round(sa_time, 5)), 
            ('GA', round(ga_time, 5)), 
            ('MIMIC', round(mimic_time, 5))] 
    
    df = pd.DataFrame(data, columns =['Algorithm', 'Time (s)']) 
    dfi.export(df,"results/flipflop_times.png")

"""
One Max Optimization Problem
    1. Random Hill Climb Experiment
    2. Simulated Annealing Experiment
    3. Genetic Algorithm Experiment
    4. MIMIC Algorithm Experiment <--
"""
def run_one_max():
    print("Running Experiments for One Max")
    print()

    # Define Fitness function and discrete problem object
    fitness = mlrose.OneMax()
    problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True)

    # RHC
    
    print("Running Random Hill Climb Experiment")

    start_time = time.time()
    rhc_best_state, rhc_best_fitness, rhc_fitness_curve = mlrose_hiive.random_hill_climb(problem, max_attempts = 100, max_iters=100, curve=True, random_state=42)
    end_time = time.time()
    rhc_time = end_time - start_time
    print("Time (s): {}".format(rhc_time))
    print()

    # SA
    print("Running Simulated Annealing Experiment")
    start_time = time.time()

    sa_best_state, sa_best_fitness, sa_fitness_curve = mlrose_hiive.simulated_annealing(
                                                        problem, 
                                                        max_attempts=100, 
                                                        max_iters=100, 
                                                        curve=True, 
                                                        random_state=42)
    end_time = time.time()
    sa_time = end_time - start_time
    print("Time (s): {}".format(sa_time))
    print()

    # GA
    print("Running Genetic Algorithm Experiment")
    start_time = time.time()
    ga_best_state, ga_best_fitness, ga_fitness_curve = mlrose_hiive.genetic_alg(
                                                        problem, 
                                                        max_attempts=100, 
                                                        max_iters=100, 
                                                        curve=True, 
                                                        random_state=42)
    end_time = time.time()
    ga_time = end_time - start_time
    print("Time (s): {}".format(ga_time))
    print()

    # MIMIC
    print("Running MIMIC Algorithm Experiment")
    start_time = time.time()
    mimic_best_state, mimic_best_fitness, mimic_fitness_curve = mlrose_hiive.mimic(
                                                                problem, 
                                                                max_attempts = 100, 
                                                                max_iters = 100,  
                                                                curve = True, 
                                                                random_state = 42,
                                                                pop_size=400)
    end_time = time.time()
    mimic_time = end_time - start_time
    print("Time (s): {}".format(mimic_time))
    print()

    # Plot Iterations vs Fitness
    iterations = range(1, 101)
    plt.plot(iterations, rhc_fitness_curve, label='RHC', color='green')
    plt.plot(iterations, sa_fitness_curve, label='SA', color='red')
    plt.plot(iterations, ga_fitness_curve, label='GA', color='blue')
    plt.plot(iterations, mimic_fitness_curve, label='MIMIC', color='orange')
    plt.legend(loc="best")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.savefig("results/onemax_fitness.png")

    # Plot Time Table
    # https://www.geeksforgeeks.org/creating-a-pandas-dataframe-using-list-of-tuples/
    data = [('RHC', round(rhc_time, 5)), 
            ('SA', round(sa_time, 5)), 
            ('GA', round(ga_time, 5)), 
            ('MIMIC', round(mimic_time, 5))] 
    
    df = pd.DataFrame(data, columns =['Algorithm', 'Time (s)']) 
    dfi.export(df,"results/onemax_times.png")

def load_data(dataset):
    # df = pd.read_csv("data/" + dataset, header=None)
    df = pd.read_csv("data/" + dataset)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Dataset Filename: {}".format(dataset))
    print()

    print("Number of features: {}".format(len(df.columns) - 1))
    print()

    print("Total Samples: {}".format(len(df)))
    print()

    return X_train, X_test, y_train, y_test