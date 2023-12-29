from DKP import DKP
from PLS import *
import elicitation as eli
import utils as ut
import PL_optimal as opt
import gurobipy as gp
from time import time
import matplotlib.pyplot as plt
from copy import copy, deepcopy

def run_exp_first_procedure(number_of_items: int, 
                            dimension: int, 
                            file_path: str,
                            number_of_parameters_set: int,
                            size_pop_init: int,
                            struct: str) -> tuple[dict, int]:
    """
    Runs the experiment for the first procedure.
    :param number_of_items: the number of items
    :param dimension: the dimension of the capacities
    :param file_path: the path of the file
    :param number_of_parameters_set: the number of parameters set
    :param size_pop_init: the size of the initial population for the PLS algorithm
    :param struct: the structure used for the PLS algorithm, either "NDTree" or "NDList"
    :return: the results of the experiment and the number of points in the Pareto front approximation
    """

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()

    # Read the instance from a file
    dkp = DKP.from_file(file_path)
    sub_dkp = dkp.subinstance(number_of_items, dimension, shuffle=True)
    
    # Execute the PLS
    X = PLS(sub_dkp, size_pop_init, struct=struct)
    print(len(X))
    results = {"ws": {"times": [], 
                      "nb_questions": [], 
                      "mmr_variations": []}, 
                "owa": {"times": [], 
                        "nb_questions": [], 
                        "mmr_variations": []},
                "choquet": {"times": [], 
                            "nb_questions": [], 
                            "mmr_variations": []}}
    
    for pref_model in results.keys():
        print("Preference model: {}".format(pref_model))
        # Simulate decision makers
        dms = ut.simulate_decision_makers(dimension, 
                                          number_of_parameters_set, 
                                          pref_model=pref_model)
        
        for i, dm in enumerate(dms):
            print("\tDecision maker: {}/{}".format(i+1, len(dms)), end="\r")
            start = time()
            _, nb_questions, mmr_hist = eli.current_solution_strategy(copy(X), dm, pref_model=pref_model, env=env)
            end = time()
            results[pref_model]["times"].append(end - start)
            results[pref_model]["nb_questions"].append(nb_questions)
            results[pref_model]["mmr_variations"].append(mmr_hist)
        print()

    return results, len(X)

def plot_results_first_procedure(results: dict, output_directory: str) -> None:
    """
    Plots the results of the first procedure.
    :param results: the results of the first procedure
    :param output_directory: the output directory
    """
    # 3 pretty colors
    cols = ["indigo", "tomato", "darkgreen"]

    # Plot average time for each preference model on a bar chart
    plt.figure()
    plt.title("Time")
    plt.bar(results.keys(), [np.mean(results[pref_model]["times"]) for pref_model in results.keys()], color=cols)
    plt.savefig(output_directory + "time_first_procedure.png")

    # Plot average number of questions for each preference model on a bar chart
    plt.figure()
    plt.title("Number of questions")
    plt.bar(results.keys(), [np.mean(results[pref_model]["nb_questions"]) for pref_model in results.keys()], color=cols)
    plt.savefig(output_directory + "nb_questions_first_procedure.png")

    # Graph average MMR variations for each preference model
    plt.figure()
    plt.title("MMR variations")
    plt.grid(zorder=0)
    for pref_model, col in zip(results.keys(), cols):
        # find the max number of questions
        max_len = max([len(mmr_variations) for mmr_variations in results[pref_model]["mmr_variations"]])
        # compute average MMR for each number of questions
        mmr_variations_points = np.zeros(max_len)
        mmr_variations_points_count = np.zeros(max_len)
        for mmr_variations in results[pref_model]["mmr_variations"]:
            mmr_variations_points[:len(mmr_variations)] += np.array(mmr_variations)
            mmr_variations_points_count[:len(mmr_variations)] += 1
        mmr_variations_points /= mmr_variations_points_count
        plt.plot(range(max_len), mmr_variations_points, label=pref_model, color=col)
    plt.legend()
    plt.savefig(output_directory + "mmr_variations_first_procedure.png")

    
