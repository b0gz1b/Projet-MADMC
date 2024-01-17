from DKP import DKP
from PLS import *
from RBLS import RBLS
import elicitation as eli
import utils as ut
import PL_optimal as opt
import gurobipy as gp
from time import time
import matplotlib.pyplot as plt
from copy import copy
from typing import Dict

class Results:
    def __init__(self, pref_model: str, pareto_front_size: int = 0):
        """
        Constructor of the Results class.
        :param pref_model: the preference model
        :param pareto_front_size: the size of the Pareto front
        """

        self.pref_model = pref_model
        self.pareto_front_size = 0
        self.times = []
        self.nb_questions = []
        self.errors = []
        self.mmr_variations = []

class Experiment:
    def __init__(self, number_of_items: int, dimension: int, file_path: str, number_of_parameters_set: int):
        """
        :param number_of_items: the number of items
        :param dimension: the dimension of the capacities
        :param file_path: the path of the file
        :param number_of_parameters_set: the number of parameters set
        """
        self.number_of_items = number_of_items
        self.dimension = dimension
        self.file_path = file_path
        self.number_of_parameters_set = number_of_parameters_set
        self.env = gp.Env(empty=True)
        self.env.setParam('OutputFlag', 0)
        self.env.start()
        dkp = DKP.from_file(file_path)
        self.sub_dkp = dkp.subinstance(number_of_items, dimension, shuffle=True)
        self.pref_models = ["ws", "owa", "choquet"]

    def run_exp_first_procedure(self, size_pop_init: int, struct: str) -> Dict[str, Results]:
        """
        Runs the experiment for the first procedure.
        :param size_pop_init: the size of the initial population for the PLS algorithm
        :param struct: the structure used for the PLS algorithm, either "NDTree" or "NDList"
        :return: the results of the experiment
        """

        X = PLS(self.sub_dkp, size_pop_init, struct=struct)

        size = len(X)
        print("Size of the Pareto front: {}".format(size))
        res = {pref_model: Results(pref_model, size) for pref_model in self.pref_models}

        for pref_model in self.pref_models:
            print("Preference model: {}".format(pref_model))
            # Simulate decision makers
            dms = ut.simulate_decision_makers(self.dimension, 
                                            self.number_of_parameters_set, 
                                            pref_model=pref_model)

            for i, dm in enumerate(dms):
                print("\tDecision maker: {}/{}".format(i+1, len(dms)))
                start = time()
                x, nb_questions, mmr_hist = eli.current_solution_strategy(copy(X), dm, pref_model=pref_model, env=self.env)
                end = time()
                res[pref_model].times.append(end - start)
                res[pref_model].nb_questions.append(nb_questions)
                res[pref_model].errors.append(opt.opt_decision_maker(self.sub_dkp, dm, pref_model=pref_model, env=self.env)[1].evaluate(dm, pref_model=pref_model) - x.evaluate(dm, pref_model=pref_model))
                res[pref_model].mmr_variations.append(mmr_hist)

        return res
    
    def run_exp_second_procedure(self) -> Dict[str, Results]:
        """
        Runs the experiment for the first procedure.
        :param size_pop_init: the size of the initial population for the PLS algorithm
        :param struct: the structure used for the PLS algorithm, either "NDTree" or "NDList"
        :return: the results of the experiment
        """

        res = {pref_model: Results(pref_model) for pref_model in self.pref_models}

        for pref_model in self.pref_models:
            print("Preference model: {}".format(pref_model))
            # Simulate decision makers
            dms = ut.simulate_decision_makers(self.dimension, 
                                            self.number_of_parameters_set, 
                                            pref_model=pref_model)

            for i, dm in enumerate(dms):
                print("\tDecision maker: {}/{}".format(i+1, len(dms)))
                start = time()
                x, nb_questions, mmr_hist = RBLS(self.sub_dkp,  dm, pref_model=pref_model, env=self.env)
                end = time()
                res[pref_model].times.append(end - start)
                res[pref_model].nb_questions.append(nb_questions)
                res[pref_model].errors.append(opt.opt_decision_maker(self.sub_dkp, dm, pref_model=pref_model, env=self.env)[1].evaluate(dm, pref_model=pref_model) - x.evaluate(dm, pref_model=pref_model))
                res[pref_model].mmr_variations.append(mmr_hist)

        return res

    def plot_results_first_procedure(self, results: Dict[str, Results], output_directory: str) -> None:
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
        plt.bar(self.pref_models, [np.mean(results[pref_model].times) for pref_model in self.pref_models], color=cols)
        plt.savefig(output_directory + "time_first_procedure.png")
        with open(output_directory + "time_first_procedure.out", "w") as f:
            f.write(" ".join(self.pref_models) + "\n")
            for pref_model in self.pref_models:
                f.write("{} ".format(np.mean(results[pref_model].times)))
            f.write("\n")
            for pref_model in self.pref_models:
                f.write("{} ".format(np.std(results[pref_model].times)))
            f.write("\n")


        # Plot average number of questions for each preference model on a bar chart
        plt.figure()
        plt.title("Number of questions")
        plt.bar(results.keys(), [np.mean(results[pref_model].nb_questions) for pref_model in results.keys()], color=cols)
        plt.savefig(output_directory + "nb_questions_first_procedure.png")
        with open(output_directory + "nb_questions_first_procedure.out", "w") as f:
            f.write(" ".join(self.pref_models) + "\n")
            for pref_model in self.pref_models:
                f.write("{} ".format(np.mean(results[pref_model].nb_questions)))
            f.write("\n")
            for pref_model in self.pref_models:
                f.write("{} ".format(np.std(results[pref_model].nb_questions)))
            f.write("\n")


        # Plot average error for each preference model on a bar chart
        plt.figure()
        plt.title("Error to optimal solution")
        plt.bar(results.keys(), [np.mean(results[pref_model].errors) for pref_model in results.keys()], color=cols)
        plt.savefig(output_directory + "error_first_procedure.png")
        with open(output_directory + "error_first_procedure.out", "w") as f:
            f.write(" ".join(self.pref_models) + "\n")
            for pref_model in self.pref_models:
                f.write("{} ".format(np.mean(results[pref_model].errors)))
            f.write("\n")
            for pref_model in self.pref_models:
                f.write("{} ".format(np.std(results[pref_model].errors)))
            f.write("\n")

        # Graph average MMR variations for each preference model
        plt.figure()
        plt.title("MMR variations (averaged)")
        plt.grid(zorder=0)
        for pref_model, col in zip(results.keys(), cols):
            # find the max number of questions
            max_len = max([len(mmr_variations) for mmr_variations in results[pref_model].mmr_variations])
            # compute average MMR for each number of questions
            mmr_variations_points = np.zeros(max_len)
            mmr_variations_points_count = np.zeros(max_len)
            for mmr_variations in results[pref_model].mmr_variations:
                mmr_variations_points[:len(mmr_variations)] += np.array(mmr_variations)
                mmr_variations_points_count[:len(mmr_variations)] += 1
            mmr_variations_points /= mmr_variations_points_count
            plt.plot(range(max_len), mmr_variations_points, label=pref_model, color=col)
        plt.xlabel("Number of questions")
        plt.ylabel("Average Minimax Regret (MMR)")
        plt.legend()
        plt.savefig(output_directory + "mmr_variations_average_first_procedure.png")

        for pref_model in results.keys():
            plt.figure()
            regrets = results[pref_model].mmr_variations
            for mmrhist in regrets:
                plt.plot(mmrhist, "o", alpha=0.3, linestyle="dashed")
            plt.title("MMR variations for {}".format(pref_model))
            plt.xlabel("Number of questions")
            plt.ylabel("Minimax Regret (MMR)")
            plt.savefig(output_directory + "mmr_variations_{}_first_procedure.png".format(pref_model))

    def plot_results(self, results1: Dict[str, Results], results2: Dict[str, Results], output_directory: str) -> None:
        """
        Plots the results of the first procedure.
        :param results: the results of the first procedure
        :param output_directory: the output directory
        """
        # 3 pretty colors
        cols = ["indigo", "tomato", "darkgreen"]

        # Plot average time for each preference model on a bar chart
        # plt.title("Time")
        x = np.arange(len(self.pref_models))
        width = 0.25
        fig, ax = plt.subplots(layout='constrained')
        
        rects = ax.bar(x, [np.mean(results1[pref_model].times) for pref_model in self.pref_models], width, label="¨P1")
        ax.bar_label(rects, padding=3)
        rects = ax.bar(x + width, [np.mean(results2[pref_model].times) for pref_model in self.pref_models], width, label="¨P2")
        ax.bar_label(rects, padding=3)

        ax.set_title('Time(s)')
        ax.set_xticks(x + width, self.pref_models)
        ax.legend(loc='upper left', ncols=2)

        plt.savefig(output_directory + "time_second_procedure.png")
        with open(output_directory + "time_second_procedure.out", "w") as f:
            f.write(" ".join(self.pref_models) + "\n")
            for pref_model in self.pref_models:
                f.write("{} ".format(np.mean(results1[pref_model].times)))
            f.write("\n")
            for pref_model in self.pref_models:
                f.write("{} ".format(np.std(results1[pref_model].times)))
            f.write("\n")
            
            f.write(" ".join(self.pref_models) + "\n")
            for pref_model in self.pref_models:
                f.write("{} ".format(np.mean(results2[pref_model].times)))
            f.write("\n")
            for pref_model in self.pref_models:
                f.write("{} ".format(np.std(results2[pref_model].times)))
            f.write("\n")


        # Plot average number of questions for each preference model on a bar chart
        x = np.arange(len(self.pref_models))
        width = 0.25
        fig, ax = plt.subplots(layout='constrained')
        
        rects = ax.bar(x, [np.mean(results1[pref_model].nb_questions) for pref_model in self.pref_models], width, label="¨P1")
        ax.bar_label(rects, padding=3)
        rects = ax.bar(x + width, [np.mean(results2[pref_model].nb_questions) for pref_model in self.pref_models], width, label="¨P2")
        ax.bar_label(rects, padding=3)
        
        ax.set_title('Number of questions')
        ax.set_xticks(x + width, self.pref_models)
        ax.legend(loc='upper left', ncols=2)
        plt.savefig(output_directory + "nb_questions_second_procedure.png")
        with open(output_directory + "nb_questions_second_procedure.out", "w") as f:
            f.write(" ".join(self.pref_models) + "\n")
            for pref_model in self.pref_models:
                f.write("{} ".format(np.mean(results1[pref_model].nb_questions)))
            f.write("\n")
            for pref_model in self.pref_models:
                f.write("{} ".format(np.std(results1[pref_model].nb_questions)))
            f.write("\n")
            
            f.write(" ".join(self.pref_models) + "\n")
            for pref_model in self.pref_models:
                f.write("{} ".format(np.mean(results2[pref_model].nb_questions)))
            f.write("\n")
            for pref_model in self.pref_models:
                f.write("{} ".format(np.std(results2[pref_model].nb_questions)))
            f.write("\n")


        # Plot average error for each preference model on a bar chart
        
        x = np.arange(len(self.pref_models))
        width = 0.25
        fig, ax = plt.subplots(layout='constrained')
        
        rects = ax.bar(x, [np.mean(results1[pref_model].errors) for pref_model in self.pref_models], width, label="¨P1")
        ax.bar_label(rects, padding=3)
        rects = ax.bar(x + width, [np.mean(results2[pref_model].errors) for pref_model in self.pref_models], width, label="¨P2")
        ax.bar_label(rects, padding=3)
        

        ax.set_title('Error to optimal solution')
        ax.set_xticks(x + width, self.pref_models)
        ax.legend(loc='upper left', ncols=2)
        plt.savefig(output_directory + "error_second_procedure.png")
        with open(output_directory + "error_second_procedure.out", "w") as f:
            f.write(" ".join(self.pref_models) + "\n")
            for pref_model in self.pref_models:
                f.write("{} ".format(np.mean(results1[pref_model].errors)))
            f.write("\n")
            for pref_model in self.pref_models:
                f.write("{} ".format(np.std(results1[pref_model].errors)))
            f.write("\n")
            
            f.write(" ".join(self.pref_models) + "\n")
            for pref_model in self.pref_models:
                f.write("{} ".format(np.mean(results2[pref_model].errors)))
            f.write("\n")
            for pref_model in self.pref_models:
                f.write("{} ".format(np.std(results2[pref_model].errors)))
            f.write("\n")

        # Graph average MMR variations for each preference model
        plt.figure()
        plt.title("MMR variations (averaged)")
        plt.grid(zorder=0)
        for pref_model, col in zip(results1.keys(), cols):
            # find the max number of questions
            max_len = max([len(mmr_variations) for mmr_variations in results1[pref_model].mmr_variations])
            # compute average MMR for each number of questions
            mmr_variations_points = np.zeros(max_len)
            mmr_variations_points_count = np.zeros(max_len)
            for mmr_variations in results1[pref_model].mmr_variations:
                mmr_variations_points[:len(mmr_variations)] += np.array(mmr_variations)
                mmr_variations_points_count[:len(mmr_variations)] += 1
            mmr_variations_points /= mmr_variations_points_count
            plt.plot(range(max_len), mmr_variations_points, label=pref_model+"_1", color=col)
        cols = ["blue", "red", "green"]
        for pref_model, col in zip(results2.keys(), cols):
            # find the max number of questions
            max_len = max([len(mmr_variations) for mmr_variations in results2[pref_model].mmr_variations])
            # compute average MMR for each number of questions
            mmr_variations_points = np.zeros(max_len)
            mmr_variations_points_count = np.zeros(max_len)
            for mmr_variations in results2[pref_model].mmr_variations:
                mmr_variations_points[:len(mmr_variations)] += np.array(mmr_variations)
                mmr_variations_points_count[:len(mmr_variations)] += 1
            mmr_variations_points /= mmr_variations_points_count
            plt.plot(range(max_len), mmr_variations_points, label=pref_model+"_2", color=col)
        plt.xlabel("Number of questions")
        plt.ylabel("Average Minimax Regret (MMR)")
        plt.legend()
        plt.savefig(output_directory + "mmr_variations_average_comparaison.png")

        for pref_model in results1.keys():
            plt.figure()
            regrets1 = results1[pref_model].mmr_variations
            regrets2 = results1[pref_model].mmr_variations
            for mmrhist in regrets1:
                plt.plot(mmrhist, "o", alpha=0.3, label="P1", linestyle="dashed", color="red")
            for mmrhist in regrets2:
                plt.plot(mmrhist, "o", alpha=0.3, label="P2", linestyle="dashed", color ="blue")
            plt.title("MMR variations for {}".format(pref_model))
            plt.xlabel("Number of questions")
            plt.ylabel("Minimax Regret (MMR)")
            plt.legend()
            plt.savefig(output_directory + "mmr_variations_{}_comparaison.png".format(pref_model))

    
