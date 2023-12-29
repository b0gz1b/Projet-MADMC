from experiments import *

if __name__ == '__main__':
	results, pf_size = run_exp_first_procedure(number_of_items = 20, 
						    		  dimension = 4, 
									  file_path = "data/2KP200-TA-0.dat", 
							          number_of_parameters_set = 50,
							          size_pop_init = 10, 
							          struct = "NDTree")
	
	print("Pareto front size: {}".format(pf_size))
	
	plot_results_first_procedure(results, "out/partie_1/")