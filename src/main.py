import experiments as xp

if __name__ == '__main__':
	experiment1 = xp.Experiment(number_of_items = 30, 
							 dimension = 5, 
							 file_path = "data/2KP200-TA-0.dat", 
							 number_of_parameters_set = 20)

	# results1 = experiment1.run_exp_first_procedure(size_pop_init = 5, struct = "NDTree")
	
	# experiment1.plot_results_first_procedure(results, "../out/partie_1/")
 
	results1, results2 = experiment1.run_exp_all_procedure(size_pop_init = 100, struct = "NDTree")
	
	experiment1.plot_results(results1, results2, "out/comparaison/")
