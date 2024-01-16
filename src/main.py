import experiments as xp

if __name__ == '__main__':
	experiment1 = xp.Experiment(number_of_items = 20, 
							 dimension = 3, 
							 file_path = "../data/2KP200-TA-0.dat", 
							 number_of_parameters_set = 1)

	# results = experiment1.run_exp_first_procedure(size_pop_init = 20, struct = "NDTree")
	
	# experiment1.plot_results_first_procedure(results, "../out/partie_1/")
 
	results2 = experiment1.run_exp_second_procedure()
	
	experiment1.plot_results_second_procedure(results2, "../out/partie_2/")