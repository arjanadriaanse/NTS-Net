import os 
from config import accuracy_text_location
small_sizes_grid = [36,48, 60]
medium_sizes_grid = [84,96, 108]
large_sizes_grid = [180,192,204]

best_accuracy = 0.0 #The initial value
best_small  = -1
best_medium = -1
best_large  = -1

def read_accuracy():
	file = open(os.path.join(accuracy_text_location, "accuracies.txt"),"r")
	acc_string = file.read()
	if acc_string == "":
		return 0.0
	else:
		return float(acc_string)

def write_accuracy_for_config(small, medium, large, accuracy):
    file = acc_file = open(os.path.join(accuracy_text_location, "all_accuracies.txt"),"a+")
    acc_string = "s: " + str(small) + "; m: " + str(medium) + "; l: " + str(large) + "; accuracy: " + str(accuracy) + "\n"
    file.write(acc_string)
    file.close
    

#Call train with all possible combinations of anchors
open(os.path.join(accuracy_text_location, "all_accuracies.txt"), 'w').close()
for small_scale in small_sizes_grid:
	for medium_scale in medium_sizes_grid:
		for large_scale in large_sizes_grid:
			print("training : " + str(small_scale) + ' ' + str(medium_scale) + ' ' + str(large_scale))
			#Training for current config
			os.system('python3 train.py ' + str(small_scale) + ' ' + str(medium_scale) + ' ' + str(large_scale))
			print("done training")
			#Test for currently generated model
			model_accuracy = read_accuracy()
			write_accuracy_for_config(small_scale, medium_scale, large_scale, model_accuracy)
			if (model_accuracy > best_accuracy):
				best_accuracy = model_accuracy
				best_small = small_scale
				best_medium = medium_scale
				best_large = large_scale

write_accuracy_for_config(best_small, best_medium, best_large, best_accuracy)



