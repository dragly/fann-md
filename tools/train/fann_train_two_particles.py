import os
from pylab import *
from os.path import join, abspath, dirname, realpath, exists
from glob import glob
import argparse
from locale import setlocale, LC_ALL
from pyfann import libfann
from fann_converters import convert_three_particle_hdf5_to_fann, convert_two_particle_hdf5_to_fann

setlocale(LC_ALL, "C") # Fixes problems loading fann network on systems with
                       # locales using "," as the punctuation character

parser = argparse.ArgumentParser()
parser.add_argument("states_files", nargs="+")
parser.add_argument("--fast", action="store_true")
parser.add_argument("--min_distance", default=0.0, type=float)
parser.add_argument("--max_distance", default=inf, type=float)
parser.add_argument("--id", nargs='?', default="tmp")
args = parser.parse_args()

output_dir = abspath("tmp/two")

if args.id != "tmp":
    try:
        from sumatra.projects import load_project
        output_dir = join(abspath(load_project().data_store.root), args.id)
    except ImportError:
        pass
    
if not exists(output_dir):
    os.makedirs(output_dir)

states_files = args.states_files
if len(states_files) == 1:
    states_files = glob(states_files[0])

# Convert the files and move them to the build path
if args.fast:
    n_max = 200
else:
    n_max = inf
convert_two_particle_hdf5_to_fann(states_files, output_dir, train_ratio=0.85, n_max=n_max, min_distance=args.min_distance, max_distance=args.max_distance)

# Load data
train_data = libfann.training_data()
validate_data = libfann.training_data()
test_data = libfann.training_data()

train_data_filename = str(join(output_dir, "train.fann"))
validate_data_filename = str(join(output_dir, "validate.fann"))
test_data_filename = str(join(output_dir, "test.fann"))

print "Loading data:\n", train_data_filename, "\n", validate_data_filename, "\n", test_data_filename

train_data.read_train_from_file(train_data_filename)
validate_data.read_train_from_file(validate_data_filename)
test_data.read_train_from_file(test_data_filename)

# Create and train networks
best_test_result = inf
networks = []
for network_count in range(15):
    ann = libfann.neural_net()
    
    ann.set_training_algorithm(libfann.TRAIN_INCREMENTAL)
    
    ann.create_shortcut_array((1,8,8,1))
    ann.set_cascade_weight_multiplier(0.001)
    #ann.create_standard_array((2,5,5,1))
    ann.set_activation_function_hidden(libfann.GAUSSIAN)
    ann.set_activation_function_output(libfann.LINEAR)
    
    network_pre_filename = str(join(output_dir, "fann_network_pre_" + str(network_count) + ".net"))
    best_result = inf
    for i in range(20):
        ann.train_on_data(train_data, 2000, 500, 0.00000001)
        ann.reset_MSE()
        validate_result = ann.test_data(validate_data)
        print "Validation: Best:", best_result, ", current:", validate_result
        if validate_result < best_result:
            best_result = validate_result
            
            ann.save(network_pre_filename)
        else:
            print "Validation: Early stopping!"
            break
    
    network_filename = str(join(output_dir, "fann_network_" + str(network_count) + ".net"))
    networks.append(network_filename)
    for i in range(10):
        ann.cascadetrain_on_data(train_data, 1, 1, 1e-5)
        ann.reset_MSE()
        validate_result = ann.test_data(validate_data)
        print "Validation: Best:", best_result, ", current:", validate_result
        if validate_result < best_result:
            best_result = validate_result
            ann.save(network_filename)
        else:
            print "Validation: Early stopping!"
            break
        
    #Test and compare network with others
    ann.reset_MSE()    
    test_result = ann.test_data(test_data)
    if test_result < best_test_result:
        print "Currently the best. Saving to final file..."
        best_test_result = test_result
        network_filename_final = str(join(output_dir, "fann_network.net"))
        ann.save(network_filename_final)
    ann.destroy()

train_data.destroy_train()
validate_data.destroy_train()
test_data.destroy_train()

print "Lowest validation error was:\n", best_result
print "Lowest test error was:\n", best_test_result
print "Network saved to:\n", network_filename_final
