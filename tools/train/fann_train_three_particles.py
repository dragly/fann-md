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
parser.add_argument("--three_body_states", nargs="+")
parser.add_argument("--r12_states", nargs="+")
parser.add_argument("--r13_states", nargs="+")
parser.add_argument("--r23_states", nargs="+")
parser.add_argument("--min_distance", default=0.0, type=float)
parser.add_argument("--fast", action="store_true")
parser.add_argument("--id", nargs='?', default="tmp")
args = parser.parse_args()

output_dir = abspath("tmp/three")

if args.id != "tmp":
    try:
        from sumatra.projects import load_project
        output_dir = join(abspath(load_project().data_store.root), args.id)
    except ImportError:
        pass
    
if not exists(output_dir):
    os.makedirs(output_dir)

# Convert the files and move them to the build path
if args.fast:
    n_max = 1000
else:
    n_max = inf
convert_three_particle_hdf5_to_fann(args.three_body_states, 
                                    args.r12_states, args.r13_states, args.r23_states, 
                                    output_dir, train_ratio=0.85, n_max=n_max, min_distance=args.min_distance)

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
networks = []
for network_count in range(10):
    ann = libfann.neural_net()
    
    ann.set_training_algorithm(libfann.TRAIN_RPROP)
    
    ann.create_shortcut_array((3,8,8,1))
    ann.set_cascade_weight_multiplier(0.001)
    #ann.create_standard_array((2,5,5,1))
    ann.set_activation_function_hidden(libfann.GAUSSIAN)
    ann.set_activation_function_output(libfann.ELLIOT_SYMMETRIC)
    
    
    network_filename = str(join(output_dir, "fann_network_" + str(network_count) + ".net"))
    networks.append(network_filename)
    network_pre_filename = str(join(output_dir, "fann_network_pre_" + str(network_count) + ".net"))
    best_result = inf
    for i in range(20):
        ann.train_on_data(train_data, 2000, 500, 0.00001)
        ann.reset_MSE()
        validate_result = ann.test_data(validate_data)
        print "Validation: Best:", best_result, ", current:", validate_result
        if validate_result < best_result:
            best_result = validate_result
            ann.save(network_filename)
            ann.save(network_pre_filename)
        elif validate_result < best_result*1.01:
            print "Validation nearly bad..."
        else:
            print "Validation: Early stopping!"
            break
    for i in range(10):
        ann.cascadetrain_on_data(train_data, 1, 1, 1e-5)
        ann.reset_MSE()
        validate_result = ann.test_data(validate_data)
        print "Validation: Best:", best_result, ", current:", validate_result
        if validate_result < best_result:
            best_result = validate_result
            ann.save(network_filename)
        elif validate_result < best_result*1.01:
            print "Validation nearly bad..."
        else:
            print "Validation: Early stopping!"
            break
    ann.destroy()

# Network comparison
best_test_result = inf
for network_filename in networks:
    ann = libfann.neural_net()
    ann.create_from_file(network_filename)
    ann.reset_MSE()
    test_result = ann.test_data(test_data)
    print "Final results for network", network_filename, ":\n", test_result
    if test_result < best_test_result:
        print "Currently the best. Saving..."
        best_test_result = test_result
        network_filename_final = str(join(output_dir, "fann_network.net"))
        ann.save(network_filename_final)
    ann.destroy()

train_data.destroy_train()
validate_data.destroy_train()
test_data.destroy_train()

print "Network saved to:\n", network_filename_final
