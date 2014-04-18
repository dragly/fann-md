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
parser.add_argument("--id", nargs='?', default="tmp")
args = parser.parse_args()

output_dir = abspath("tmp")

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
convert_two_particle_hdf5_to_fann(states_files, output_dir, n_max=10000)

ann = libfann.neural_net()

#ann.create_shortcut_array((2,10,1))
#ann.set_cascade_weight_multiplier(0.001)
ann.create_standard_array((2,5,5,1))
ann.set_training_algorithm(libfann.TRAIN_RPROP)
ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC)
ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC)

train_data = libfann.training_data()
test_data = libfann.training_data()

train_data_filename = str(join(output_dir, "train.fann"))
test_data_filename = str(join(output_dir, "test.fann"))
print "Loading data:\n", train_data_filename, "\n", test_data_filename
train_data.read_train_from_file(train_data_filename)
test_data.read_train_from_file(test_data_filename)

best_result = inf
#for i in range(5):
    #ann.cascadetrain_on_data(train_data, 1, 1, 1e-5)
ann.train_on_data(train_data, 100000, 1000, 0.00001)

network_filename = str(join(output_dir, "fann_network.net"))
ann.save(network_filename)

train_data.destroy_train()
test_data.destroy_train()
ann.destroy()
