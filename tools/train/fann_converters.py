import h5py
import numpy
from os.path import join
from numpy import dtype, zeros, linspace, pi, cos, sin, arctan, arctan2, sqrt, meshgrid, array, inf
from pylab import imshow, plot, figure, subplot, title
from glob import glob
from scipy.interpolate import griddata
from collections import OrderedDict
from sys import argv
from random import shuffle

def rescale(value, value_min, value_max):
    #return value
    return (value - value_min) / (value_max - value_min) * 0.8 + 0.1

def convert_two_particle_hdf5_to_fann(filenames, output_dir, n_max=inf, train_ratio=0.8):    
    if train_ratio >= 1.0 or train_ratio <= 0.0:
        raise Exception("train_ratio must be in range [0.0, 1.0]. Got " + str(train_ratio))
    
    train_file = open(join(output_dir, "train.fann"), "w")
    validate_file = open(join(output_dir, "validate.fann"), "w")
    test_file = open(join(output_dir, "test.fann"), "w")
    bounds_file = open(join(output_dir, "bounds.fann"), "w")
    
    n_states_total = 0
    n_parameters = 1
    n_outputs = 1
    
    energy_min = inf
    energy_max = -inf
    r12_min = inf
    r12_max = -inf
    
    all_states = []
    
    for statesFile in filenames:
        f = h5py.File(statesFile, "r")
        atomsMeta = f.get("atomMeta")
        r12_min = atomsMeta.attrs["r12Min"]
        r12_max = atomsMeta.attrs["r12Max"]
        states = f.get("/states")
        n_states_total += len(states)
        for stateName in states:
            atoms = states.get(stateName)
            energy_min = min(energy_min, atoms.attrs["energy"])
            energy_max = max(energy_max, atoms.attrs["energy"])
            all_states.append([atoms.attrs["r12"], atoms.attrs["energy"]])
            
        f.close()
    n_states_total = min(n_states_total, n_max)
    
    bounds_file.write("%d %d\n" % (n_parameters, n_outputs))
    bounds_file.write("%.16e %.16e\n" % (r12_min, r12_max))
    bounds_file.write("%.16e %.16e\n" % (energy_min, energy_max))
    
    bounds_file.close()
    
    n_states_train = int(train_ratio * n_states_total)
    n_states_validate = n_states_total - n_states_train
    n_states_test = n_states_total
    
    print "Using", n_states_train, " of the states for training and", n_states_validate, "for validation"
    print "All states are used for testing"
        
    train_file.write("%d %d %d\n\n" % (n_states_train, n_parameters, n_outputs))
    validate_file.write("%d %d %d\n\n" % (n_states_validate, n_parameters, n_outputs))
    test_file.write("%d %d %d\n\n" % (n_states_test, n_parameters, n_outputs))
    
    state_counter = 0
    
    shuffle(all_states)
    
    for state in all_states:
        if state_counter > n_max:
            break
        if state_counter < n_states_train:
            target_file = train_file
        else:
            target_file = validate_file
            
        atoms = states.get(stateName)
        
        r12 = rescale(state[0], r12_min, r12_max)
        
        energy = rescale(state[1], energy_min, energy_max)
        
        target_file.write("%.10f\n\n" % (r12))
        target_file.write("%.10f\n\n" % (energy))
        
        test_file.write("%.10f\n\n" % (r12))
        test_file.write("%.10f\n\n" % (energy))
        state_counter += 1
        
    train_file.close()
    test_file.close()
    
def convert_three_particle_hdf5_to_fann(filename, n_max=inf):
    train_file = open("train.fann", "w")
    test_file = open("test.fann", "w")
    bounds_file = open("bounds.fann", "w")
    
    n_states_total = 0
    n_parameters = 3
    n_outputs = 1
    
    energy_min = inf
    energy_max = -inf
    r12_min = inf
    r12_max = -inf
    r13_min = inf
    r13_max = -inf
    angle_min = inf
    angle_max = -inf
    
    for statesFile in filenames:
        f = h5py.File(statesFile, "r")
        atomsMeta = f.get("atomMeta")
        r12_min = atomsMeta.attrs["r12Min"]
        r12_max = atomsMeta.attrs["r12Max"]
        r13_min = atomsMeta.attrs["r13Min"]
        r13_max = atomsMeta.attrs["r13Max"]
        angle_min = atomsMeta.attrs["angleMin"]
        angle_max = atomsMeta.attrs["angleMax"]
        states = f.get("/states")
        n_states_total += len(states)
        for stateName in states:
            atoms = states.get(stateName)
            energy_min = min(energy_min, atoms.attrs["energy"])
            energy_max = max(energy_max, atoms.attrs["energy"])
            
        f.close()
        
    n_states_total = min(n_states_total, n_max)
    
    bounds_file.write("%d %d\n" % (n_parameters, n_outputs))
    bounds_file.write("%.16e %.16e\n" % (r12_min, r12_max))
    bounds_file.write("%.16e %.16e\n" % (r13_min, r13_max))
    bounds_file.write("%.16e %.16e\n" % (angle_min, angle_max))
    bounds_file.write("%.16e %.16e\n" % (energy_min, energy_max))
    
    bounds_file.close()
    
    n_states_train = int(0.8 * n_states_total)
    n_states_test = n_states_total - n_states_train
    
    print "Using", n_states_train, " of the states for training and", n_states_test, "for testing"
        
    train_file.write("%d %d %d\n\n" % (n_states_train, n_parameters, n_outputs))
    test_file.write("%d %d %d\n\n" % (n_states_test, n_parameters, n_outputs))
    
    state_counter = 0
    
    for statesFile in filenames:
        f = h5py.File(statesFile, "r")
        atomsMeta = f.get("atomMeta")
        states = f.get("/states")
        stateNames = []
        
        for stateName in states:
            stateNames.append(stateName)
        
        # Shuffle to get a random order
        shuffle(stateNames)
        
        for stateName in stateNames:
            if state_counter > n_max:
                break
            if state_counter < n_states_train:
                target_file = train_file
            else:
                target_file = test_file
                
            atoms = states.get(stateName)
            r12 = rescale(atoms.attrs["r12"], r12_min, r12_max)
            r13 = rescale(atoms.attrs["r13"], r13_min, r13_max)
            
            dx = atoms[2]["x"] - atoms[1]["x"]
            dy = atoms[2]["y"] - atoms[1]["y"]
            dz = atoms[2]["z"] - atoms[1]["z"]
            
            r23 = sqrt(dx*dx + dy*dy + dz*dz)
            
            r23 = rescale(r23, 0, 12)
            
            angle = rescale(atoms.attrs["angle"], angle_min, angle_max)
            energy = rescale(atoms.attrs["energy"], energy_min, energy_max)
            
            target_file.write("%.10f %.10f %.10f\n\n" % (r12, r13, angle))
            #target_file.write("%.10f %.10f %.10f %.10f\n\n" % (r12, r13, angle, r23))
            target_file.write("%.10f\n\n" % (energy))
            state_counter += 1
            
        f.close()
        
    train_file.close()
    test_file.close()