import h5py
import numpy
from os.path import join
from numpy import dtype, zeros, linspace, pi, cos, sin, arctan, arctan2, sqrt, meshgrid, array, inf
from pylab import *
from glob import glob
from scipy.interpolate import griddata
from collections import OrderedDict
from sys import argv
from random import shuffle
from pyfann import libfann
from glob import glob

def find_nearest_index(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx

def rescale(value, value_min, value_max):
    #return value
    return (value - value_min) / (value_max - value_min) * 0.8 + 0.1

def rescale_inverse(value, value_min, value_max):
    return (value - 0.1) / 0.8 * (value_max - value_min) + value_min

def convert_two_particle_hdf5_to_fann(filenames, output_dir, n_max=inf, train_ratio=0.8, min_distance=0.0):    
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
        energyOffset = atomsMeta.attrs["energyOffset"]
        states = f.get("/states")
        for stateName in states:
            atoms = states.get(stateName)
            r12 = atoms.attrs["r12"]
            if r12 < min_distance:
                continue
            energy = atoms.attrs["energy"] - energyOffset
            energy_min = min(energy_min, energy)
            energy_max = max(energy_max, energy)
            all_states.append([r12, energy])
            
        f.close()
    n_states_total = min(len(all_states), n_max)
    
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
    
def convert_three_particle_hdf5_to_fann(filenames, r12_filenames, r13_filenames, r23_filenames,  output_dir, n_max=inf, train_ratio=0.8, min_distance=0.0):
    
    if len(filenames) == 1:
        filenames = glob(filenames[0])
        
    if len(r12_filenames) == 1:
        r12_filenames = glob(r12_filenames[0])
        
    if len(r13_filenames) == 1:
        r13_filenames = glob(r13_filenames[0])
        
    if len(r23_filenames) == 1:
        r23_filenames = glob(r23_filenames[0])  
    
    if train_ratio >= 1.0 or train_ratio <= 0.0:
        raise Exception("train_ratio must be in range [0.0, 1.0]. Got " + str(train_ratio))
        
    r12_energies = []
    r12_r12s = []
    r13_energies = []
    r13_r12s = []
    r23_energies = []
    r23_r12s = []
    
    for i in range(3):
        if i == 0:
            states_files = r12_filenames
            r12s = r12_r12s
            energies = r12_energies
        elif i == 1:
            states_files = r13_filenames
            r12s = r13_r12s
            energies = r13_energies
        elif i == 2:
            states_files = r23_filenames
            r12s = r23_r12s
            energies = r23_energies
            
        print "Reading", len(states_files), "state files"
        for statesFile in states_files:
            f = h5py.File(statesFile, "r")
            atomsMeta = f.get("atomMeta")
            energyOffset = atomsMeta.attrs["energyOffset"]
            if len(atomsMeta) != 2:
                raise Exception("Wrong number of atoms in atomsMeta. Found %d, should be 2." % len(atomsMeta))
            states = f.get("/states")
            print "Reading", len(states), "states..."
            for stateName in states:
                atoms = states.get(stateName)
                r12 = atoms.attrs["r12"]
                energy = atoms.attrs["energy"] - energyOffset
                
                r12s.append(r12)
                energies.append(energy)                
            f.close()
        
    train_file = open(join(output_dir, "train.fann"), "w")
    validate_file = open(join(output_dir, "validate.fann"), "w")
    test_file = open(join(output_dir, "test.fann"), "w")
    bounds_file = open(join(output_dir, "bounds.fann"), "w")
    
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
    
    all_states = []
    
    print "Reading", len(filenames), "state files"
    for statesFile in filenames:
        f = h5py.File(statesFile, "r")
        atomsMeta = f.get("atomMeta")
        energy_offset = atomsMeta.attrs["energyOffset"] # energy of the system with atoms infinitely far away from each other
        r12_min = atomsMeta.attrs["r12Min"]
        r12_max = atomsMeta.attrs["r12Max"]
        r13_min = atomsMeta.attrs["r13Min"]
        r13_max = atomsMeta.attrs["r13Max"]
        angle_min = atomsMeta.attrs["angleMin"]
        angle_max = atomsMeta.attrs["angleMax"]
        states = f.get("/states")
        print "Reading", len(states), "states..."
        for stateName in states:
            atoms = states.get(stateName)
            
            r12 = atoms.attrs["r12"]
            r13 = atoms.attrs["r13"]
            angle = atoms.attrs["angle"]
            
            if r12 < min_distance and r13 < min_distance:
                continue
            
            x = atoms[1][0] - atoms[2][0]
            y = atoms[1][1] - atoms[2][1]            
            z = atoms[1][2] - atoms[2][2]
            
            r23 = sqrt(x*x + y*y + z*z)
        
            index12 = find_nearest_index(r12_r12s, r12)
            index13 = find_nearest_index(r13_r12s, r13)
            index23 = find_nearest_index(r23_r12s, r23)
        
            r12_energy = r12_energies[index12]
            r13_energy = r13_energies[index13]
            r23_energy = r23_energies[index23]
            
            energy = (atoms.attrs["energy"] - energy_offset - r12_energy - r13_energy - r23_energy)
            
            energy_min = min(energy_min, energy)
            energy_max = max(energy_max, energy)
            #energy_min = min(energy_min, atoms.attrs["energy"])
            #energy_max = max(energy_max, atoms.attrs["energy"])
            
            all_states.append([r12, r13, angle, r23, energy])
            
        f.close()
        
    n_states_total = min(len(all_states), n_max)
    
    bounds_file.write("%d %d\n" % (n_parameters, n_outputs))
    bounds_file.write("%.16e %.16e\n" % (r12_min, r12_max))
    bounds_file.write("%.16e %.16e\n" % (r13_min, r13_max))
    bounds_file.write("%.16e %.16e\n" % (angle_min, angle_max))

    print "energy_min: ", energy_min, " energy_max: ", energy_max
    
    bounds_file.write("%.16e %.16e\n" % (energy_min, energy_max))
    
    bounds_file.close()
    
    n_states_train = int(train_ratio * n_states_total)
    n_states_validate = n_states_total - n_states_train
    n_states_test = n_states_total
    
    print "Using", n_states_train, " of the states for training and", n_states_validate, "for validation"
        
    train_file.write("%d %d %d\n\n" % (n_states_train, n_parameters, n_outputs))
    validate_file.write("%d %d %d\n\n" % (n_states_validate, n_parameters, n_outputs))
    test_file.write("%d %d %d\n\n" % (n_states_test, n_parameters, n_outputs))
    
    shuffle(all_states)
    
    state_counter = 0        
    for state in all_states:
        if state_counter > n_max:
            break
        if state_counter < n_states_train:
            target_file = train_file
        else:
            target_file = validate_file
            
        r12 = rescale(state[0], r12_min, r12_max)
        r13 = rescale(state[1], r13_min, r13_max)        
        angle = rescale(state[2], angle_min, angle_max)
        
#        r12_energy = r12_network.run([rescale(state[0], r12_network_min, r12_network_max)])[0]
#        r12_energy = rescale_inverse(r12_energy, r12_energy_min, r12_energy_max)
#        r13_energy = r13_network.run([rescale(state[1], r13_network_min, r13_network_max)])[0]
#        r13_energy = rescale_inverse(r13_energy, r13_energy_min, r13_energy_max)
#        r23_energy = r23_network.run([rescale(state[3], r23_network_min, r23_network_max)])[0]
#        r23_energy = rescale_inverse(r23_energy, r23_energy_min, r23_energy_max)

        # TODO Fix this scaling
        energy = state[4]
        #energy = state[4]
        
        energy = rescale(energy, energy_min, energy_max)
        
        target_file.write("%.10f %.10f %.10f\n\n" % (r12, r13, angle))
        target_file.write("%.10f\n\n" % (energy))
        
        test_file.write("%.10f %.10f %.10f\n\n" % (r12, r13, angle))
        test_file.write("%.10f\n\n" % (energy))
        state_counter += 1
        
    train_file.close()
    test_file.close()
    