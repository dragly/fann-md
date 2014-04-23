#include <iostream>
#include <iomanip>
#include <fstream>
#include <armadillo>
#include <doublefann.h>

using namespace std;
using arma::vec;
using arma::linspace;
using arma::mat;
using arma::zeros;


double rescale(double value, double valueMin, double valueMax) {
    return (value - valueMin) / (valueMax - valueMin) * 0.8 + 0.1;
}

void createTwoParticlePlots(const mat &bounds) {
    int num_input = 1;
    int num_output = 1;

    vec r12s = linspace(bounds(0,0), bounds(0,1), 200);
    struct fann *ann;
    string inFileName = "/home/svenni/Dropbox/studies/master/results/fann_train/20140418-165800/fann_network.net";

    ann = fann_create_from_file(inFileName.c_str());

    stringstream energiesOutFileName;
    energiesOutFileName << "energies.dat";
    ofstream energiesOutFile(energiesOutFileName.str());
    for(int i = 0; i < int(r12s.n_elem); i++) {
        double r12 = r12s[i];
        fann_type *output;
        fann_type input[num_input];
        input[0] = rescale(r12, r12s.min(), r12s.max());
        output = fann_run(ann, input);
        double energy = output[0];
        energiesOutFile << r12 << " " << energy << " " << endl;
        cout << r12 << " " << energy << endl;
    }
    energiesOutFile.close();
    fann_destroy(ann);
}

void createThreeParticlePlots(const mat &bounds) {
    int num_input = 3;
    int num_output = 1;

    vec r12s = linspace(bounds(0,0), bounds(0,1), 7);
    vec r13s = linspace(bounds(1,0), bounds(1,1), 100);
    vec angles = linspace(bounds(2,0), bounds(2,1), 100);
    struct fann *ann;
    string inFileName = "../fann-trainer/fann_network0.net";

    ann = fann_create_from_file(inFileName.c_str());

    for(int i = 0; i < int(r12s.n_elem); i++) {
        cout << double(i) / double(r12s.n_elem) * 100 << "%" << endl;
        double r12 = r12s[i];
        stringstream energiesOutFileName;
        energiesOutFileName << "energies_" << std::fixed << setprecision(4) << r12 << ".dat";
        ofstream energiesOutFile(energiesOutFileName.str());

        stringstream r12DerivativesOutFileName;
        r12DerivativesOutFileName << "derivativesr12_" << std::fixed << setprecision(4) << r12 << ".dat";
        ofstream r12DerivativesOutFile(r12DerivativesOutFileName.str());

        stringstream r13DerivativesOutFileName;
        r13DerivativesOutFileName << "derivativesr13_" << std::fixed << setprecision(4) << r12 << ".dat";
        ofstream r13DerivativesOutFile(r13DerivativesOutFileName.str());

        stringstream angleDerivativesOutFileName;
        angleDerivativesOutFileName << "derivativesrangle_" << std::fixed << setprecision(4) << r12 << ".dat";
        ofstream angleDerivativesOutFile(angleDerivativesOutFileName.str());

        fann_type *output;
        for (int j = 0; j < int(r13s.n_elem); ++j) {
            double r13 = r13s[j];
            for (int k = 0; k < int(angles.n_elem); ++k) {
                double angle = angles[k];

                //                double r23 = sqrt(r12*r12 + r13*r13 - 2*r12*r13*cos(angle));

                fann_type input[num_input];
                input[0] = rescale(r12, r12s.min(), r12s.max());
                input[1] = rescale(r13, r13s.min(), r13s.max());
                input[2] = rescale(angle, angles.min(), angles.max());
                //                input[3] = rescale(r23, 0, 12);
                //                output = fann_run_diff(ann, input);
                output = fann_run(ann, input);
                double energy = output[0];
                energiesOutFile << energy << " ";

                double energyMinus = 0;
                double energyPlus = 0;
                double derivative = 0;
                double h = 1e-8;

                input[0] = rescale(r12 + h, r12s.min(), r12s.max());
                output = fann_run(ann, input);
                energyPlus = output[0];
                input[0] = rescale(r12 - h, r12s.min(), r12s.max());
                output = fann_run(ann, input);
                energyMinus = output[0];
                derivative = (energyPlus - energyMinus) / (2 * h);
                r12DerivativesOutFile << derivative << " ";

                input[1] = rescale(r13 + h, r13s.min(), r13s.max());
                output = fann_run(ann, input);
                energyPlus = output[0];
                input[1] = rescale(r13 - h, r13s.min(), r13s.max());
                output = fann_run(ann, input);
                energyMinus = output[0];
                derivative = (energyPlus - energyMinus) / (2 * h);
                r13DerivativesOutFile << derivative << " ";

                input[2] = rescale(angle + h, angles.min(), angles.max());
                output = fann_run(ann, input);
                energyPlus = output[0];
                input[2] = rescale(angle - h, angles.min(), angles.max());
                output = fann_run(ann, input);
                energyMinus = output[0];
                derivative = (energyPlus - energyMinus) / (2 * h);
                angleDerivativesOutFile << derivative << " ";
            }
            energiesOutFile << "\n";
            r12DerivativesOutFile << "\n";
            r13DerivativesOutFile << "\n";
            angleDerivativesOutFile << "\n";
        }
        energiesOutFile.close();
        r12DerivativesOutFile.close();
        r13DerivativesOutFile.close();
        angleDerivativesOutFile.close();
    }
    fann_destroy(ann);
}

int main()
{
    // Read bounds
    ifstream boundsFile("/home/svenni/Dropbox/studies/master/results/fann_train/20140418-165800/bounds.fann");
    int nInputs;
    boundsFile >> nInputs;
    cout << nInputs << endl;
    int nOutputs;
    boundsFile >> nOutputs;
    cout << nOutputs << endl;

    mat bounds = zeros(nInputs + nOutputs, 2);
    int boundIndex = 0;
    for(int i = 0; i < nInputs + nOutputs; i++) {
        boundsFile >> bounds(boundIndex,0) >> bounds(boundIndex, 1);
        boundIndex++;
    }

    switch(nInputs) {
    case 1:
        createTwoParticlePlots(bounds);
        break;
    case 3:
        createThreeParticlePlots(bounds);
        break;
    default:
        cerr << "Unknown number of inputs. Not sure how to create plot..." << endl;
        break;
    }

    return 0;
}


//fann_type* fann_run_diff(struct fann * ann, fann_type * input)
//{
//    fann_type *derivative_values = new fann_type[ann->total_neurons];
//    bool *derivative_initialized = new bool[ann->total_neurons];
//    for(int i = 0; i < ann->total_neurons; i++) {
//        derivative_initialized[i] = false;
//    }

//    struct fann_neuron *neuron_it, *last_neuron, *neurons, **neuron_pointers;
//    unsigned int i, num_connections, num_input, num_output;
//    fann_type neuron_sum, *output;
//    fann_type *weights;
//    struct fann_layer *layer_it, *last_layer;
//    unsigned int activation_function;
//    fann_type steepness;

//    /* store some variabels local for fast access */
//    struct fann_neuron *first_neuron = ann->first_layer->first_neuron;

//    fann_type max_sum;

//    /* first set the input */
//    num_input = ann->num_input;
//    int neuronCounterTotal = 0;
//    for(i = 0; i != num_input; i++)
//    {
//        first_neuron[i].value = input[i];
//        if(i == 0) {
//            derivative_values[i] = 1;
//        } else {
//            derivative_values[i] = 0;
//        }
//        derivative_initialized[i] = true;
//        neuronCounterTotal++;
//    }
//    /* Set the bias neuron in the input layer */
//    (ann->first_layer->last_neuron - 1)->value = 1;

//    /* Set the derivative of the bias neuron in the input layer */
//    long bias_index = (ann->first_layer->last_neuron - 1) - first_neuron;
//    derivative_values[bias_index] = 0;
//    derivative_initialized[bias_index] = true;

//    last_layer = ann->last_layer;
//    int layerCounter = 1;
//    for(layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++)
//    {
//        int neuronCounter = 0;
//        last_neuron = layer_it->last_neuron;
//        for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
//        {
//            fann_type derivative_sum = 0;

//            if(neuron_it->first_con == neuron_it->last_con)
//            {
//                /* bias neurons */
//                neuron_it->value = 1;
//                continue;
//            }

//            activation_function = neuron_it->activation_function;
//            steepness = neuron_it->activation_steepness;

//            neuron_sum = 0;
//            num_connections = neuron_it->last_con - neuron_it->first_con;
//            weights = ann->weights + neuron_it->first_con;

//            if(ann->connection_rate >= 1)
//            {
//                if(ann->network_type == FANN_NETTYPE_SHORTCUT)
//                {
//                    neurons = ann->first_layer->first_neuron;
//                }
//                else
//                {
//                    neurons = (layer_it - 1)->first_neuron;
//                }


//                /* unrolled loop start */
//                for(uint j = 0; j < num_connections; j++) {
//                    neuron_sum += fann_mult(weights[j], neurons[j].value);
//                    derivative_sum += fann_mult(weights[j], derivative_values[neurons - first_neuron + j]);
//                }
//            }
//            else
//            {
//                cerr << "Currently only support for high connection rates" << endl;
//                exit(0);
//            }

//            neuron_sum = fann_mult(steepness, neuron_sum);

//            max_sum = 150/steepness;
//            if(neuron_sum > max_sum) {
//                neuron_sum = max_sum;
//            }
//            else if(neuron_sum < -max_sum) {
//                neuron_sum = -max_sum;
//            }

//            neuron_it->sum = neuron_sum;

//            fann_activation_switch(activation_function, neuron_sum, neuron_it->value);

////            cout << "Neuron sum: " << neuron_sum << endl;
////            cout << "Derivative sum: " << derivative_sum << endl;
////            cout << "Neuron value: " << neuron_it->value << endl;
////            cout << "steepness: " << steepness << endl;

//            fann_type derivative_value = fann_activation_derived(activation_function,
//                                                                 steepness,
//                                                                 neuron_it->value,
//                                                                 neuron_sum);
////            cout << "Derivative value: " << derivative_value << endl;

//            derivative_values[neuron_it - first_neuron] = derivative_sum * derivative_value;


////            cout << "Derivative " << neuron_it - first_neuron << ": " << derivative_values[neuron_it - first_neuron] << endl;
//            derivative_initialized[neuron_it - first_neuron] = true;
//            neuronCounter++;
//            neuronCounterTotal++;
////            break;
//        }
////        break;
//        layerCounter++;
//    }

//    /* set the output */
//    output = ann->output;
//    num_output = ann->num_output;
//    neurons = (ann->last_layer - 1)->first_neuron;
//    for(i = 0; i != num_output; i++)
//    {
//        output[i] = neurons[i].value;
//    }
//    fann_type* derivative_output = derivative_values + ((ann->last_layer - 1)->first_neuron - first_neuron);
//    cout << "Hopeful: " << derivative_output[0] << endl;
//    cout << "Hopeful: " << derivative_values[ann->total_neurons-1] << endl;
//    return ann->output;
//}
