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

fann_type fann_activation_derived2(unsigned int activation_function,
                                   fann_type steepness, fann_type value, fann_type sum)
{
    switch (activation_function)
    {
    case FANN_LINEAR:
    case FANN_LINEAR_PIECE:
    case FANN_LINEAR_PIECE_SYMMETRIC:
        return (fann_type) fann_linear_derive(steepness, value);
    case FANN_SIGMOID:
        return (fann_type) fann_sigmoid_derive(steepness, value);
    case FANN_SIGMOID_STEPWISE:
        fann_error(NULL, FANN_E_CANT_TRAIN_ACTIVATION);
    case FANN_SIGMOID_SYMMETRIC:
        return (fann_type) fann_sigmoid_symmetric_derive(steepness, value);
    case FANN_SIGMOID_SYMMETRIC_STEPWISE:
        fann_error(NULL, FANN_E_CANT_TRAIN_ACTIVATION);
    case FANN_GAUSSIAN:
        return (fann_type) fann_gaussian_derive(steepness, value, sum/steepness);
    case FANN_GAUSSIAN_SYMMETRIC:
        return (fann_type) fann_gaussian_symmetric_derive(steepness, value, sum/steepness);
    case FANN_ELLIOT:
        return (fann_type) fann_elliot_derive(steepness, value, sum);
    case FANN_ELLIOT_SYMMETRIC:
        return (fann_type) fann_elliot_symmetric_derive(steepness, value, sum);
    case FANN_SIN_SYMMETRIC:
        return (fann_type) fann_sin_symmetric_derive(steepness, sum/steepness);
    case FANN_COS_SYMMETRIC:
        return (fann_type) fann_cos_symmetric_derive(steepness, sum/steepness);
    case FANN_SIN:
        return (fann_type) fann_sin_derive(steepness, sum/steepness);
    case FANN_COS:
        return (fann_type) fann_cos_derive(steepness, sum/steepness);
    case FANN_THRESHOLD:
        fann_error(NULL, FANN_E_CANT_TRAIN_ACTIVATION);
    }
    return 0;
}

void fann_backpropagate_derivative(struct fann *ann)
{
    fann_type tmp_error;
    unsigned int i;
    struct fann_layer *layer_it;
    struct fann_neuron *neuron_it, *last_neuron;
    struct fann_neuron **connections;

    fann_type *error_begin = ann->train_errors;
    fann_type *error_prev_layer;
    fann_type *weights;
    const struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
    const struct fann_layer *second_layer = ann->first_layer + 1;
    struct fann_layer *last_layer = ann->last_layer;

    for(int i = 0; i < (last_layer-1)->first_neuron - first_neuron; i++) {
        ann->train_errors[i] = 0.0;
    }

    /* go through all the layers, from last to first.
     * And propagate the error backwards */
    for(layer_it = last_layer - 1; layer_it > ann->first_layer; --layer_it)
    {
        last_neuron = layer_it->last_neuron;

        /* for each connection in this layer, propagate the error backwards */
        if(ann->connection_rate >= 1)
        {
            if(ann->network_type == FANN_NETTYPE_LAYER)
            {
                error_prev_layer = error_begin + ((layer_it - 1)->first_neuron - first_neuron);
            }
            else
            {
                error_prev_layer = error_begin;
            }

            for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
            {
                tmp_error = error_begin[neuron_it - first_neuron];
                weights = ann->weights + neuron_it->first_con;
                for(i = neuron_it->last_con - neuron_it->first_con; i--;)
                {
                    /*printf("i = %d\n", i);
                     * printf("error_prev_layer[%d] = %f\n", i, error_prev_layer[i]);
                     * printf("weights[%d] = %f\n", i, weights[i]); */
                    error_prev_layer[i] += tmp_error * weights[i];
                }
            }
        }
        else
        {
            for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
            {

                tmp_error = error_begin[neuron_it - first_neuron];
                weights = ann->weights + neuron_it->first_con;
                connections = ann->connections + neuron_it->first_con;
                for(i = neuron_it->last_con - neuron_it->first_con; i--;)
                {
                    error_begin[connections[i] - first_neuron] += tmp_error * weights[i];
                }
            }
        }

        /* then calculate the actual errors in the previous layer */
        error_prev_layer = error_begin + ((layer_it - 1)->first_neuron - first_neuron);
        last_neuron = (layer_it - 1)->last_neuron;

        if(layer_it - 1 > ann->first_layer) {
            for(neuron_it = (layer_it - 1)->first_neuron; neuron_it != last_neuron; neuron_it++)
            {
                *error_prev_layer *= fann_activation_derived2(neuron_it->activation_function,
                                                              neuron_it->activation_steepness,
                                                              neuron_it->value,
                                                              neuron_it->sum);
                error_prev_layer++;
            }
        }
    }
}

int main()
{
    cout << setprecision(20);
    string inFileName = "/home/svenni/Dropbox/studies/master/results/fann_train/20140418-165800/fann_network.net";

    struct fann* ann = fann_create_from_file(inFileName.c_str());
    int layer_number = 1;
    for(fann_layer* layer_it = ann->first_layer + 1; layer_it < ann->last_layer; layer_it++) {
        int neuron_number = 0;
        for(fann_neuron* neuron_it = layer_it->first_neuron; neuron_it < layer_it->last_neuron; neuron_it++) {
            fann_activationfunc_enum activation_function = fann_get_activation_function(ann, layer_number, neuron_number);
            switch (activation_function)
            {
            case FANN_SIGMOID_STEPWISE:
                fann_set_activation_function(ann, FANN_SIGMOID, layer_number, neuron_number);
                break;
            case FANN_SIGMOID_SYMMETRIC_STEPWISE:
                fann_set_activation_function(ann, FANN_SIGMOID_SYMMETRIC, layer_number, neuron_number);
                break;
            case FANN_THRESHOLD:
                fann_error(NULL, FANN_E_CANT_TRAIN_ACTIVATION);
                break;
            case FANN_THRESHOLD_SYMMETRIC:
                fann_error(NULL, FANN_E_CANT_TRAIN_ACTIVATION);
                break;
            default:
                break;
            }
            neuron_number++;
        }
        layer_number++;
    }

    fann_type input[1];
    fann_type value = 0.15;
    input[0] = value;

    fann_type* output = fann_run(ann, input);

    cout << "Output for " << value << ": " << output[0] << endl;
    clock_t begin_time = clock();
    fann_type derivative = 0;
    for(int i = 0; i < 5000000; i++) {
        // Numerical derivative
        fann_type h = 1e-8;
        input[0] = value + h;
        fann_type outputPlus = fann_run(ann, input)[0];
        input[0] = value - h;
        fann_type outputMinus = fann_run(ann, input)[0];
        derivative = (outputPlus - outputMinus) / (2*h);
    }
    cout << "Numerical deriv.: " << derivative << endl;
    std::cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;

    // Backpropagated derivative
    fann_type desiredOutput[1];
    desiredOutput[0] = 0.0;
    fann_compute_MSE(ann, desiredOutput);


    begin_time = clock();
    for(int i = 0; i < 5000000; i++) {
        input[0] = value;
        output = fann_run(ann, input);
        fann_type *error_begin = ann->train_errors;
        fann_neuron *last_layer_begin = (ann->last_layer - 1)->first_neuron;
        const struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
        fann_type *error_it = error_begin + (last_layer_begin - first_neuron);
        error_it[0] = fann_activation_derived2(last_layer_begin->activation_function,
                                               last_layer_begin->activation_steepness,
                                               last_layer_begin->value,
                                               last_layer_begin->sum);

        fann_backpropagate_derivative(ann);
    }

    cout << "Final derivative: " << ann->train_errors[0] << endl;
    std::cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;

    fann_destroy(ann);
    return 0;
}
