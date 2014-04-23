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
        case FANN_SIGMOID_STEPWISE:
            return (fann_type) fann_sigmoid_derive(steepness, value);
        case FANN_SIGMOID_SYMMETRIC:
        case FANN_SIGMOID_SYMMETRIC_STEPWISE:
            return (fann_type) fann_sigmoid_symmetric_derive(steepness, value);
        case FANN_GAUSSIAN:
            /* value = fann_clip(value, 0.01f, 0.99f); */
            return (fann_type) fann_gaussian_derive(steepness, value, sum);
        case FANN_GAUSSIAN_SYMMETRIC:
            /* value = fann_clip(value, -0.98f, 0.98f); */
            return (fann_type) fann_gaussian_symmetric_derive(steepness, value, sum);
        case FANN_ELLIOT:
            return (fann_type) fann_elliot_derive(steepness, value, sum);
        case FANN_ELLIOT_SYMMETRIC:
            return (fann_type) fann_elliot_symmetric_derive(steepness, value, sum);
        case FANN_SIN_SYMMETRIC:
            return (fann_type) fann_sin_symmetric_derive(steepness, sum);
        case FANN_COS_SYMMETRIC:
            return (fann_type) fann_cos_symmetric_derive(steepness, sum);
        case FANN_SIN:
            return (fann_type) fann_sin_derive(steepness, sum);
        case FANN_COS:
            return (fann_type) fann_cos_derive(steepness, sum);
        case FANN_THRESHOLD:
            fann_error(NULL, FANN_E_CANT_TRAIN_ACTIVATION);
    }
    return 0;
}

void backpropagate_derivative(struct fann *ann)
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
    const struct fann_layer *first_layer = ann->first_layer;
    const struct fann_layer *second_layer = ann->first_layer + 1;
    struct fann_layer *last_layer = ann->last_layer;

    for(int i = 0; i < (last_layer-1)->first_neuron - first_neuron; i++) {
        ann->train_errors[i] = 0.0;
    }

    fann_type neuron_value, neuron_diff, *error_it = 0;
    struct fann_neuron *last_layer_begin = (ann->last_layer - 1)->first_neuron;
    const struct fann_neuron *last_layer_end = last_layer_begin + ann->num_output;
    error_it = error_begin + (last_layer_begin - first_neuron);

    for(; last_layer_begin != last_layer_end; last_layer_begin++)
    {
        neuron_value = last_layer_begin->value;
        neuron_diff = 1.0;
        *error_it = fann_activation_derived(last_layer_begin->activation_function,
                                            last_layer_begin->activation_steepness, neuron_value,
                                            last_layer_begin->sum) * neuron_diff;
        cout << "Initial error: " << *error_it << endl;
        error_it++;
    }

    //    fann_type* last_neuron_error = error_begin + (ann->last_layer->first_neuron - ann->first_layer->first_neuron);
    //    last_neuron_error[0] = 1.0;

    /* go through all the layers, from last to first.
     * And propagate the error backwards */
    for(layer_it = last_layer - 1; layer_it > ann->first_layer; --layer_it)
    {
        last_neuron = layer_it->last_neuron;

        /* for each connection in this layer, propagate the error backwards */
        //        if(ann->connection_rate >= 1)
        //        {
        //        if(ann->network_type == FANN_NETTYPE_LAYER)
        //        {
        error_prev_layer = error_begin + ((layer_it - 1)->first_neuron - first_neuron);
        //        }
        //            else
        //            {
        //                error_prev_layer = error_begin;
        //            }

        cout << endl;
        cout << endl;
        cout << "Layer" << endl;
        cout << endl;
        cout << endl;

        if( (layer_it - 1) == first_layer) {
            cout << endl;
            cout << endl;
            cout << "First!" << endl;
            cout << endl;
            cout << endl;
        }
        for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
        {
            tmp_error = error_begin[neuron_it - first_neuron];
            weights = ann->weights + neuron_it->first_con;
            for(i = neuron_it->last_con - neuron_it->first_con; i--;)
            {
                //                    printf("error_prev_layer[%d] = %f\n", i, error_prev_layer[i]);
                printf("weights[%d] = %f\n", i, weights[i]);
                error_prev_layer[i] += tmp_error * weights[i];
                printf("error_prev_layer[%d] = %f\n", i, error_prev_layer[i]);
            }
        }
        //        }
        //        else
        //        {
        //            for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
        //            {

        //                tmp_error = error_begin[neuron_it - first_neuron];
        //                weights = ann->weights + neuron_it->first_con;
        //                connections = ann->connections + neuron_it->first_con;
        //                for(i = neuron_it->last_con - neuron_it->first_con; i--;)
        //                {
        //                    error_begin[connections[i] - first_neuron] += tmp_error * weights[i];
        //                }
        //            }
        //        }

        /* then calculate the actual errors in the previous layer */
        error_prev_layer = error_begin + ((layer_it - 1)->first_neuron - first_neuron);
        last_neuron = (layer_it - 1)->last_neuron;

        if((layer_it - 1) > ann->first_layer) {
            for(neuron_it = (layer_it - 1)->first_neuron; neuron_it != last_neuron; neuron_it++)
            {
                *error_prev_layer *= fann_activation_derived(neuron_it->activation_function,
                                                             neuron_it->activation_steepness,
                                                             neuron_it->value,
                                                             neuron_it->sum / neuron_it->activation_steepness);
                error_prev_layer++;
            }
        }
        cout << ann->train_errors[0] << endl;
    }
    //    fann_type sum = 0;
    //    error_prev_layer = error_begin + (second_layer->first_neuron - first_neuron);
    //    for(neuron_it = second_layer->first_neuron; neuron_it != second_layer->last_neuron; neuron_it++) {
    //        cout << *error_prev_layer << endl;
    //        sum += *error_prev_layer;
    //        error_prev_layer++;
    //    }
    //    cout << "Sum: " << sum << endl;
}

/* INTERNAL FUNCTION
   Propagate the error backwards from the output layer.

   After this the train_errors in the hidden layers will be:
   neuron_value_derived * sum(outgoing_weights * connected_neuron)
*/
void fann_backpropagate_MSE2(struct fann *ann)
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
        //        if(ann->connection_rate >= 1)
        //        {
        //            if(ann->network_type == FANN_NETTYPE_LAYER)
        //            {
        error_prev_layer = error_begin + ((layer_it - 1)->first_neuron - first_neuron);
        //            }
        //            else
        //            {
        //                error_prev_layer = error_begin;
        //            }

        for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
        {
            tmp_error = error_begin[neuron_it - first_neuron];
            weights = ann->weights + neuron_it->first_con;
            for(i = neuron_it->last_con - neuron_it->first_con; i--;)
            {
//                cout << "tmp_error: " << tmp_error << endl;
//                cout << "weights[i]: " << weights[i] << endl;
                /*printf("i = %d\n", i);
                     * printf("error_prev_layer[%d] = %f\n", i, error_prev_layer[i]);
                     * printf("weights[%d] = %f\n", i, weights[i]); */
                error_prev_layer[i] += tmp_error * weights[i];
            }
        }
        //        }
        //        else
        //        {
        //            for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
        //            {

        //                tmp_error = error_begin[neuron_it - first_neuron];
        //                weights = ann->weights + neuron_it->first_con;
        //                connections = ann->connections + neuron_it->first_con;
        //                for(i = neuron_it->last_con - neuron_it->first_con; i--;)
        //                {
        //                    error_begin[connections[i] - first_neuron] += tmp_error * weights[i];
        //                }
        //            }
        //        }

        /* then calculate the actual errors in the previous layer */
        error_prev_layer = error_begin + ((layer_it - 1)->first_neuron - first_neuron);
        last_neuron = (layer_it - 1)->last_neuron;

        if(layer_it - 1 > ann->first_layer) {
            for(neuron_it = (layer_it - 1)->first_neuron; neuron_it != last_neuron; neuron_it++)
            {
                fann_type sum = neuron_it->sum;
                if(neuron_it->activation_function != FANN_ELLIOT && neuron_it->activation_function != FANN_ELLIOT_SYMMETRIC) {
                    sum /= neuron_it->activation_steepness;
                }
                *error_prev_layer *= fann_activation_derived2(neuron_it->activation_function,
                                                             neuron_it->activation_steepness,
                                                             neuron_it->value,
                                                             sum);
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
                fann_set_activation_function(ann, FANN_SIGMOID, layer_number, neuron_number);
                neuron_number++;
            }
            layer_number++;
        }
//    struct fann* ann = fann_create_standard(2, 1, 1);
//    fann_set_activation_function(ann, FANN_SIGMOID, 1, 0);
////    fann_set_activation_function(ann, FANN_SIGMOID, 1, 1);
////    fann_set_activation_function(ann, FANN_SIGMOID, 2, 0);
////    //    fann_set_activation_function(ann, FANN_COS, 2, 0);
////    //    fann_set_activation_function(ann, FANN_LINEAR, 2, 0);

//    fann_set_activation_steepness(ann, 0.35, 1, 0);
////    fann_set_activation_steepness(ann, 0.35, 1, 1);
////    fann_set_activation_steepness(ann, 0.35, 2, 0);
//    //    fann_set_activation_steepness(ann, 1.0, 2, 0);
//    //    fann_set_activation_steepness(ann, 1.0, 2, 0);

//    fann_connection connections[fann_get_total_connections(ann)];
//    fann_get_connection_array(ann, connections);
//    for(int i = 0; i < fann_get_total_connections(ann); i++) {
//        connections[i].weight = 5.0;
//    }
//    fann_set_weight_array(ann, connections, fann_get_total_connections(ann));

//    fann_save(ann, "banana.out");

    fann_type input[1];
    fann_type value = 0.34;
    //    input[0] = rescale(1.4, 0.5, 6.0);
    input[0] = value;

    fann_type* output = fann_run(ann, input);

    cout << "Output for " << value << ": " << output[0] << endl;

    // Numerical derivative
    fann_type h = 1e-8;
    input[0] = value + h;
    fann_type outputPlus = fann_run(ann, input)[0];
    input[0] = value - h;
    fann_type outputMinus = fann_run(ann, input)[0];
    fann_type derivative = (outputPlus - outputMinus) / (2*h);

    input[0] = value;
    output = fann_run(ann, input);

    // Backpropagated derivative
    fann_type desiredOutput[1];
    desiredOutput[0] = 0.0;
    fann_compute_MSE(ann, desiredOutput);

    //    backpropagate_derivative(ann);

    fann_type *error_begin = ann->train_errors;
    fann_neuron *last_layer_begin = (ann->last_layer - 1)->first_neuron;
    const struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
    fann_type *error_it = error_begin + (last_layer_begin - first_neuron);
    cout << "Output for x: " << last_layer_begin->value << endl;
    cout << "Numerical deriv.: " << derivative << endl;
    fann_type sum = last_layer_begin->sum;
    if(last_layer_begin->activation_function != FANN_ELLIOT && last_layer_begin->activation_function != FANN_ELLIOT_SYMMETRIC) {
        sum /= last_layer_begin->activation_steepness;
//        cout << "Sum: " << sum << endl;
    }
//    cout << "Value: " << last_layer_begin->value << endl;
//    cout << "Steepness: " << last_layer_begin->activation_steepness << endl;
//    cout << "Diff: " << (1.0 - last_layer_begin->value) << endl;
//    cout << "Result: " << (2.0 * last_layer_begin->activation_steepness * last_layer_begin->value * (1.0 - last_layer_begin->value)) << endl;
//    cout << "Result2: " << fann_sigmoid_derive(last_layer_begin->activation_steepness, last_layer_begin->value) << endl;
    error_it[0] = fann_activation_derived2(last_layer_begin->activation_function,
                                          last_layer_begin->activation_steepness,
                                          last_layer_begin->value,
                                          sum);
//    cout << "Error_it[0]: " << error_it[0] << endl;
    //    error_it[0] = fann_elliot_derive(last_layer_begin->activation_steepness,
    //                                     last_layer_begin->value,
    //                                     sum);

    fann_backpropagate_MSE2(ann);

    cout << "Final derivative: " << ann->train_errors[0] << endl;

    fann_destroy(ann);
    return 0;
}
