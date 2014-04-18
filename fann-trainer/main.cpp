#include <iostream>
#include <sstream>
#include <fstream>
#include <doublefann.h>
#include <mpi/mpi.h>
#include <armadillo>

using namespace std;
using arma::mat;
using arma::zeros;

void train(int rank, string outputDirectory) {
    stringstream inFileName;
    inFileName << "train.fann";
    stringstream inFileTestName;
    inFileTestName << "test.fann";

    int num_data = 0;
    int num_input = 0;
    int num_output = 0;
    ifstream inFile(inFileName.str());
    inFile >> num_data;
    inFile >> num_input;
    inFile >> num_output;

    cout << "Found " << num_input << " input(s) and " << num_output << " output(s) in training file." << endl;

    const unsigned int num_layers = 3;
    const unsigned int num_neurons_hidden = 10;
    const double desired_error = (const double) 0.00005;
    const unsigned int neurons_between_reports = 1;
    double bestTestResult = 999999999;

    bool cascade = true;
    struct fann *ann;
    if(cascade) {
        ann = fann_create_shortcut(num_layers,
                                   num_input,
                                   num_neurons_hidden,
                                   num_output);
    } else {
        ann = fann_create_standard(num_layers,
                                   num_input,
                                   num_neurons_hidden,
                                   num_output);
    }
//    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
//    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
//    fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
    struct fann_train_data *trainData = fann_read_train_from_file(inFileName.str().c_str());
    struct fann_train_data *testData = fann_read_train_from_file(inFileTestName.str().c_str());
//    enum fann_activationfunc_enum activation = FANN_SIGMOID_SYMMETRIC;
//    fann_set_cascade_activation_functions(ann, &activation, 1);
//    fann_set_cascade_num_candidate_groups(ann, 20);
//            fann_set_cascade_candidate_limit(ann, 1000);
    //        fann_set_cascade_output_change_fraction(ann, 0.2);
//    fann_set_cascade_weight_multiplier(ann, 0.00010);

    for(int i = 1; i < 20; i++) {
        cout << "---------- RANK " << rank << " ITERATION " << i << "-----------" << endl;
        //            fann_set_cascade_candidate_limit(ann, i*1000);
        fann_cascadetrain_on_data(ann, trainData, 1, neurons_between_reports, desired_error);
        //        cout << "Reset and test" << endl;
        fann_reset_MSE(ann);
        fann_test_data(ann, testData);
        double testResult = fann_get_MSE(ann);
        cout << "Results from testing data: " << testResult << endl;
        if(testResult / bestTestResult > 1.01) {
            cout << "No longer converging. Stopping to avoid overfitting." << endl;
            break;
        } else {
            bestTestResult = testResult;
            stringstream outFileName;
            outFileName << outputDirectory << "/fann_network.net";
            fann_save(ann, outFileName.str().c_str());
            stringstream outFileName2;
            outFileName2 << outputDirectory << "/testresult.out";
            ofstream testResultFile(outFileName2.str());
            testResultFile << bestTestResult;
            testResultFile.close();
        }
    }
    //    }

    fann_destroy_train(trainData);
    fann_destroy_train(testData);
    fann_destroy(ann);
}

int main(int argc, char* argv[])
{
    int rank;
    int nProcessors;
    string outputDirectory = ".";
    if(argc > 1) {
        outputDirectory = argv[1];
    }
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcessors);
    train(0, outputDirectory);
    MPI_Finalize();
    return 0;
}

