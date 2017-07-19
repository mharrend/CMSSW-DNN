/*
 * Load a tensorflow model
 *
 * Graph created using NNFlow Framework:
 *   Input:
 *     - name = "input:0", shape = (batch, numberOfInputNeurons)
 *   Output:
 *     - name = "output:0", shape = (batch, numberOfOutputNeurons)
 *
 * Usage:
 *     - Hand over a model file in the constructor
 *     - Hand over number of input and output neurons in constructor as two string lists
 *     - Call evalModel function with float vector containing variables of a single event and empty float vector which will contain  NN output
 *
 * Author:
 *   Marco A. Harrendorf
 */

#include <iostream>
#include <string>
#include <vector>

#include "DNN/Tensorflow/interface/Graph.h"
#include "DNN/Tensorflow/interface/Tensor.h"


class tfModelUser
{
public:
    // Constructors
    // delete empty default one, C++11
    tfModelUser() = delete;
    // Give model file location and either number of input and output neurons or strings containing input and output neuron names
    tfModelUser(std::string modelFileLocation, std::vector<std::string> inputNeurons, std::vector<std::string> outputNeurons);
    tfModelUser(std::string modelFileLocation, unsigned int numberOfInputNeurons, unsigned int numberOfOutputNeurons);
    
    // Evaluate model after initialization took place in constructor
    void tfModelUser::evalModel(std::vector<float> eventVariables, std::vector<float> &NNOutput);
    
    // Destructor
    // Cleaning created Tensorflow tensors
    ~tfModelUser();
    
private:
    // Storing number of output neurons since required during eval step
    unsigned int m_numberOfOutputNeurons;
    
}