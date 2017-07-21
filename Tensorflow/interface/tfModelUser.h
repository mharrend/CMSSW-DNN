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

#ifndef DNN_TENSORFLOW_MODELUSER_H
#define DNN_TENSORFLOW_MODELUSER_H

#include <iostream>
#include <string>
#include <vector>

#include "DNN/Tensorflow/interface/Graph.h"
#include "DNN/Tensorflow/interface/Tensor.h"

namespace dnn
{

namespace tf
{

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
    std::vector<float> evalModel(std::vector<float> eventVariables);
    
    // Destructor
    // Cleaning created Tensorflow tensors
    ~tfModelUser();
    
private:
    // tensorflow graph
    dnn::tf::Graph m_g;    
    // tensorflow x and y variables
    dnn::tf::Tensor* m_x;
    dnn::tf::Tensor* m_y;
    
    // Storing number of output neurons since required during eval step
    unsigned int m_numberOfOutputValues;
    
};

} // namepace tf

} // namepace dnn

#endif
