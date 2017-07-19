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



tfModelUser::tfModelUser(std::string modelFileLocation, std::vector<std::string> inputNeurons, std::vector<std::string> outputNeurons)
{
    tfModelUser(modelFileLocation, inputNeurons.size(), outputNeurons.size());
    
}

tfModelUser::tfModelUser(std::string modelFileLocation, unsigned int numberOfInputNeurons, unsigned int numberOfOutputNeurons)
{
    //
    // object definitions
    //

    // load and initialize the graph
    std::cout << "Load Tensorflow model graph from following location: " << modelFileLocation << std::endl;
    dnn::tf::Graph g(modelFileLocation, dnn::LogLevel::ALL);

    // prepare input and output tensors
    // input tensor contains only 1 event with n input variables
    dnn::tf::Shape xShape[] = { 1, numberOfInputNeurons };
    dnn::tf::Tensor* x = g.defineInput(new dnn::tf::Tensor("input:0", 2, xShape));
    // output tensor can contain either one or more neurons
    dnn::tf::Tensor* y = g.defineOutput(new dnn::tf::Tensor("output:0"));
    // save number of output neurons
    m_numberOfOutputNeurons =  numberOfOutputNeurons;

}

void tfModelUser::evalModel(std::vector<float> eventVariables, std::vector<float> &NNOutput)
{
    // Clear output vector before filling it again
    NNOutput.clear();

    // Fill event variables x values
    x->setVector<float>(1, 0, eventVariables); // axis: 1, axis 0 (= batch dim) value: 0, vector: v

    // evaluation call
    // this does not return anything but changes the output tensor(s) in place which is faster
    g.eval();

    for (int i = 0; i < m_numberOfOutputNeurons; i++) {
    	NNOutput.append(y->getValue<float>(0, i));
    }
}


tfModelUser::~tfModelUser()
{
    // cleanup
    // remove tensors manually (you type 'new', you type 'delete')
    delete x;
    delete y;

}

