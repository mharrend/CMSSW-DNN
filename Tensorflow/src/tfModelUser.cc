/*
 * Load a tensorflow model
 *
 * Author:
 *   Marco A. Harrendorf
 */

#include "DNN/Tensorflow/interface/tfModelUser.h"

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

