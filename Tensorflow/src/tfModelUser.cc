/*
 * Load a tensorflow model
 *
 * Author:
 *   Marco A. Harrendorf
 */

#include "DNN/Tensorflow/interface/tfModelUser.h"

namespace dnn
{

namespace tf
{

tfModelUser::tfModelUser(std::string modelFileLocation, std::vector<std::string> inputNeurons, std::vector<std::string> outputNeurons)
{
    init(modelFileLocation, inputNeurons.size(), outputNeurons.size());
    
}

tfModelUser::tfModelUser(std::string modelFileLocation, unsigned int numberOfInputNeurons, unsigned int numberOfOutputValues)
{
    init(modelFileLocation, numberOfInputNeurons, numberOfOutputValues);
}

void tfModelUser::init(std::string modelFileLocation, unsigned int numberOfInputNeurons, unsigned int numberOfOutputValues)
{
    // load and initialize the graph
    std::cout << "Load Tensorflow model graph from following location: " << modelFileLocation << std::endl;
    m_g = new dnn::tf::Graph(modelFileLocation, dnn::LogLevel::ERROR);

    // prepare input and output tensors
    // input tensor contains only 1 event with n input variables
    std::cout << "Model contains " << numberOfInputNeurons << " input variables" << std::endl;
    dnn::tf::Shape xShape[] = { 1, numberOfInputNeurons };
    m_x = new dnn::tf::Tensor("input:0", 2, xShape);
    m_x = m_g->defineInput(m_x);
    
    // save number of output values
    m_numberOfOutputValues =  numberOfOutputValues;
    std::cout << "Model returns " << m_numberOfOutputValues << " output values" << std::endl;
    // output tensor can contain either one or more neurons
    dnn::tf::Shape yShape[] = { 1, m_numberOfOutputValues };
    m_y = new dnn::tf::Tensor("output:0", 2, yShape);
    m_y = m_g->defineOutput(m_y);

    std::cout << "Finished with Tensorflow model initialization" << std::endl;
}

std::vector<float> tfModelUser::evalModel(std::vector<float> eventVariables)
{
    // Use RVO and create local vector here
    std::vector<float> NNOutput;

    // Fill event variables x values
    m_x->setVector<float>(1, 0, eventVariables); // axis: 1, axis 0 (= batch dim) value: 0, vector: v

    // evaluation call
    // this does not return anything but changes the output tensor(s) in place which is faster
    m_g->eval();

    for (unsigned int i = 0; i < m_numberOfOutputValues; i++) {
    	NNOutput.push_back(m_y->getValue<float>(0, i));
    }
    return NNOutput;
}


tfModelUser::~tfModelUser()
{
    // cleanup
    // remove tensors manually (you type 'new', you type 'delete')
    delete m_x;
    delete m_y;
    delete m_g;

}

} // namepace tf

} // namepace dnn
