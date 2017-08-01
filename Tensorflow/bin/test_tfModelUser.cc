/*
 * Test of loading a multimodal tensorflow model produced by NNFlow using the tfModelUser class
 *
 * Usage:
 *   > test_tfModelUser
 *
 * Author:
 *   Marco A. Harrendorf
 *   Lukas Hilser
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "DNN/Tensorflow/interface/tfModelUser.h"


std::vector<std::vector<float>> createVectorOfEventVectors();
std::vector<std::vector<float>> createVectorOfOutputValues();
std::vector<std::string> readinNumberOfVariables(std::string variableListLocation);

// Function opens file with variable list and returns std::vector<std::string> of variables
std::vector<std::string> readinNumberOfVariables(std::string variableListLocation)
{
    // Define string vector
    std::vector<std::string> variableList;
    
    // Reading in variables
    std::string tempVariable;
    std::ifstream variableListFile;
    variableListFile.open(variableListLocation);
    if (variableListFile.is_open()) {
        while (!variableListFile.eof()) {
            getline(variableListFile,tempVariable);
            variableList.push_back(tempVariable);
        }
    }
    variableListFile.close();
    // Remove last empyt entry
    variableList.pop_back();

    for(const auto &i: variableList)
	std::cout << "VariableList: " << i << std::endl;

    return variableList;
}



int main(int argc, char* argv[])
{
    std::cout << std::endl
              << "Testing tfModelUser class" << std::endl;

    // get the file containing the graph
    std::string cmsswBase = std::string(getenv("CMSSW_BASE"));
    std::string dataDir = cmsswBase + "/src/DNN/Tensorflow/data";
    std::string modelLoc = dataDir + "/tfModelUserModel";
    std::cout << "Will use the tfModelUser model: " << modelLoc << std::endl;

    // Read in input variable list
    std::string inputvariableListLoc = dataDir + "/tfModelUserInputvariables.txt";
    std::cout << "Will use the tfModelUser input variable list: " << inputvariableListLoc << std::endl;
    std::vector<std::string> inputvariableList = readinNumberOfVariables(inputvariableListLoc);
    
    // Read in output label list
    std::string outputLabelListLoc = dataDir + "/tfModelUserOutputLabels.txt";
    std::cout << "Will use the tfModelUser output label list: " << outputLabelListLoc << std::endl;
    std::vector<std::string> outputLabelList = readinNumberOfVariables(outputLabelListLoc);
    
    

    // load and initialize the model
    dnn::tf::tfModelUser modelUser(modelLoc, inputvariableList, outputLabelList);
    //dnn::tf::tfModelUser modelUser(modelLoc, 243, 4);
    
    // create vector list containing events 
    std::vector<std::vector<float>> eventList= createVectorOfEventVectors();
    unsigned int sizeOfEventList = eventList.size();
    std::cout << "Size of eventList: " << sizeOfEventList << std::endl;

    // create vector list containing known output values of events
    std::vector<std::vector<float>> outputValuesList= createVectorOfOutputValues();
    unsigned int sizeOfOutputValuesList = outputValuesList.size();
    std::cout << "Size of outputValuesList: " << sizeOfOutputValuesList << std::endl;
    
    // Evaluate model for events and compare with known output values
    std::vector<float> outputValuesReturnVec;
    for(unsigned int i = 0; i < sizeOfEventList; i++) {
        outputValuesReturnVec.clear();
        outputValuesReturnVec = modelUser.evalModel(eventList[i]);
        std::cout << "Output values obtained from model:" << std::endl;
        for (unsigned int j = 0; j < outputLabelList.size(); j++) 
            std::cout << outputValuesReturnVec[j] << " ";
        std::cout << std::endl;
        std::cout << "Known output values for model:" << std::endl;
        for (unsigned int j = 0; j < outputLabelList.size(); j++) 
            std::cout << outputValuesList[i][j] << " ";
        std::cout << std::endl;
    }
    
}

std::vector<std::vector<float>> createVectorOfEventVectors()
{
std::vector<std::vector<float>> eventsVec;
std::vector<float> tempEventVec;
tempEventVec.clear();
tempEventVec={ 1.28824432e+02,   6.78074829e+02,   6.16816320e-02,   3.34909964e+00,
   0.00000000e+00,   2.56470612e+02,   5.96144721e-02,  -2.35678256e-01,
   0.00000000e+00,   2.05367255e+00,   2.53674340e+00,   3.63764215e+00,
   1.57992852e+00,   1.16999650e+00,   3.84582251e-01,   2.06280485e-01,
   1.61104575e-01,   1.61287129e-01,   1.02325574e+03,   8.93915057e-01,
   4.20884401e-01,   2.24505141e-01,   2.56470612e+02,   4.53542978e-01,
   9.45549607e-01,   1.24088801e-01,   2.66612368e-03,   1.44469276e-01,
   8.93915057e-01,   1.39741433e+00,   2.05367255e+00,   1.47587287e+00,
   2.18415761e+00,   9.35344160e-01,   1.16999650e+00,   1.16999650e+00,
   3.34909964e+00,   9.35344160e-01,   1.86470079e+00,   1.24784653e+02,
  -1.05334961e+00,   2.43417174e-01,  -1.26648986e+00,  -9.46779490e-01,
   6.78074829e+02,   4.64869598e+02,   4.20884371e-01,   1.86996067e+00,
   1.56364670e+02,   2.56470612e+02,   1.32077759e+02,   1.45535172e+02,
   6.75133209e+01,   4.93053484e+00,   2.56470612e+02,   8.89395294e+01,
   1.28824432e+02,   1.28824432e+02,   2.56470612e+02,   8.89395294e+01,
  -2.08748002e-02,   1.24728699e+01,   2.56470612e+02,   1.02325574e+03,
   1.22530718e+01,  -3.09134603e+00,   2.94345856e+00,   9.20283127e+01,
   1.88931473e+02,   5.75021362e+01,   1.88931473e+02,   1.21176910e+02,
   1.86996067e+00,   4.34699869e+00,   1.78306490e-01,  -1.52786362e+00,
   0.00000000e+00,   9.93907392e-01,   1.38608897e+00,   9.59460378e-01,
   1.38608897e+00,   6.01708770e-01,   1.08539200e+00,   1.45668352e+00,
   1.44019103e+00,   1.45668352e+00,   1.26943278e+00,   1.53786457e+00,
   1.83627057e+00,   2.07134509e+00,   1.83627057e+00,   1.26943278e+00,
   1.53786457e+00,   1.83627057e+00,   2.07134509e+00,   1.83627057e+00,
   1.26943278e+00,   1.53786457e+00,   1.83627057e+00,   2.07134509e+00,
   1.83627057e+00,   5.74134350e-01,   3.48256290e-01,   1.63522983e+00,
   4.76872660e-02,   2.01580361e-01,   2.54328775e+00,   3.05446720e+00,
   2.04435182e+00,   1.88210846e+02,   1.05822472e+02,   6.63933396e-01,
   6.14999473e-01,   8.74909014e-02,   6.63933396e-01,   8.52244198e-01,
   7.82968819e-01,   3.20915192e-01,   8.52244198e-01,   7.64504075e-01,
   7.51442850e-01,   5.81777751e-01,   7.64504015e-01,   7.44870663e-01,
   6.93104804e-01,   8.31331611e-01,   7.44870663e-01,   6.21674359e-01,
   6.54288888e-01,   9.35519457e-01,   6.21674359e-01,   5.34222603e-01,
   6.43497705e-01,   7.29329169e-01,   5.65843693e-05,   1.32717952e-08,
   6.90705180e-02,   1.86112165e-01,   6.48993519e-05,   2.39560016e-08,
   3.97629846e-10,   6.95764502e-10,   1.21940832e-10,   3.54247243e-10,
   7.06483816e-10,   8.30261471e-10,   4.14252799e-11,   6.29404362e-10,
   2.57654915e-06,   1.65090808e-06,   7.99120414e-07,   2.29544003e-06,
   1.54326510e-04,   4.21443518e-04,   1.52593813e-04,   1.54326510e-04,
   2.74197693e-04,   5.02911978e-04,   5.18385968e-05,   2.74197693e-04,
   4.50568623e-04,   9.51805501e-04,   7.52103282e-04,   4.50568623e-04,
   2.29349917e-09,   2.51006238e-09,   5.76255918e-11,   2.04327155e-09,
   5.09023266e-06,   2.63715901e-06,   7.66192514e-08,   4.53487291e-06,
   2.00000000e+00,   2.00000000e+00,   1.00000000e+00,   6.00000000e+00,
   1.00000000e+00,   6.00000000e+00,   0.00000000e+00,   2.50000000e+01,
   1.00000000e+00,   0.00000000e+00,   1.00000000e+00,   3.00000000e+00,
   1.24784653e+02,   2.43417174e-01,  -2.08748002e-02,   2.94345856e+00,
   1.21176910e+02,   9.97184098e-01,   8.93915057e-01,   2.98951536e-01,
   2.04623923e-01,   1.82113796e-01,   1.44469276e-01,   1.44469276e-01,
   9.97184098e-01,   8.93915057e-01,   2.98951536e-01,   1.82113796e-01,
   2.04623923e-01,  -1.53782099e-01,   5.30539565e-02,  -2.32247368e-01,
  -1.22703046e-01,   2.81881075e-02,   4.11577731e-01,   3.24652283e+02,
   1.08916878e+02,   3.19357269e+02,   8.11669464e+01,   2.16394531e+02,
   5.40187416e+01,  -1.43581676e+00,  -2.39653647e-01,  -2.29332614e+00,
  -9.27352428e-01,  -2.10310316e+00,   6.79154396e-01,   2.22757130e+01,
   1.44275742e+01,   1.05181656e+01,   9.55952740e+00,   9.62076759e+00,
   7.55627775e+00,   6.51207387e-01,  -2.27411222e+00,   1.36353004e+00,
  -1.33861184e-01,  -1.37701899e-01,  -1.39818335e+00,   1.45862350e+02,
   1.04929382e+02,   6.37808800e+01,   5.51425285e+01,   5.20063171e+01,
   4.31481514e+01,   9.87608850e-01,   9.88287687e-01,   8.92324388e-01,
   9.70717967e-01,   9.54866230e-01,   9.67990100e-01};
eventsVec.push_back(tempEventVec);

tempEventVec.clear();
tempEventVec={ 9.26560135e+01,   9.12297485e+02,   6.50671571e-02,   1.55338776e+00,
   0.00000000e+00,   1.38406769e+02,   2.63872147e-01,   6.31640673e-01,
   0.00000000e+00,   1.07439864e+00,   1.02042079e+00,   1.12190962e+00,
   2.62745261e-01,   1.05370474e+00,   3.90635729e-01,  -4.90344130e-02,
   6.93645328e-02,   7.03089163e-02,   1.19539587e+03,   9.55741644e-01,
   7.02182114e-01,   3.91550243e-01,   1.38406769e+02,   5.39315403e-01,
   9.65322554e-01,   1.32681385e-01,   9.17938378e-05,   1.18896596e-01,
   9.55741644e-01,   1.22097015e+00,   1.07439864e+00,   1.51818025e+00,
   2.20694232e+00,   5.96129537e-01,   1.05370474e+00,   1.05370474e+00,
   1.55338776e+00,   9.11863685e-01,   2.53352618e+00,   8.30587540e+01,
  -4.44069296e-01,  -5.12725450e-02,  -5.34493983e-01,  -3.98856997e-01,
   9.12297424e+02,   7.93224487e+02,   7.02182114e-01,   1.42038107e+00,
   2.38520294e+02,   1.38406769e+02,   2.97488190e+02,   3.45858063e+02,
   6.24441109e+01,   6.17735004e+00,   1.38406769e+02,   5.49198151e+01,
   9.26560135e+01,   9.26560135e+01,   1.38406769e+02,   9.55900192e+01,
  -2.54825149e-02,   1.16062794e+01,   1.38406769e+02,   1.19539587e+03,
   1.73619232e+01,  -2.82229137e+00,   2.24810743e+00,   3.61232491e+01,
   1.63406250e+02,   1.48784760e+02,   1.90263702e+02,   8.29496994e+01,
   9.51260805e-01,   2.14609194e+00,   8.05034637e-01,   1.41806352e+00,
   1.55338776e+00,   1.67788911e+00,   1.77577174e+00,   1.81014657e+00,
   1.81014657e+00,   2.45988822e+00,   2.55707550e+00,   2.34476781e+00,
   2.63354111e+00,   2.63354111e+00,   3.02864528e+00,   3.13823581e+00,
   2.88387060e+00,   3.21580863e+00,   3.21580863e+00,   3.02864528e+00,
   3.13823581e+00,   2.88387060e+00,   3.21580863e+00,   3.21580863e+00,
   3.02864528e+00,   3.13823581e+00,   2.88387060e+00,   3.21580863e+00,
   3.21580863e+00,   5.95616937e-01,   1.10339677e+00,   1.82500792e+00,
   1.82500792e+00,   1.94387943e-01,   6.67111933e-01,   1.04947722e+00,
   1.04947722e+00,   1.50240356e+02,   1.50240356e+02,   5.19048274e-01,
   5.29605746e-01,   2.62849092e-01,   2.62849092e-01,   7.80087769e-01,
   7.45482385e-01,   5.98929971e-02,   5.98929971e-02,   8.32155466e-01,
   7.99925864e-01,   8.17629918e-02,   8.17629918e-02,   7.66731799e-01,
   7.22340286e-01,   1.51585251e-01,   1.51585251e-01,   8.21237087e-01,
   7.80274987e-01,   1.99820235e-01,   1.99820235e-01,   3.50661397e-01,
   3.59631181e-01,   7.27803171e-01,   1.02884178e-05,   2.26789476e-09,
   4.85890731e-02,   1.29918039e-01,   5.55603947e-06,   1.27364996e-09,
   2.08107698e-10,   2.52036059e-10,   2.08107698e-10,   2.08107698e-10,
   1.48896659e-10,   1.84639012e-10,   1.48896659e-10,   1.48896659e-10,
   1.00420982e-06,   9.62587137e-07,   1.00420982e-06,   1.00420982e-06,
   2.07235280e-04,   2.61831941e-04,   2.07235280e-04,   2.07235280e-04,
   1.48272462e-04,   1.91815372e-04,   1.48272462e-04,   1.48272462e-04,
   6.81163801e-04,   6.81163801e-04,   3.70264788e-05,   3.70264788e-05,
   7.38213990e-10,   7.38213990e-10,   1.32582721e-11,   1.32582721e-11,
   1.08375400e-06,   1.08375400e-06,   3.58075425e-07,   3.58075425e-07,
   3.00000000e+00,   2.00000000e+00,   2.00000000e+00,   6.00000000e+00,
   1.00000000e+00,   6.00000000e+00,   0.00000000e+00,   1.90000000e+01,
   1.00000000e+00,   0.00000000e+00,   1.00000000e+00,   3.00000000e+00,
   8.30587540e+01,  -5.12725450e-02,  -2.54825149e-02,   2.24810743e+00,
   8.29496994e+01,   9.74903464e-01,   9.55741644e-01,   7.44514585e-01,
   2.96177775e-01,   1.45658270e-01,   1.18896596e-01,   7.44514585e-01,
   1.45658270e-01,   9.74903464e-01,   9.55741644e-01,   1.18896596e-01,
   2.96177775e-01,   1.37607078e-03,   1.47908509e-01,  -5.36438286e-01,
   9.94287506e-02,   3.04107994e-01,  -1.72343060e-01,   3.66970673e+02,
   2.49756302e+02,   9.20497665e+01,   1.38543030e+02,   1.99386063e+02,
   8.29505463e+01,   2.54094869e-01,  -1.28966832e+00,   2.70534074e-03,
  -1.07169330e+00,  -1.54081357e+00,   9.80959237e-01,   2.80072460e+01,
   1.70217991e+01,   1.43810177e+01,   8.83154106e+00,   1.44730921e+01,
   9.94555283e+00,  -9.98603046e-01,  -3.05339980e+00,   8.63452613e-01,
   1.98536217e+00,   2.35318899e+00,  -2.41588756e-01,   3.54397980e+02,
   1.27555237e+02,   9.09191208e+01,   8.47513962e+01,   8.14567413e+01,
   5.41440659e+01,   9.82985854e-01,   9.84790504e-01,   9.87612724e-01,
   9.93693769e-01,   9.73564386e-01,   9.77695644e-01};
eventsVec.push_back(tempEventVec);

tempEventVec.clear();
tempEventVec={ 3.57774200e+02,   7.68042786e+02,   9.19603482e-02,   3.41006255e+00,
   0.00000000e+00,   3.55093567e+02,  -8.32444727e-01,   2.56481934e-02,
   0.00000000e+00,   1.62641633e+00,   8.82317424e-01,   3.28597069e+00,
   2.32848024e+00,   6.72775924e-01,   3.91853064e-01,  -8.01468268e-02,
   7.54411623e-04,   3.24057043e-02,   1.06714172e+03,   8.66264045e-01,
   7.15622544e-01,   6.82696819e-01,  -9.90000000e+01,   4.97480482e-01,
   9.31496143e-01,   1.10725440e-01,   4.25523054e-03,   1.18173130e-01,
   8.66264045e-01,   1.25610018e+00,   1.62641633e+00,   1.24996376e+00,
   2.37492681e+00,   1.04629314e+00,   6.72775924e-01,   2.49004102e+00,
   3.41006255e+00,   2.22326899e+00,   2.52538466e+00,   2.56925293e+02,
  -1.29373342e-01,  -1.30457366e+00,   3.90951931e-01,  -3.89535993e-01,
   7.68042786e+02,   5.55740479e+02,   7.15622485e-01,   1.65616703e+00,
   1.78600784e+02,   3.55093567e+02,   1.41868698e+02,   1.90960251e+02,
   2.91492386e+01,   4.55150270e+00,   3.55093567e+02,   1.16718094e+02,
   6.51204453e+01,   3.57774200e+02,   3.55093567e+02,   1.11036674e+02,
   1.05092570e-01,   1.45676556e+01,   3.55093567e+02,   1.06714172e+03,
   9.78430653e+00,  -1.25301465e-01,   1.17723644e-01,   8.24569778e+01,
   2.74505859e+02,   1.61158722e+02,   7.19546738e+01,   1.29845306e+02,
   8.34317923e-01,   2.83057642e+00,   3.47313076e-01,  -6.30870998e-01,
   0.00000000e+00,   1.87278461e+00,   2.27002954e+00,   2.61475492e+00,
   2.61475492e+00,   2.77364224e-01,   1.80752003e+00,   2.02231669e+00,
   2.43112350e+00,   2.43112350e+00,   1.28681970e+00,   2.06207633e+00,
   1.36162806e+00,   2.11008286e+00,   2.11008286e+00,   1.28681970e+00,
   2.06207633e+00,   1.36162806e+00,   2.11008286e+00,   2.11008286e+00,
   1.28681970e+00,   2.06207633e+00,   1.36162806e+00,   2.11008286e+00,
   2.11008286e+00,   2.17515659e+00,   2.17515659e+00,   1.71959567e+00,
   1.71959567e+00,   2.75140595e+00,   2.75140595e+00,   3.10912156e+00,
   1.15060449e+00,   2.78614807e+02,   9.36738892e+01,   6.64088488e-01,
   6.64088488e-01,   2.47431751e-02,   4.92230356e-01,   6.65085077e-01,
   6.65085077e-01,   7.76926726e-02,   1.25460565e-01,   4.92098153e-01,
   4.92098153e-01,   5.27106881e-01,   6.54147938e-02,   5.01117706e-01,
   5.01117706e-01,   7.68530786e-01,   1.28910661e-01,   3.28897059e-01,
   3.28897059e-01,   9.77745056e-01,   6.73408136e-02,   6.01968825e-01,
   5.10181904e-01,   6.55264318e-01,   1.90575120e-05,   3.76974452e-09,
   7.41475150e-02,   1.40937582e-01,   2.88219344e-05,   3.92646893e-09,
   3.49032997e-10,   3.49032997e-10,   3.25534669e-11,   3.24229033e-10,
   7.15380755e-10,   7.15380755e-10,   2.46017464e-12,   6.64542366e-10,
   1.37493362e-06,   1.37493362e-06,   2.10143995e-07,   1.27722421e-06,
   2.53854436e-04,   2.53854436e-04,   1.54910289e-04,   2.53854436e-04,
   5.20302041e-04,   5.20302041e-04,   1.17070904e-05,   5.20302041e-04,
   2.54991901e-04,   2.54991901e-04,   5.14337560e-04,   3.75673808e-05,
   6.93121227e-10,   6.93121227e-10,   2.74221596e-12,   4.65135777e-11,
   2.71820886e-06,   2.71820886e-06,   5.33154898e-09,   1.23813732e-06,
   2.00000000e+00,   2.00000000e+00,   1.00000000e+00,   6.00000000e+00,
   1.00000000e+00,   8.00000000e+00,   0.00000000e+00,   9.00000000e+00,
   1.00000000e+00,   0.00000000e+00,   1.00000000e+00,   3.00000000e+00,
   2.56925293e+02,  -1.30457366e+00,   1.05092570e-01,   1.17723644e-01,
   1.29845306e+02,   9.96728301e-01,   8.66264045e-01,   4.88363177e-01,
   3.58012617e-01,   1.57341525e-01,   1.18173130e-01,   9.96728301e-01,
   3.58012617e-01,   8.66264045e-01,   4.88363177e-01,   1.18173130e-01,
   1.57341525e-01,   7.94883370e-02,   1.68729410e-01,   1.01900190e-01,
   4.67274427e-01,   3.68837029e-01,  -4.95854914e-02,   2.55274490e+02,
   1.22386543e+02,   1.34811050e+02,   1.32336044e+02,   9.23027420e+01,
   3.94724197e+01,  -4.22256202e-01,   8.35525393e-01,   1.20416009e+00,
  -1.24410534e+00,  -1.20522690e+00,   5.56629263e-02,   1.37435656e+01,
   1.35847063e+01,   1.53917456e+01,   9.45126629e+00,   8.65157509e+00,
   7.44968033e+00,  -2.21075654e+00,   2.16516781e+00,   1.07521403e+00,
   7.87776649e-01,  -2.90479279e+00,  -1.01597703e+00,   2.33753616e+02,
   8.87918472e+01,   7.37111359e+01,   7.02486115e+01,   5.05322189e+01,
   3.87030792e+01,   9.97332335e-01,   9.90880072e-01,   9.84979630e-01,
   9.92297649e-01,   9.81479943e-01,   9.67972279e-01};
eventsVec.push_back(tempEventVec);

return eventsVec;
}

std::vector<std::vector<float>> createVectorOfOutputValues()
{
std::vector<std::vector<float>> outputValuesVec;
std::vector<float> tempOutputValuesVec;
tempOutputValuesVec.clear();
tempOutputValuesVec={ 0.59879774,  0.2098868 ,  0.13719904,  0.05411637 };
outputValuesVec.push_back(tempOutputValuesVec);

tempOutputValuesVec.clear();
tempOutputValuesVec={ 0.32654387,  0.27284592,  0.20312098,  0.19748923 };
outputValuesVec.push_back(tempOutputValuesVec);

tempOutputValuesVec.clear();
tempOutputValuesVec={ 0.13048398,  0.28502944,  0.23853934,  0.34594727 };
outputValuesVec.push_back(tempOutputValuesVec);

return outputValuesVec;
}
