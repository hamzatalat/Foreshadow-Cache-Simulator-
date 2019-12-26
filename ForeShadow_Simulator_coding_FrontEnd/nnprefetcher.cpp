#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include </home/hammad/fypcoding/mdnnlib/include/MiniDNN.h>
using namespace MiniDNN;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;
class neuralnetp{
	private:
	Network net;
	int len;// specifies the number of next best possible addresses which could be asked for by cpu
	public:
	neuralnetp(int lenght=4){
		len=lenght;
	}
	void trainnet(string filex,string filey,unsigned int rows){// keeping the structure of nn same for now - rows is max number of data rows to use for training
		// Layer  -- fully connected, input size XxYxZ, output size n
		Layer* layer1=new FullyConnected<Identity>(5*1*1,4);
		Layer* layer2=new FullyConnected<Identity>(4*1*1,3);
		Layer* layer3=new FullyConnected<Identity>(3*1*1,2);
		Layer* layer4=new FullyConnected<Identity>(2*1*1,1);
		// Add layers to the network object
		net.add_layer(layer1);
		net.add_layer(layer2);
		net.add_layer(layer3);
		net.add_layer(layer4);
		// Set output layer
		net.set_output(new RegressionMSE());
		// Create optimizer object
		RMSProp opt;
		opt.m_lrate = 0.001;
		// (Optional) set callback function object
		VerboseCallback callback;
		net.set_callback(callback);
		// Initialize parameters with N(0, 0.01^2) using random seed 123
		net.init(0, 0.01, 123);
		// Fit the model with a batch size of 100, running 10 epochs with random seed 123
		net.fit(opt, x, y, 100, 10, 123);// need to read data and convert into matrix
		//Matrix pred = net.predict(x);
	}
	void predictn(string address, string* result){// need to add addition logic and conversion here
		/*unsigned long  setnum=stol(address,0,16),temp;
		for(int i=0;i<len;i++){
			temp=setnum+1+i;
			stringstream sstream;
			sstream<<std::hex<<temp;
			result[i]=sstream.str();
		}*/
		
	}
	int getsize(){
		return len;
	}	
};

int main()
{
    // Set random seed and generate some data
    std::srand(123);
    // Predictors -- each column is an observation
    Matrix x = Matrix::Random(400, 100);
    // Response variables -- each column is an observation
    Matrix y = Matrix::Random(2, 100);
    // Construct a network object
    Network net;
    // Create three layers
    // Layer 1 -- convolutional, input size 20x20x1, 3 output channels, filter size 5x5
    Layer* layer1 = new Convolutional<ReLU>(20, 20, 1, 3, 5, 5);
    // Layer 2 -- max pooling, input size 16x16x3, pooling window size 3x3
    Layer* layer2 = new MaxPooling<ReLU>(16, 16, 3, 3, 3);
    // Layer 3 -- fully connected, input size 5x5x3, output size 2
    Layer* layer3 = new FullyConnected<Identity>(5 * 5 * 3, 2);
    // Add layers to the network object
    net.add_layer(layer1);
    net.add_layer(layer2);
    net.add_layer(layer3);
    // Set output layer
    net.set_output(new RegressionMSE());
    // Create optimizer object
    RMSProp opt;
    opt.m_lrate = 0.001;
    // (Optional) set callback function object
    VerboseCallback callback;
    net.set_callback(callback);
    // Initialize parameters with N(0, 0.01^2) using random seed 123
    net.init(0, 0.01, 123);
    // Fit the model with a batch size of 100, running 10 epochs with random seed 123
    net.fit(opt, x, y, 100, 10, 123);
    // Obtain prediction -- each column is an observation
    Matrix pred = net.predict(x);
    // Layer objects will be freed by the network object,
    // so do not manually delete them
    return 0;
}
