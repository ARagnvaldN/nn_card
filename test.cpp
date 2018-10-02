#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// NxKxHxW
// TODO: Data layer
// TODO: Load weights
// TODO: Format for reading network (LOW)

static const int INNER_PRODUCT = 0;
static const int CONVOLUTIONAL = 1;
static const int MAX_POOL 	 = 2;
static const int DATA 		 = 3;

struct layer_params {
    float * data;
    int input_channels;
    int output_channels;
    int input_shape;
    int output_shape;
    int type;
    int kernel_size;
    int stride;
    bool relu;
};

struct layer{
    layer(): input(0), output(0), type(INNER_PRODUCT), bias(nullptr), weights(nullptr) {}
    layer(const layer_params & params)
	   : bias(nullptr),
          weights(nullptr),
          activations(nullptr),
		shape(params.output_shape),
		channels(params.output_channels),
		type(params.type),
		relu(params.relu)
    {
        std::random_device rd {};
        std::mt19937 gen{rd()};
        //std::mt19937 gen{0};
        std::normal_distribution<> nd(0, 1);

        // std::cout << "Creating new layer: " << this << std::endl;
	   input = params.input_channels
			  * params.input_shape
			  * params.input_shape;
	   output = params.output_channels
	 		   * params.output_shape
		 	   * params.output_shape;
	   switch (params.type) 
	   {
	       case INNER_PRODUCT:
			{
			    int n_bias = params.input_channels;
			    bias = new float [n_bias];
			    for (int i = 0; i < n_bias; ++i) {
				   bias[i] = nd(gen);
			    }

			    int n_weights = params.input_channels * params.output_channels;
			    weights = new float [n_weights];
			    for (int i = 0; i < n_weights; ++i) {
				   weights[i] = nd(gen);
			    }

			    int n_activations = params.output_channels;
			    activations = new float [n_activations];
			    for (int i = 0; i < n_activations; ++i) {
				   activations[i] = 0.0f;
			    }
			}
			break;
		  
		  case CONVOLUTIONAL:
			//INPUT_channels x OUTPUT_channels x HEIGHT x WIDTH
			// x x x o o
			// x x x o o
			// x x x o o
			// o o o o o
			// o o o o o
			//
			{
			    int n_bias = params.input_channels;
			    bias = new float [n_bias];
			    for (int i = 0; i < n_bias; ++i) {
				   bias[i] = nd(gen);
			    }

			    int n_weights = params.input_channels
						  * params.output_channels
						  * params.kernel_size
						  * params.kernel_size;
			    weights = new float [n_weights];
			    for (int i = 0; i < n_weights; ++i) {
				   weights[i] = nd(gen);
			    }

			    int n_activations = params.output_channels
				   				* (params.input_shape
								   - (params.kernel_size - 1))
				   				* (params.input_shape
								   - (params.kernel_size - 1));
			    activations = new float [n_activations];
			    for (int i = 0; i < n_activations; ++i) {
				   activations[i] = 0.0f;
			    }
			}
			break;

		  case MAX_POOL:
			// Pooling layers have no weights or bias!
			{
			    int n_activations = params.input_channels
			    				     * (params.input_shape / 2)
				    				* (params.input_shape / 2);
			    activations = new float [n_activations];
			    for (int i = 0; i < n_activations; ++i) {
				   activations[i] = 0.0f;
			    }
			}
			break;

		  case DATA:
			// Data layers have no weights or bias!
			{
			    int n_activations = params.input_channels
			       			     * params.input_shape
							     * params.input_shape;
			    activations = new float [n_activations];
        		    std::copy(params.data, params.data + n_activations, activations);
			}
			break;
	   }
    }
    ~layer()
    {
        // std::cout << "Layer destructor: " << this << std::endl;
        // std::cout << "Bias: " <<  bias << std::endl;
        // std::cout << "Weights: " <<  weights << std::endl;
	   
	   delete[] bias;
	   delete[] weights;
	   delete[] activations;
    }

    layer(const layer & other)
        : input(other.input),
          output(other.output),
          bias(new float [output]),
          weights(new float [input * output]) {
        std::cout << "Copy constructor!" << std::endl;

        std::copy(other.bias, other.bias + output, bias);
        std::copy(other.weights, other.weights + output, weights);
    }

    int channels;
    int input;
    int output;
    int shape;
    int type;
    bool relu;
    float * bias;
    float * weights;
    float * activations;
};

struct network{
    network(layer_params arr [], int size)
	   : num_layers(size),
	     layers(new layer * [size])
    {
        for (int i = 0; i < size; ++i) {
            layers[i] = new layer(arr[i]);
        }
    }
    ~network()
    {
        for (int i = 0; i < num_layers; ++i) {
            delete layers[i];           
        }
        delete[] layers;
    }
    
    layer** layers;
    int num_layers;
};

// Deprecated
void print(const network & net)
{
    for (int i = 0; i < net.num_layers; ++i) {
        std::cout << net.layers[i]->input << "x"
                  << net.layers[i]->output << " weights:"
                  << std::endl;
        for (int x = 0; x < net.layers[i]->input; ++x) {
            for (int y = 0; y < net.layers[i]->output; ++y) {
                std::cout << net.layers[i]->weights[x * net.layers[i]->output + y]
                          << " ";
            }
            std ::cout << std::endl;
        }

        std::cout << net.layers[i]->output << " bias:"
                  << std::endl;
        for (int y = 0; y < net.layers[i]->output; ++y) {
            std::cout << net.layers[i]->bias[y]
                      << " ";
        }
        std ::cout << std::endl << std::endl;
    }
}

void inner_product(layer * input, layer * output)
{
    for (int outer = 0; outer < output->channels; ++outer) {

	   float sum = 0;
	   for (int inner = 0; inner < input->channels; ++inner) {

		  sum += input->activations[inner]
			 	* output->weights[inner * output->channels + outer];
	   }

	   // Apply bias and ReLU
	   // Hinge on activation type and bias type
	   float result = sum + output->bias[outer];
	   if (output->relu) {
		  if (result > 0.0f)
		  	 output->activations[outer] = result;
	   } else {
		  output->activations[outer] = sum + output->bias[outer];
	   }
    }
}

void max_pool(layer * current_layer, float * input)
{
    // Hardcode to 2x2 for now...
    //
    // return i > j? (i > k? i: k): (j > k? j: k);
}

void convolution(layer * current_layer, float * input)
{

}

int forward(const network & net)
{

    if (net.layers[0]->type != DATA) {
        std::cout << "First layer is not a DATA layer!" << std::endl;
        return -1;
    } 

    // For each layer
    for (int i = 1; i < net.num_layers; ++i) {

	   layer * last_layer = net.layers[i - 1];
	   layer * current_layer = net.layers[i];
        
        // Calculate the dot product
	   if (current_layer->type == INNER_PRODUCT) {

		inner_product(last_layer, current_layer);

	   } else if (current_layer->type == CONVOLUTIONAL) {
	   	  // TODO: Convolutional logic
		  //
	   } else if (current_layer->type == MAX_POOL) {
		  // TODO: Max pooling logic
	   }

        // Print activations
        std::cout << std::endl << "Activations: ";
        for (int j = 0; j < current_layer->channels; ++j) {
            std::cout << current_layer->activations[j] << " ";
        }
        std::cout << std::endl << std::endl;
    }

    // ArgMax of final activation
    int last = net.num_layers - 1;
    float max = net.layers[last]->activations[0];
    int argmax = 0;
    for (int i = 1; i < net.layers[last]->channels; ++i) {
        if (net.layers[last]->activations[i] > max) {
            argmax = i;
            max = net.layers[last]->activations[i];
        }
    }

    return argmax;
}

int main()
{
    float data [10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};

    layer_params net_spec[6] = {};
    net_spec[0].output_shape = 1;
    net_spec[0].output_channels = 10;
    net_spec[0].data = data;
    net_spec[0].type = DATA;

    net_spec[1].input_shape = 1;
    net_spec[1].input_channels = 10;
    net_spec[1].output_shape = 1;
    net_spec[1].output_channels = 8;
    net_spec[1].relu = true;

    net_spec[2].input_shape = 1;
    net_spec[2].input_channels = 8;
    net_spec[2].output_shape = 1;
    net_spec[2].output_channels = 5;
    net_spec[2].relu = true;

    net_spec[3].input_shape = 1;
    net_spec[3].input_channels = 5;
    net_spec[3].output_shape = 1;
    net_spec[3].output_channels = 5;
    net_spec[3].relu = true;

    net_spec[4].input_shape = 1;
    net_spec[4].input_channels = 5;
    net_spec[4].output_shape = 1;
    net_spec[4].output_channels = 5;
    net_spec[4].relu = true;

    net_spec[5].input_shape = 1;
    net_spec[5].input_channels = 5;
    net_spec[5].output_shape = 1;
    net_spec[5].output_channels = 4;

    network net = network(net_spec, 6);

    int class_ = forward(net);
    std::cout << "Classification: " << class_ << std::endl << std::endl;

    return 0;
}
