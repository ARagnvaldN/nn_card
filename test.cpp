#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>

// NxKxHxW
// TODO: Allow preallocating bias and weights
// TODO: Load weights
// TODO: Format for reading network (LOW)

static const int INNER_PRODUCT = 0;
static const int CONVOLUTIONAL = 1;
static const int MAX_POOL 	 = 2;
static const int DATA 		 = 3;
static const char* type2string [4] {"Inner product",
    						      "Convolution",
						      "Max pooling",
						      "Data"};

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
    layer(const layer_params & params)
	   : bias(nullptr),
          weights(nullptr),
          activations(nullptr),
		shape(params.output_shape),
		channels(params.output_channels),
		type(params.type),
		relu(params.relu),
		kernel_size(params.kernel_size)
    {
        std::random_device rd {};
        std::mt19937 gen{rd()};
        //std::mt19937 gen{0};
        std::normal_distribution<> nd(0, 1);

        // std::cout << "Creating new layer: " << this << std::endl;
	   //
	   switch (params.type) 
	   {
	       case INNER_PRODUCT:
			{
			    shape = 1;

			    int n_bias = channels;
			    bias = new float [n_bias];
			    for (int i = 0; i < n_bias; ++i) {
				   bias[i] = 0.f;  // nd(gen);
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
			{
			    shape = params.input_shape - (params.kernel_size - 1);

			    int n_bias = params.output_channels;
			    bias = new float [n_bias];
			    for (int i = 0; i < n_bias; ++i) {
				   bias[i] = 0;  // nd(gen);
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
			    shape = params.input_shape / 2;
			    channels = params.input_channels;

			    int n_activations = params.input_channels * shape * shape;
			    activations = new float [n_activations];
			    for (int i = 0; i < n_activations; ++i) {
				   activations[i] = 0.0f;
			    }
			}
			break;

		  case DATA:
			// Data layers have no weights or bias!
			{
			    int n_activations = params.output_channels
			       			     * params.output_shape
							     * params.output_shape;
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
    {
        std::cout << "Copy constructor!" << std::endl;
    }

    int channels;
    int shape;
    int kernel_size;
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

void print(const network & net)
{
    for (int i = 0; i < net.num_layers; ++i) {
        std::cout << "Type: " << type2string[net.layers[i]->type] << std::endl;
        std::cout << "Shape: "
		  	   << net.layers[i]->channels << "x"
		  	   << net.layers[i]->shape << "x"
			   << net.layers[i]->shape
			   << std::endl;
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

void max_pool(layer * input, layer * output)
{
    // Hardcode to 2x2 for now...
    //
    // x x i i  i i i i  i i i i
    // x x i i  i i i i  i i i i
    // i i i i  i i i i  i i i i
    // i i i i  i i i i  i i i i
    //
    int kernel_size = 2;
    for (int c_out = 0; c_out < output->channels; ++c_out) {
	   for (int h_out = 0; h_out < output->shape; ++h_out) {
		  for (int w_out = 0; w_out < output->shape; ++w_out) {

			 // Loop over inside of kernel
			 float max = -10000;
			 for (int y_kernel = 0; y_kernel < kernel_size; ++y_kernel) {
				for (int x_kernel = 0; x_kernel < kernel_size; ++x_kernel) {
				    float current_val = input->activations[c_out * input->shape
					   								    * input->shape
								   				   + h_out * kernel_size
												   		 * input->shape
								   				   + w_out * kernel_size];
				    if (current_val > max)
					   max = current_val;
				}
			 }
			 output->activations[c_out * output->shape * output->shape
							 + h_out * output->shape
							 + w_out] = max;
		  }
	   }
    }
    // return i > j? (i > k? i: k): (j > k? j: k);
}

void convolution(layer * input, layer * output)
{
    // input: INPUT_channels x INPUT_height x INPUT_width
    // output: OUTPUT_channels x OUTPUT_height x OUTPUT_width
    // weights: INPUT_channels x OUTPUT_channels x kernel_HEIGHT x kernel_WIDTH
    // bias: OUTPUT_channels
    //
    // x x x i i  i i i i i  i i i i i
    // x x x i i  i i i i i  i i i i i
    // x x x i i  i i i i i  i i i i i
    // i i i i i  i i i i i  i i i i i
    // i i i i i  i i i i i  i i i i i
    //
    // x o o  o o o  o o o  o o o
    // o o o  o o o  o o o  o o o
    // o o o  o o o  o o o  o o o
    //
    int kernel_size = output->kernel_size;

    //for (int j = 0; j < 28; ++j) {
    //    for (int i = 0; i < 1; ++i) {
    // 	  std::cout << "(" << i << ", " << j << "): "
    // 		 	  << input->activations[j * 28 + i] << std::endl;
    //    }
    //}
    for (int c_outer = 0; c_outer < output->channels; ++c_outer) {
        for (int h_outer = 0; h_outer < output->shape; ++h_outer) {
        	  for (int w_outer = 0; w_outer < output->shape; ++w_outer) {
        
        		float sum = 0.f;
        		for (int c_inner = 0; c_inner < input->channels; ++c_inner) {
    
      		    for (int kernel_y = 0; kernel_y < kernel_size; ++kernel_y) {
    				    for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x) {
    			    	    
    			    	    sum += input->activations[c_inner * input->shape * input->shape
    			    	    					    + (h_outer + kernel_y) * input->shape
    			    	    					    + w_outer + kernel_x]
    			    	    		  * output->weights[c_inner * output->channels
    			    	    		  					   * kernel_size * kernel_size
    			    	    		  				+ c_outer * kernel_size * kernel_size
    			    	    		  				+ kernel_y * kernel_size
    			    	    		  				+ kernel_x];
    
   			        }
      		    }
			    //std::cout << "Sum: " << sum << std::endl;
			    //std::cout << "Bias: " << output->bias[c_outer] << std::endl;
                   output->activations[c_outer * output->shape * output->shape
                     		         + h_outer * output->shape
                   	     	         + w_outer] = sum + output->bias[c_outer];
			}
		  }
	   }
    }
    
}

int forward(const network & net)
{

    if (net.layers[0]->type != DATA) {
        std::cout << "First layer is not a DATA layer!" << std::endl;
        return -1;
    } 

    //for (int j = 0; j < 28; ++j) {
    //    for (int i = 0; i < 1; ++i) {
    // 	  std::cout << "(" << i << ", " << j << "): "
    // 		 	  << net.layers[0]->activations[j * 28 + i] << std::endl;
    //    }
    //}

    // For each layer
    for (int i = 1; i < net.num_layers; ++i) {

	   layer * last_layer = net.layers[i - 1];
	   layer * current_layer = net.layers[i];
        
        // Calculate the dot product
	   if (current_layer->type == INNER_PRODUCT) {

		  inner_product(last_layer, current_layer);

	   } else if (current_layer->type == CONVOLUTIONAL) {

		  convolution(last_layer, current_layer);

	   } else if (current_layer->type == MAX_POOL) {

		  max_pool(last_layer, current_layer);

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

int write_layer_spec(const char* file_name, const layer_params * net_spec, int * size) {

    std::ofstream out(file_name);
    out << *size;
    for (int i = 0; i < *size; ++i) {
        out.write(reinterpret_cast<const char *>(& net_spec[i]), sizeof(layer_params));
    }

    std::cout << "Done writing " << *size << " layers!" << std::endl;
    return true;
}
int read_layer_spec(const char* file_name, layer_params * net_spec, int * size) {

    std::ifstream in(file_name);
    in >> *size;
    for (int i = 0; i < *size; ++i) {
        in.read(reinterpret_cast<char *>(& net_spec[i]), sizeof(layer_params));
    }

    std::cout << "Done reading " << *size << " layers!" << std::endl;
    return true;
}

int main()
{
    {
    float data [784] = 
    {
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    };

    
    int num_layers = 7;
    layer_params net_spec[7] = {};
    net_spec[0].type = DATA;
    net_spec[0].data = data;
    net_spec[0].output_shape = 28;
    net_spec[0].output_channels = 1;

    net_spec[1].type = CONVOLUTIONAL;
    net_spec[1].input_shape = 28;
    net_spec[1].input_channels = 1;
    net_spec[1].kernel_size = 5;
    net_spec[1].output_channels = 20;

    net_spec[2].type = MAX_POOL;
    net_spec[2].kernel_size = 2;
    net_spec[2].input_channels = 20;
    net_spec[2].input_shape = 24;

    net_spec[3].type = CONVOLUTIONAL;
    net_spec[3].input_channels = 20;
    net_spec[3].input_shape = 12;
    net_spec[3].kernel_size = 5;
    net_spec[3].output_channels = 50;

    net_spec[4].type = MAX_POOL;
    net_spec[4].kernel_size = 2;
    net_spec[4].input_channels = 50;
    net_spec[4].input_shape = 8;

    net_spec[5].type = INNER_PRODUCT;
    net_spec[5].output_channels = 500;
    net_spec[5].relu = true;

    net_spec[6].type = INNER_PRODUCT;
    net_spec[6].output_channels = 10;

    std::cout << "Writing" << std::endl;
    write_layer_spec("lenet.bin", net_spec, &num_layers);

    net_spec[1].type = CONVOLUTIONAL;
    net_spec[1].input_shape = 0;
    net_spec[1].input_channels = 0;
    net_spec[1].kernel_size = 0;
    net_spec[1].output_channels = 0;

    std::cout << "reading" << std::endl;
    read_layer_spec("lenet.bin", net_spec, &num_layers);

    network net = network(net_spec, num_layers);
    print(net);

    int class_ = forward(net);
    std::cout << "Classification: " << class_ << std::endl << std::endl;
    }
    return 0;
}
