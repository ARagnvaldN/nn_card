#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>

static const int INNER_PRODUCT = 0;
static const int CONVOLUTIONAL = 1;
static const int MAX_POOL 	 = 2;
static const int DATA 		 = 3;
static const char* type2string [4] {"Inner product",
    						      "Convolution",
						      "Max pooling",
						      "Data"};

struct layer {

    int channels;
    int shape;
    int kernel_size;
    int type;
    int stride;
    bool relu;

    float * data;
    float * weights;
    float * bias;
    float * activations;

    int n_bias;
    int n_weights;
    int n_activations;
};

void free_layer(layer * l) {

    if(l->data == nullptr) {
	   delete[] l->bias;
	   delete[] l->weights;
    }
    delete[] l->activations;
}

void init_layer(layer * previous, layer * l) {

    std::random_device rd {};
    std::mt19937 gen{rd()};
    std::normal_distribution<> nd(0, 1);

    switch (l->type) 
    {
	   case INNER_PRODUCT:
		 {
			l->shape = 1;

			l->n_bias = l->channels;
			l->n_weights = previous->channels 
			    			* previous->shape 
						* previous->shape
						* l->channels;
			l->n_activations = l->channels;

			if (l->data == nullptr) {

			    l->bias = new float [l->n_bias];
			    for (int i = 0; i < l->n_bias; ++i) {
				   l->bias[i] =  nd(gen);
			    }

			    l->weights = new float [l->n_weights];
			    for (int i = 0; i < l->n_weights; ++i) {
				   l->weights[i] = nd(gen);
			    }


			} else {

			    l->weights = l->data;
			    l->bias = l->data + l->n_weights;

			}

			l->activations = new float [l->n_activations];
			for (int i = 0; i < l->n_activations; ++i) {
			    l->activations[i] = 0.0f;
			}
		 }
		 break;
	   
	   case CONVOLUTIONAL:
		 {
			l->shape = previous->shape - (l->kernel_size - 1);

			l->n_bias = l->channels;
			l->n_weights = previous->channels
					   * l->channels
					   * l->kernel_size
					   * l->kernel_size;


			if (l->data == nullptr) {

			    l->bias = new float [l->n_bias];
			    for (int i = 0; i < l->n_bias; ++i) {
				   l->bias[i] = 0;  // nd(gen);
			    }
			    l->weights = new float [l->n_weights];
			    for (int i = 0; i < l->n_weights; ++i) {
				   l->weights[i] = nd(gen);
			    }

			} else {
				l->weights = l->data;
				l->bias = l->data + l->n_weights;
			}

			l->n_activations = l->channels
							* (previous->shape - (l->kernel_size - 1))
							* (previous->shape - (l->kernel_size - 1));
			l->activations = new float [l->n_activations];
			for (int i = 0; i < l->n_activations; ++i) {
			    l->activations[i] = 0.0f;
			}


		 }
		 break;

	   case MAX_POOL:
		 // Pooling layers have no weights or bias!
		 {
			l->shape = previous->shape / 2;
			l->channels = previous->channels;

			l->n_activations = previous->channels * l->shape * l->shape;
			l->activations = new float [l->n_activations];
			for (int i = 0; i < l->n_activations; ++i) {
			    l->activations[i] = 0.0f;
			}
		 }
		 break;

	   case DATA:
		 // Data layers have no weights or bias!
		 {
			l->n_activations = l->channels
							 * l->shape
							 * l->shape;
			l->activations = new float [l->n_activations];
			std::copy(l->data, l->data + l->n_activations, l->activations);
		 }
		 break;
    }
}

void print(layer * network, int size)
{
    for (int i = 0; i < size; ++i) {
	   layer * current_layer = network + i;
        std::cout << "Type: " << type2string[network[i].type] << std::endl;
        std::cout << "Shape: "
		  	   << network[i].channels << "x"
		  	   << network[i].shape << "x"
			   << network[i].shape
			   << std::endl;
	   if (network[i].weights) {
		     std::cout << "n_weights: " << network[i].n_weights << std::endl;
			std::cout << "weights: ";
			for(int j = 0; j < 10; ++j) {
				std::cout << network[i].weights[j] << " ";
			}
			std::cout << std::endl;
		     std::cout << "n_bias: " << network[i].n_bias << std::endl;
			std::cout << "bias: ";
			for(int j = 0; j < 10; ++j) {
				std::cout << network[i].bias[j] << " ";
			}
			std::cout << std::endl;
	   }

	   std::cout << std::endl << "Activations: " << std::endl;
	   if (current_layer->type == CONVOLUTIONAL || current_layer->type == MAX_POOL) {
		  for (int c = 1; c < 2; ++c) {
			 for (int h = 0; h < current_layer->shape; ++h) {
				for (int w = 0; w < current_layer->shape; ++w) {
				   std::cout << current_layer->activations[c * current_layer->shape 
													* current_layer->shape
												   + h * current_layer->shape
												   + w] << " ";
				}
				std::cout << std::endl;
			 }
			 std::cout << std::endl;
		  }
		  std::cout << std::endl << std::endl;
	   } else if (current_layer->type == INNER_PRODUCT) {
		  for (int i = 0; i < 10; ++i) {
			 std::cout << current_layer->activations[i] << " ";
		  }
		  std::cout << std::endl;
	   }
    }
}

void inner_product(layer * input, layer * output)
{
    for (int outer = 0; outer < output->channels; ++outer) {

	   float sum = 0;
	   for (int inner = 0; inner < input->channels; ++inner) {
		  for (int y = 0; y < input->shape; ++y) {
			 for (int x = 0; x < input->shape; ++x) {

			 sum += input->activations[inner * input->shape * input->shape
								  + y * input->shape
								  + x]
				    * output->weights[outer * input->channels * input->shape * input->shape
				    				  + inner * input->shape * input->shape
								  + y * input->shape
								  + x];
	   		}
		  }
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
    for (int c_out = 0; c_out < output->channels; ++c_out) {
	   for (int h_out = 0; h_out < output->shape; ++h_out) {
		  for (int w_out = 0; w_out < output->shape; ++w_out) {

			 // Loop over inside of kernel
			 float max = -10000.0f;
			 for (int y_kernel = 0; y_kernel < output->kernel_size; ++y_kernel) {
				for (int x_kernel = 0; x_kernel < output->kernel_size; ++x_kernel) {
				    float current_val = input->activations[c_out * input->shape
					   								    * input->shape
								   				   + (h_out * output->kernel_size
												      + y_kernel)
												   		 * input->shape
								   				   + w_out * output->kernel_size
												   + x_kernel];
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
}

void convolution(layer * input, layer * output)
{
    int kernel_size = output->kernel_size;

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
    			    	    		  * output->weights[c_outer * input->channels
    			    	    		  					   * kernel_size * kernel_size
    			    	    		  				+ c_inner * kernel_size * kernel_size
    			    	    		  				+ kernel_y * kernel_size
    			    	    		  				+ kernel_x];
   			        }
      		    }
			}
               output->activations[c_outer * output->shape * output->shape
                                   + h_outer * output->shape
                   	     	     + w_outer] = sum + output->bias[c_outer];
		  }
	   }
    }
    
}

int forward(layer * last_layer, layer * current_layer)
{
    switch(current_layer->type) {
    case INNER_PRODUCT:
	   inner_product(last_layer, current_layer);
	   break;
    case CONVOLUTIONAL:
	   convolution(last_layer, current_layer);
	   break;
    case MAX_POOL:
	   max_pool(last_layer, current_layer);
	   break;
    default:
	   break;
    }

    return 0;

}

int write_layer_spec(const char* file_name, const layer * net_spec, int * size) {

    std::ofstream out(file_name);
    out << *size;
    for (int i = 0; i < *size; ++i) {
        out.write(reinterpret_cast<const char *>(& net_spec[i]), sizeof(layer));
    }

    std::cout << "Done writing " << *size << " layers!" << std::endl;
    return true;
}
int read_layer_spec(const char* file_name, layer * net_spec, int * size) {

    std::ifstream in(file_name);
    in >> *size;
    for (int i = 0; i < *size; ++i) {
        in.read(reinterpret_cast<char *>(& net_spec[i]), sizeof(layer));
    }

    std::cout << "Done reading " << *size << " layers!" << std::endl;
    return true;
}

void read_binary_float(float * data, const char * filename, unsigned int length)
{
    std::ifstream fin(filename, std::ios::binary);
    fin.read(reinterpret_cast<char *> (data), length * sizeof(float));
}

int main(int c, char ** argv)
{
    const int INPUT_SIZE = 784;
    float * data = new float[INPUT_SIZE];
    read_binary_float(data, argv[1], INPUT_SIZE);

    const int NUM_WEIGHTS = 431080;
    float * weights = new float[NUM_WEIGHTS];
    read_binary_float(weights, argv[2], NUM_WEIGHTS);

    int num_layers = 7;
    layer net_spec[7] = {
	   {1,28,0,DATA, 0, 0, data},
	   {20,0,5,CONVOLUTIONAL, 0, 0, weights},
	   {0,0,2,MAX_POOL},
	   {50,0,5,CONVOLUTIONAL, 0, 0, weights + 520},
	   {0,0,2,MAX_POOL},
	   {500,0,0,INNER_PRODUCT, 0, 1, weights + 25570},
	   {10,0,0,INNER_PRODUCT, 0, 0, weights + 426070}
    };

    // Initialize network
    layer * last = nullptr;
    for (int i = 0; i < num_layers; ++i) {
        init_layer(last, &net_spec[i]);
        last = &net_spec[i];
    }

    // Forward all layers
    for (int i = 0; i < num_layers - 1; ++i) {
	   forward(&net_spec[i], &net_spec[i+1]);
    }

    // Print netspec
    //print(net_spec, 7);

    // ArgMax of final activation
    int idx = num_layers - 1;
    float max = net_spec[idx].activations[0];
    int argmax = 0;
    for (int i = 0; i < net_spec[idx].channels; ++i) {
        if (net_spec[idx].activations[i] > max) {
            argmax = i;
            max = net_spec[idx].activations[i];
        }
    }
    std::cout << "Classification: " << argmax << std::endl;

    // Free network
    for (int i = 0; i < num_layers; ++i) {
	   free_layer(&net_spec[i]);
    }
    delete[] weights;

    return 0;
}
