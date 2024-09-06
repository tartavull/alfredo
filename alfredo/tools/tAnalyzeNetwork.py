import jax
import jax.numpy as jp

def analyze_neural_params(params):
    """
    This function provides a metric summary of input neural parameter file
    Structure of contents are included in a tuple where index:
        1. RunningStatisticState
        2. FrozenDict(Policy Network Params)
        3. FrozenDict(Value Network Params)
    """

    summary = {}
      
    summary['Running Statistics'] = params[0]
    summary['Policy Network'] = analyze_one_neural_network(params[1])
    summary['Value Network'] = analyze_one_neural_network(params[2])

    return summary
        

def analyze_one_neural_network(sn_params):
    """
    Helper function that unpacks a single network parameters
    Assuming the contents of these parameters are provided as:

    FrozenDict({params: {'hidden_x': {bias: Jax.Array, kernel: Jax.Array}}})

    where x in hidden_x represents order of hidden layer 
    (but these also include input and output layers?) 
    """

    summary = {}

    num_layers = 0
    per_layer_info = {}
    total_parameters = 0
    
    for k, v in sn_params.items():
        
        for layer_name, layer_data in v.items():
            
            num_layers += 1
            param_count = 0
            
            for param_name, param_data in layer_data.items():
                
                if param_name == 'bias':
                    per_layer_info[layer_name] = {'size_of_layer': len(param_data)}
                    param_count += len(param_data)
                    # print(len(param_data))
                
                if param_name  == 'kernel':
                    param_count += len(param_data)*len(param_data[0])
                    # print(len(param_data)*len(param_data[0]))

            total_parameters += param_count

    summary['num_layers'] = num_layers
    summary['per_layer_info'] = per_layer_info
    summary['total_parameters'] = total_parameters

    return summary
