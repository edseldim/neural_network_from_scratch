import pandas as pd
import numpy as np
import logging
import itertools
import time

from datetime import datetime

np.random.seed(2023)

class Perceptron:
    def __init__(self, neurons, weights, biases, training_data):
        self.neurons = neurons
        self.weights = weights
        self.biases = biases
        self.training_data = training_data
        self.activations = []
        self.z_values = []
        self.z_values.append([])
        self.gradients_coordinates_weights = []
        self.gradients_coordinates_biases = []

    def calculate_forward_pass(self):
        """'trains' the neural network by calculating all the activations"""

        for L_layer, L_layer_neurons in enumerate(self.neurons): # moves across all layers
            if L_layer == 0:
                for input_neuron,input_data_point in enumerate(self.training_data[:-(self.neurons[-1])]):
                    if input_neuron == 0:
                        self.activations.append([input_data_point])
                    else:
                        self.activations[0].append(input_data_point)
            else:
                for neuron_j_k_L in range(L_layer_neurons): # moves across all neurons in layer L
                    logging.debug("current_layer: "+str(L_layer)+" current_neuron: "+str(neuron_j_k_L)+" previous_layer: "+str(L_layer-1))
                    z_array = np.dot(self.weights[L_layer][neuron_j_k_L],np.array(self.activations[L_layer-1])) + self.biases[L_layer][neuron_j_k_L] # finds Z = W*x + b
                    a_j_k_L = self.calculate_activation(z_array) # calculates a_j_k^(L)
                    if neuron_j_k_L == 0:
                        self.activations.append([a_j_k_L]) # stores the activation in the corresponding layer activation values
                        self.z_values.append([z_array])
                    else:
                        self.activations[L_layer].append(a_j_k_L)
                        self.z_values[L_layer].append(z_array)

    def calculate_activation(self, z):
        """Calculates the activation of a single neuron"""

        return np.sum(1/(1+np.exp(-np.sum(z)))) #max([0, np.sum(z)]) #np.sum(1/(1+np.exp(-np.sum(z))))
    
    def calculate_activation_derivative(self, z):
        return self.calculate_activation(z)*(1-self.calculate_activation(z))#(np.exp(-np.sum(z))/((1+np.exp(-np.sum(z)))**2)) #1 if np.sum(z) >= 0 else 0 #(np.exp(-np.sum(z))/(1+np.exp(-np.sum(z)))**2)

    def calculate_cost(self):
        """Calculates the total cost for a given data point"""

        last_layer = len(self.neurons) - 1
        return np.sum(
            (np.array(
                    self.training_data[-(self.neurons[-1]):]
                    )-self.activations[last_layer])**2
            )/2

    def calculate_gradients_weights(self):
        """Calculate partial derivative for all the weights"""
        total_layers = len(self.neurons)
        logging.debug(f"calculating backpropagation for {self.neurons[:-1][::-1]}")
        for L_layer, L_layer_neurons in enumerate(self.neurons[:-1][::-1]): # starting from the second to last layer to the beginner
            #self.gradients_coordinates_weights.append([]) # creates a dimension for all the weights in layer
            for neuron_j_k_L in range(L_layer_neurons): # move across all the neurons in current layer (total - L_layer - 2)
                for neuron_j_k_L_forward in range(self.neurons[(total_layers-L_layer-1)]): # move across all the neurons in the next layer
                    #logging.debug(f"Layer #{(total_layers-2) - L_layer} Layer L-1 neuron: {neuron_j_k_L} Layer L neuron: {neuron_j_k_L_forward}")
                    forward_layer = (total_layers-2) - L_layer + 1
                    current_layer = (total_layers-2) - L_layer
                    if total_layers-L_layer == total_layers: # at the last layer
                        #last_layer_for_parameters_lists = (total_layers-1)-L_layer
                        Z = self.z_values[forward_layer][neuron_j_k_L_forward] # L+1 Z values
                        dCdw = self.activations[current_layer][neuron_j_k_L]*self.calculate_activation_derivative(Z) # L a values total_layers-L_layer-2
                        dCdw *= (self.activations[forward_layer][neuron_j_k_L_forward] - self.training_data[neuron_j_k_L_forward])
                        logging.debug(f"dC/dw_{neuron_j_k_L_forward}_{neuron_j_k_L} = {dCdw}")
                        #if neuron_j_k_L_forward == 0:
                        self.gradients_coordinates_weights.append(dCdw)
                        #else:
                            #self.gradients_coordinates_weights.append(dCdw)
                    else:
                        logging.debug(f"forward layer: {forward_layer} forward neuron: {neuron_j_k_L_forward} z-values: {self.z_values[forward_layer]}")
                        Z = self.z_values[forward_layer][neuron_j_k_L_forward]
                        dCdw = self.activations[current_layer][neuron_j_k_L]*self.calculate_activation_derivative(Z)
                        dCdw *= self.weight_traverse(forward_layer,neuron_j_k_L_forward)
                        logging.debug(f"dC/dw_{neuron_j_k_L_forward}_{neuron_j_k_L}_{forward_layer} = {dCdw}")
                        self.gradients_coordinates_weights.append(dCdw)

    def calculate_gradients_biases(self):
        """Calculate partial derivative for all the weights"""
        total_layers = len(self.neurons)
        neurons_order = self.neurons[::-1][:-1]
        logging.debug(f"calculating biases backpropagation for {neurons_order}")
        for L_layer, L_layer_neurons in enumerate(neurons_order): # starting from the second to last layer to the beginner
            for neuron_j_k_L in range(L_layer_neurons): # move across all the neurons in current layer (total - L_layer - 2)
                    
                    if total_layers-L_layer == total_layers: # at the last layer
                        logging.debug(f"Layer #{(total_layers-1) - L_layer} Layer L neuron: {neuron_j_k_L}")
                        current_layer = (total_layers-1) - L_layer
                        #last_layer_for_parameters_lists = (total_layers-1)-L_layer
                        Z = self.z_values[current_layer][neuron_j_k_L] # L+1 Z values
                        dCdb = self.calculate_activation_derivative(Z) # L a values
                        dCdb *= (self.activations[current_layer][neuron_j_k_L] - self.training_data[neuron_j_k_L])
                        logging.debug(f"dC/db_{neuron_j_k_L}_{neuron_j_k_L} = {dCdb}")
                        self.gradients_coordinates_biases.append(dCdb)
                    else:
                        current_layer = (total_layers-1) - L_layer
                        dCdb = 0
                        dCdb_aux = 0
                        # for i,forward_neuron in enumerate([row[neuron_j_k_L] for row in self.weights[current_layer+1]]):
                        #     logging.debug(f"Layer #{(total_layers-1) - L_layer} Layer L neuron: {neuron_j_k_L} forward_neuron L+1: {i}")
                        #     Z = self.z_values[current_layer+1][i]
                        #     dCdb_aux = self.calculate_activation_derivative(Z)
                        #     dCdb_aux *= self.weight_traverse(current_layer+1,i)
                        #     dCdb += dCdb_aux
                        Z = self.z_values[current_layer][neuron_j_k_L]
                        dCdb_aux = self.calculate_activation_derivative(Z)
                        dCdb_aux *= self.weight_traverse(current_layer,neuron_j_k_L)
                        dCdb = dCdb_aux
                        # Z = self.z_values[current_layer][neuron_j_k_L]
                        # dCdb_aux = self.calculate_activation_derivative(Z)
                        # dCdb_aux *= self.weight_traverse(current_layer+1,neuron_j_k_L)
                        self.gradients_coordinates_biases.append(dCdb)



    def weight_traverse(self, current_layer, current_neuron):
        forward_layer = current_layer + 1
        #total_per_row = 0

        if (forward_layer) >= (len(self.neurons)-1):
            current_layer = len(self.neurons) - 1
            forward_weights = [row[current_neuron] for row in self.weights[current_layer]]
            total_per_row = 0
            logging.debug(f"shifted to ==>({current_layer})")
            for j,(activations_j, weight_j) in enumerate(zip(self.activations[current_layer], forward_weights)):
            #logging.debug(f"current_neuron ==> {i}_{current_neuron}({current_layer})")
            #print(len(self.activations[current_layer]),current_neuron)
            #print(self.z_values[-1])
                
                total_per_row += weight_j*self.calculate_activation_derivative(self.z_values[-1][j])*(activations_j - self.training_data[j])
            return total_per_row 

        logging.debug(f"shifted to ==>({current_layer})")
        forward_weights = [row[current_neuron] for row in self.weights[forward_layer]]
        forward_z_values = self.z_values[forward_layer]
        total_per_row = 0
        for i,(weight, z_value) in enumerate(zip(forward_weights, forward_z_values)):
                logging.debug(f"current_neuron ==> {i}_{current_neuron}({current_layer})")
                #print(weight, z_value)
                total_per_row += weight * self.calculate_activation_derivative(z_value) * self.weight_traverse(forward_layer, i)

        return total_per_row


class GradientDescent:
    
    def calculate_total_gradient_C(self, gradients_nn_list):
        """Calculates the total cost gradient vector"""
        return np.sum(np.array(gradients_nn_list), axis=0)/np.array(gradients_nn_list).shape[1]

    def generate_next_w_values(self, gradient_C_vector, weight_vector, learning_rate = 1):
        """Calculates the next w and b for the next step"""
        return weight_vector - ((learning_rate)) * np.array(gradient_C_vector)

def replace_nested_list(nested_list, flat_list, flat_list_index = 0):
    for i in range(len(nested_list)):
        if isinstance(nested_list[i], list):
            flat_list_index = replace_nested_list(nested_list[i], flat_list, flat_list_index)
        else:
            nested_list[i] = flat_list[flat_list_index]
            flat_list_index += 1
    return flat_list_index

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                        #filename=f".\\logs\\logs-{datetime.now().strftime('%d-%m-%Y')}.log",
                        #filemode="w",
                        )
    
    #neurons = np.array([2,1,1])
    number_of_weights = 0
    weights = list()
    weights.append([])
    biases = list()
    biases.append([])
    gd = GradientDescent()
    # df = pd.read_csv("./data/mnist/train.csv")
    # training_data = [row[::-1].tolist() for row in df.values]
    # training_data_copy = []
    # for row in training_data[:]:
    #     target = row[-1]
    #     row_copy = row.copy()
    #     del row_copy[-1]
    #     for i in range(10):
    #         if i == target:
    #             row_copy.append(1)
    #         else:
    #             row_copy.append(0)
    #     training_data_copy.append(row_copy)
    # training_data = training_data_copy.copy()
    # neurons = np.array([df.shape[1] - 1,8,8,10])
    neurons = np.array([2,1,2,2,1])
    training_data = [
        [1,0,1],
        [1,1,1],
        [0,1,1],
        [0,0,0],
    ]
    for i,number in enumerate(neurons):
        if i != len(neurons)-1:
            weights.append(
                np.ones(shape=(neurons[i+1], number)).tolist()
                #np.random.choice(np.linspace(5,10),size=(neurons[i+1], number)).tolist()
            )
        if i != 0:
            #biases.append(np.random.choice(np.linspace(0,5),size=neurons[i]).tolist())
            biases.append((np.zeros(shape=neurons[i])).tolist())

    data_shift = 4
    new_total_cost = 10000000000000000
    i = 0
    while i < 5:
        i += 1
        data_shift = 4
        while data_shift <= len(training_data):
            if data_shift >= len(training_data):
                data_shift = 0
            total_batch_weights = []
            total_batch_biases = []
            total_cost_batch = 0
            logging.info(f"===================================STARTED EPOCH {data_shift/4:.0f} AT {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}===================================")
            for i,data_point in enumerate(training_data[data_shift:data_shift+4]):
                logging.debug(f"=============================(Neural Network #{i+1} STARTED TRAINING)==============================")
                nn = Perceptron(neurons, weights, biases, data_point)
                nn.calculate_forward_pass()
                nn.calculate_gradients_weights()
                nn.calculate_gradients_biases()
                logging.info(f"Neural Network #{i+1}: Output: {nn.activations[-1]} Expected: {data_point[-neurons[-1]:]} \nNN Cost: {nn.calculate_cost()}")
                logging.debug(f"input: {nn.activations[0]}")
                logging.debug(f"z-values: "+str(nn.z_values))
                logging.debug(f"gradient-values-weights: "+str(nn.gradients_coordinates_weights))
                logging.debug(f"gradient-values-biases: "+str(nn.gradients_coordinates_biases))
                logging.debug(f"=============================(Neural Network #{i+1} FINISHED TRAINING)=============================")
                total_batch_weights.append(nn.gradients_coordinates_weights)
                total_batch_biases.append(nn.gradients_coordinates_biases)
                total_cost_batch += nn.calculate_cost()
            
            total_cost_batch = total_cost_batch/len(total_batch_weights)
            logging.info(f"=============> EPOCH {data_shift/4:.0f} COST AVERAGE: {total_cost_batch}")
            if new_total_cost > total_cost_batch:
                new_total_cost = total_cost_batch
                logging.info(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!New lowest cost in batch: {new_total_cost}")
                if new_total_cost < 0.005:
                    import sys
                    sys.exit()
            
            logging.debug(f"Old weights ==> {weights}")
            total_gradient_vector_weight = gd.calculate_total_gradient_C(total_batch_weights)
            total_gradient_vector_weight = total_gradient_vector_weight[::-1]
            logging.debug(f"Gradient weight vector ==> {total_gradient_vector_weight}")
            flattened_out_weight = list(itertools.chain.from_iterable(itertools.chain.from_iterable(weights)))
            new_weights = gd.generate_next_w_values(total_gradient_vector_weight,flattened_out_weight,1).tolist()
            replace_nested_list(weights, new_weights)

            logging.debug(f"Old biases ==> {biases}")
            total_gradient_vector_biases = gd.calculate_total_gradient_C(total_batch_biases)
            total_gradient_vector_biases = total_gradient_vector_biases[::-1]
            logging.debug(f"Gradient bias vector ==> {total_gradient_vector_biases}")
            flattened_out_biases = list(itertools.chain.from_iterable(biases))
            new_biases = gd.generate_next_w_values(total_gradient_vector_biases,flattened_out_biases,1).tolist()
            replace_nested_list(biases, new_biases)

            logging.debug(f"New weights ==> {weights}")
            logging.debug(f"New biases ==> {biases}")
            logging.debug(f"Neurons ==> {neurons}")
            logging.info(f"===================================FINISHED EPOCH {data_shift/4:.0f} AT {datetime.now().strftime('%d-%m-%Y %H:%M:%S')} ===================================")
            data_shift += 4
        
    logging.info(f"============================= Best cost result: {new_total_cost} =========================================")
    logging.info(f"=============================(FINISHED TRAINING)==========================================================")
        

