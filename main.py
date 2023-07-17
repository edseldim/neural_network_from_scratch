import pandas as pd
import numpy as np
import logging
import itertools
import random
import math

from datetime import datetime
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
                for neuron_j_L in range(L_layer_neurons): # moves across all neurons in layer L
                    logging.debug("current_layer: "+str(L_layer)+" current_neuron: "+str(neuron_j_L)+" previous_layer: "+str(L_layer-1))
                    z_array = np.dot(self.weights[L_layer][neuron_j_L],np.array(self.activations[L_layer-1])) + self.biases[L_layer][neuron_j_L] # finds Z = W*x + b
                    a_j_L = self.calculate_activation(z_array) # calculates a_j_k^(L)
                    if neuron_j_L == 0:
                        self.activations.append([a_j_L]) # stores the activation in the corresponding layer activation values
                        self.z_values.append([z_array])
                    else:
                        self.activations[L_layer].append(a_j_L)
                        self.z_values[L_layer].append(z_array)

    def calculate_activation(self, z):
        """Calculates the activation of a single neuron"""

        # return max([0, np.sum(z)]) # RELu
        return 1/(1+np.exp(-np.sum(z))) # Sigmoid
    
    def calculate_activation_derivative(self, z):

        # return 1 if np.sum(z) >= 0 else 0 # Relu Derivative
        return self.calculate_activation(z)*(1-self.calculate_activation(z)) # Sigmoid derivative

    def calculate_cost(self):
        """Calculates the total cost for a given data point"""

        last_layer = len(self.neurons) - 1
        return np.sum(
            (np.array(self.training_data[-(self.neurons[-1]):])-self.activations[last_layer])**2
            )

    def calculate_gradients_weights(self):
        """Calculate partial derivative for all the weights"""
        total_layers = len(self.neurons)
        logging.debug(f"calculating backpropagation for {self.neurons[:-1][::-1]}")
        for L_layer, L_layer_neurons in enumerate(self.neurons[:-1][::-1]): # starting from the second to last layer to the first one
            for neuron_j_k_L in range(L_layer_neurons): # move across all the neurons in current layer (total - L_layer - 2)
                for neuron_j_k_L_forward in range(self.neurons[(total_layers-L_layer-1)]): # move across all the neurons in the next layer
                    forward_layer = (total_layers-2) - L_layer + 1
                    current_layer = (total_layers-2) - L_layer
                    if total_layers-L_layer == total_layers: # at the last layer
                        Z = self.z_values[forward_layer][neuron_j_k_L_forward] # L+1 Z values
                        dCdw = self.activations[current_layer][neuron_j_k_L]*self.calculate_activation_derivative(Z) # L a values total_layers-L_layer-2
                        dCdw *= -2*(self.training_data[neuron_j_k_L_forward]-self.activations[forward_layer][neuron_j_k_L_forward])
                        logging.debug(f"dC/dw_{neuron_j_k_L_forward}_{neuron_j_k_L} = {dCdw}")
                        self.gradients_coordinates_weights.append(dCdw)
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
                        Z = self.z_values[current_layer][neuron_j_k_L] # L+1 Z values
                        dCdb = self.calculate_activation_derivative(Z) # L a values
                        dCdb *= -2*(self.training_data[neuron_j_k_L]-self.activations[current_layer][neuron_j_k_L])
                        logging.debug(f"dC/db_{neuron_j_k_L}_{neuron_j_k_L} = {dCdb}")
                        self.gradients_coordinates_biases.append(dCdb)
                    else:
                        current_layer = (total_layers-1) - L_layer
                        dCdb = 0
                        dCdb_aux = 0
                        Z = self.z_values[current_layer][neuron_j_k_L]
                        dCdb_aux = self.calculate_activation_derivative(Z)
                        dCdb_aux *= self.weight_traverse(current_layer,neuron_j_k_L)
                        dCdb = dCdb_aux
                        self.gradients_coordinates_biases.append(dCdb)



    def weight_traverse(self, current_layer, current_neuron):

        if (current_layer) == (len(self.neurons)-1):
            current_layer = len(self.neurons) - 1
            logging.debug(f"shifted to ==>({current_layer})")
            total_per_row = -2*(self.training_data[-1*current_neuron-1]-self.activations[current_layer][current_neuron])

            return total_per_row 

        logging.debug(f"shifted to ==>({current_layer})")
        forward_weights = [row[current_neuron] for row in self.weights[current_layer+1]]
        forward_z_values = self.z_values[current_layer+1]
        total_per_row = 0
        for i,(weight, z_value) in enumerate(zip(forward_weights, forward_z_values)):
                logging.debug(f"current_neuron ==> {i}_{current_neuron}({current_layer})")
                total_per_row += weight * self.calculate_activation_derivative(z_value) * self.weight_traverse(current_layer+1, i)

        return total_per_row


class GradientDescent:
    
    def calculate_total_gradient_C(self, gradients_nn_list):
        """Calculates the total cost gradient vector"""
        return np.sum(np.array(gradients_nn_list), axis=0)/np.array(gradients_nn_list).shape[0]

    def generate_next_w_values(self, gradient_C_vector, weight_vector, learning_rate = -1):
        """Calculates the next w and b for the next step"""
        if learning_rate > 0:
            return weight_vector - ((learning_rate)) * np.array(gradient_C_vector)
        gradient_vector_value = np.sqrt(np.array(gradient_C_vector).dot(np.array(gradient_C_vector)))
        return weight_vector - (gradient_vector_value) * np.array(gradient_C_vector)

def replace_nested_list(nested_list, flat_list, flat_list_index = 0):
    """This is needed to replace data from nested list to a flat list
        in the same positions"""
    
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
    
    number_of_weights = 0
    weights = list()
    weights.append([])
    biases = list()
    biases.append([])
    gd = GradientDescent()
    #-------------------
    # MNIST Dataset Load
    #-------------------
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
    # neurons = np.array([df.shape[1] - 1,10,10,10])

    #-------------------
    # Iris Dataset Load
    #-------------------

    # features = ["sepal-length","sepal-width","petal-length","petal-width"]
    # target = ["Iris-setosa",  "Iris-versicolor",  "Iris-virginica"]
    # df = pd.read_csv("data\iris\iris.csv",names=["sepal-length","sepal-width","petal-length","petal-width","class"])
    # merged_df = pd.concat([df,pd.get_dummies(df["class"])],axis=1)
    # merged_df.drop("class",axis=1, inplace=True)
    # merged_df = merged_df.sample(len(merged_df))
    # training_data = merged_df.values.tolist()
    # neurons = np.array([len(features),5,5,len(target)])

    #-------------------
    # Dummy Dataset (AND logic gate)
    #-------------------
    neurons = np.array([2,4,1])
    training_data = [
        [1,0,1],
        [1,1,1],
        [0,1,1],
        [0,0,0],
    ]


    #-------------------
    # Random Parameter Initialization
    #-------------------
    for i,number in enumerate(neurons):
        if i != len(neurons)-1:
            weights.append(
                np.ones(shape=(neurons[i+1], number)).tolist() # weights in 1
            )
        if i != 0:
            biases.append((np.zeros(shape=neurons[i])).tolist()) # biases in 0

    #-------------------
    # Training Parameters
    #-------------------
    data_shift = 5
    repeat = True
    max_iterations = 100
    error_to_stop = -np.inf
    learning_rate = -1
    training_data_perc = 0.6

    # do not touch
    data_shift_init = data_shift
    new_total_cost = 10000000000000000
    i = 0
    random.shuffle(training_data)
    training_data = training_data[:math.floor(len(training_data)*training_data_perc)]

    while True:
            i += 1
            total_batch_weights = []
            total_batch_biases = []
            total_cost_batch = 0
            if repeat:
                data_shift = data_shift_init
                random.shuffle(training_data) # after running out of data, shuffle the data and start over at 0
            else:
                if i == max_iterations:
                    break

            logging.info(f"===================================STARTED EPOCH {i} AT {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}===================================")
            for j,data_point in enumerate(training_data[data_shift-data_shift_init:data_shift]):
                logging.debug(f"=============================(Neural Network #{j+1} STARTED TRAINING)==============================")
                nn = Perceptron(neurons, weights, biases, data_point)
                nn.calculate_forward_pass()
                nn.calculate_gradients_weights()
                nn.calculate_gradients_biases()
                logging.info(f"Neural Network #{j+1}: Output: {nn.activations[-1]} Expected: {data_point[-neurons[-1]:]} \nNN Cost: {nn.calculate_cost()}")
                logging.debug(f"input: {nn.activations[0]}")
                logging.debug(f"z-values: "+str(nn.z_values))
                logging.debug(f"gradient-values-weights: "+str(nn.gradients_coordinates_weights))
                logging.debug(f"gradient-values-biases: "+str(nn.gradients_coordinates_biases))
                logging.debug(f"=============================(Neural Network #{j+1} FINISHED TRAINING)=============================")
                total_batch_weights.append(nn.gradients_coordinates_weights)
                total_batch_biases.append(nn.gradients_coordinates_biases)
                total_cost_batch += nn.calculate_cost()

            #-------------------
            # Error Measurement Update
            #-------------------
            
            total_cost_batch = total_cost_batch/len(total_batch_weights)
            logging.info(f"=============> EPOCH {i} COST AVERAGE: {total_cost_batch}")
            if new_total_cost > total_cost_batch:
                new_total_cost = total_cost_batch
                logging.info(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!New lowest cost in batch: {new_total_cost}")

            #-------------------
            # Weight Update
            #-------------------
            
            logging.debug(f"Old weights ==> {weights}")
            total_gradient_vector_weight = gd.calculate_total_gradient_C(total_batch_weights)
            total_gradient_vector_weight = total_gradient_vector_weight[::-1]
            logging.debug(f"Gradient weight vector ==> {total_gradient_vector_weight}")
            flattened_out_weight = list(itertools.chain.from_iterable(itertools.chain.from_iterable(weights)))
            new_weights = gd.generate_next_w_values(total_gradient_vector_weight,flattened_out_weight,learning_rate).tolist()
            replace_nested_list(weights, new_weights)

            #-------------------
            # Biases Update
            #-------------------

            logging.debug(f"Old biases ==> {biases}")
            total_gradient_vector_biases = gd.calculate_total_gradient_C(total_batch_biases)
            total_gradient_vector_biases = total_gradient_vector_biases[::-1]
            logging.debug(f"Gradient bias vector ==> {total_gradient_vector_biases}")
            flattened_out_biases = list(itertools.chain.from_iterable(biases))
            new_biases = gd.generate_next_w_values(total_gradient_vector_biases,flattened_out_biases,learning_rate).tolist()
            replace_nested_list(biases, new_biases)

            #-------------------
            # Gradient Values Check
            #-------------------
            gradient_vector_value_w = np.sqrt(np.array(total_gradient_vector_weight).dot(np.array(total_gradient_vector_weight)))
            logging.info(f"Weight Gradient Slope: {gradient_vector_value_w}")
            gradient_vector_value_b = np.sqrt(np.array(total_gradient_vector_biases).dot(np.array(total_gradient_vector_biases)))
            logging.info(f"Biases Gradient Slope: {gradient_vector_value_b}")
            if gradient_vector_value_b <= error_to_stop and gradient_vector_value_w <= error_to_stop:
                import sys
                sys.exit()

            logging.debug(f"New weights ==> {weights}")
            logging.debug(f"New biases ==> {biases}")
            logging.debug(f"Neurons ==> {neurons}")
            logging.info(f"===================================FINISHED EPOCH {i} AT {datetime.now().strftime('%d-%m-%Y %H:%M:%S')} ===================================")
            data_shift = data_shift_init*(i+1)

        
    logging.info(f"============================= Best cost result: {new_total_cost} =========================================")
    logging.info(f"=============================(FINISHED TRAINING)==========================================================")
        

