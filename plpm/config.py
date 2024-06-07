import json
import sys
from plpm.constants.BaselineCase import parameters

class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
            self.snapshot_path = self.config['snapshot_path']
            self.loss_path = self.config['loss_path']
            self.parameters = self.set_parameters()
            self.num_layers = self.config['num_layers']
            self.hidden_size = self.config['hidden_size']
            self.num_cases = self.config['num_cases']
            self.seed = self.config['seed']
    
    def set_parameters(self):
        input_params = set(self.config.get('input_params', []))

        # Set the 'is_input' field for each parameter
        if len(input_params) !=0:
            for param in input_params:
                if param in parameters:
                    parameters[param]['is_input'] = True
                else:
                    print(f"Error: Parameter {param} not found in 'parameters'")
                    sys.exit()

        return parameters