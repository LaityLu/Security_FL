import os
import argparse


from src import model, baseloader, sampler, selector, flalgo
from src.utils import parse_yaml, setup_logger
from src.utils.helper import set_random_seed

# get the config file path
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/test/fmnist.yaml', help='the path of config file')
args = parser.parse_args()

# load the config file
config = parse_yaml(args.config)
log_file_path = os.path.join('./logs', config['Dataset']['name'])
if not os.path.exists(log_file_path):
    os.makedirs(log_file_path)
logger = setup_logger(log_file_path + f'/{config["Recover"]["name"]}.log')

# set random seed
set_random_seed(config['seed'])

# get the dataset
train_dataset, test_dataset = getattr(baseloader, config['Dataset']['name'])(**config['Dataset']['args'])

# get the model
global_model = getattr(model, config['Model']['name'])(**config['Model']['args'])
global_model.to(config['Trainer']['device'])

# get the sampler
sampler_config = config['Sampler']
fl_sampler = getattr(sampler, config['Sampler']['name'])(train_dataset,
                                                         **config['Sampler']['args'],
                                                         num_clients=config['FL']['num_clients'])
client_data_dict = fl_sampler.sample()

# get the global trainer
# global_trainer = getattr(trainer, config['Trainer']['name'])(**config['Trainer']['args'])

# get the selector
selector = getattr(selector, config['Selector']['name'])(**config['Selector']['args'])

# start the federated learning training
server = getattr(flalgo, config['FL']['name'])(global_model, selector, train_dataset,
                                               test_dataset, client_data_dict, config)
server.fl_train()
server.fl_recover()
