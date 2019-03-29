#-*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg


misc_arg = add_argument_group('misc')
misc_arg.add_argument('--split', type=bool, default = True)
misc_arg.add_argument('--input_size', type=int, default = 256, 
                      help='multiplies of 256 by the structure of the model') 
misc_arg.add_argument('--use_network', type=bool, default = False)

data_arg = add_argument_group('data')
data_arg.add_argument('--downloading', type=bool, default = False)

graph_arg = add_argument_group('graph')
graph_arg.add_argument('--filter_length', type=int, default = 32)
graph_arg.add_argument('--kernel_size', type=int, default = 16)
graph_arg.add_argument('--drop_rate', type=float, default = 0.2)

train_arg = add_argument_group('train')
train_arg.add_argument('--feature', type=str, default = "MLII",
                       help='one of MLII, V1, V2, V4, V5. Favorably MLII or V1')
train_arg.add_argument('--epochs', type=int, default = 80)
train_arg.add_argument('--batch', type=int, default = 256)
train_arg.add_argument('--patience', type=int, default = 10)
train_arg.add_argument('--min_lr', type=float, default = 0.00005)
train_arg.add_argument('--checkpoint_path', type=str, default = None)
train_arg.add_argument('--resume_epoch', type=int)
train_arg.add_argument('--ensemble', type=bool, default = False)
train_arg.add_argument('--trained_model', type=str, default = None, 
                       help='dir and filename of the trained model for usage.')

predict_arg = add_argument_group('predict')
predict_arg.add_argument('--num', type=int, default = None)
predict_arg.add_argument('--upload', type=bool, default = False)
predict_arg.add_argument('--sample_rate', type=int, default = None)
predict_arg.add_argument('--cinc_download', type=bool, default = False)




def get_config():
    config, unparsed = parser.parse_known_args()

    return config
