"""

    Training of the (t)EDRAM Neural Network (see Ablavatski, Lu, & Cai, 2017)

        * main
        * list arguments
        * command line interface

"""
import os
import time
from argparse import ArgumentParser

from train import train


#####################################
###  Default Training Parameters  ###
#####################################

_learning_rate = 0.0001
_batch_size = 192
_model_id = 2
_n_steps = 6
_n_epochs = 25


def main(list_params, gpu_id, dataset_id, model_id, load_path, save_path,
         batch_size, learning_rate, n_epochs, augment_input, rotation, n_steps,
         glimpse_size, coarse_size, conv_sizes, n_filters, fc_dim, enc_dim, dec_dim,
         n_classes, output_mode, use_init_matrix, output_emotion_dims, headless,
         emission_bias, clip_value, unique_emission, unique_glimpse, init_abv,
         scale_inputs, normalize_inputs, use_batch_norm, dropout,
         use_weighted_loss, localisation_cost_factor, zoom_factor, reps):

    for i in range(reps):

        if reps == 1:
            save_path_rep = save_path
        else:
            save_path_rep = save_path + '/' + save_path + '__' + str(i+1)
            time.sleep(10)

        train(list_params, gpu_id, dataset_id, model_id, load_path, save_path_rep,
             batch_size, learning_rate, n_epochs, augment_input, rotation, n_steps,
             glimpse_size, coarse_size, conv_sizes, n_filters, fc_dim, enc_dim, dec_dim,
             n_classes, output_mode, use_init_matrix, output_emotion_dims, headless,
             emission_bias, clip_value, unique_emission, unique_glimpse, init_abv,
             scale_inputs, normalize_inputs, use_batch_norm, dropout,
             use_weighted_loss, localisation_cost_factor, zoom_factor)

    exit()


def list_args(**args):
    """
        Save (and display) arguments of a modell training call
    """
    lines = []
    print_flag = True
    # generate lines for output
    for i, arg in enumerate(args.items()):
        if i <= 1:
            if arg[1]=='none' and i==0:
                print_flag = False
            elif i == 0:
                lines.append("\n[Info] Training Parameters\n")
        else:
            if i==5:
                save_path = './output/'+arg[1]+'/'
            lines.append(' '*(25-len(arg[0]))+str(arg[0])+" = "+str(arg[1]))
    # write lines to disk and std.out
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        if input("\n[Warning] "+save_path+" already exists. Continue anyway? ")!='yes':
            exit()
    out_file = open(save_path+'training_parameters.txt', 'w')
    for i, line in enumerate(lines):
        if i>0:
            out_file.write(line+"\n")
        if print_flag and i!=1:
            print(line)
    out_file.close()


if __name__ == "__main__":

    # argument list
    parser = ArgumentParser(description="Training of the EDRAM network")
    parser.add_argument("--l", "--list_params", type=str, nargs='?', default='all', const='none',
                        dest='list_params', help="Show the parameter list")

    # high-level options
    parser.add_argument("--gpu", type=int, nargs='?', default=-1, const=1,
                        dest='gpu_id', help="Specifies the GPU.")
    parser.add_argument("--data", type=int, default=1,
                        dest='dataset_id', help="ID of the test data set or path to the dataset.")
    parser.add_argument("--model", type=int, default=_model_id,
                        dest='model_id', help="Selects model type.")
    parser.add_argument("--load_path", type=str, default='.',
                        dest='load_path', help="Path for loading the model weights.")
    parser.add_argument("--path", "--save_path", type=str, default='default',
                        dest='save_path', help="Path for saving the output.")
    parser.add_argument("--reps", type=int, default=1,
                        dest='reps', help="How many repetitions of the training.")

    # training parameters
    parser.add_argument("--bs", "--batch_size", type=int, default=_batch_size,
                        dest="batch_size", help="Size of each mini-batch")
    parser.add_argument("--lr", "--rate", '--learning_rate', type=float, default=_learning_rate,
                        dest='learning_rate', help="Learning rate of the model.")
    parser.add_argument("--epochs", "--n_epochs", type=int, default=_n_epochs,
                        dest='n_epochs', help="Number of training epochs.")
    parser.add_argument("--augment", "--augment_input", type=int, default=1,
                        dest="augment_input", help="Whether to use data augmentation.")
    parser.add_argument("--rotation", type=float, default=1.0,
                        dest='rotation', help="Factor for data augmentation.")

    # model structure
    parser.add_argument("--steps", type=int, default=_n_steps,
                        dest="n_steps", help="Step size for digit recognition.")
    parser.add_argument("--glimpse", "--glimpse_size", "-a", type=int, default=26,
                        dest='glimpse_size', help="Window size of attention mechanism.")
    parser.add_argument("--coarse_size", type=int, default=12,
                        dest='coarse_size', help="Size of the rescaled input image for initialization of the network.")
    parser.add_argument("--conv_sizes", type=int, nargs='+', default=[3, 5],
                        dest="conv_sizes", help="List of sizes of convolution filters.")
    parser.add_argument("--conv_filters", type=int, default=128,
                        dest="n_filters", help="Number of filters in convolution.")
    parser.add_argument("--fc_dim", type=int, default=1024,
                        dest="fc_dim", help="Fully connected dimension.")
    parser.add_argument("--enc_dim", type=int, default=512,
                        dest="enc_dim", help="Encoder RNN state dimension.")
    parser.add_argument("--dec_dim", type=int, default=512,
                        dest="dec_dim", help="Decoder  RNN state dimension.")
    parser.add_argument("--classes", type=int, default=10,
                        dest="n_classes", help="Number of classes for recognition.")
    parser.add_argument("--em_bias", "--emission_bias", type=float, default=1.0,
                        dest="emission_bias", help="Presets the zoom bias of the emission network.")
    parser.add_argument("--zoom", "--zoom_factor", type=float, default=1,
                        dest='zoom_factor', help="Target Zoom Factor.")
    parser.add_argument("--clip", "--clip_value", type=float, default=1.0,
                        dest="clip_value", help="Clips Zoom Value in Spatial Transformer.")
    parser.add_argument("--unique_em", "--unique_emission", type=int, default=0,
                        dest="unique_emission", help="Inserts unique emission layer")
    parser.add_argument("--unique_glimpse", type=int, default=0,
                        dest="unique_glimpse", help="Inserts unique first glimpse layer")
    parser.add_argument("--abv", "--init_abv", type=int, default=0,
                        dest="init_abv", help="Use initialisations as in Ablavtsky et al. (2017)")

    # output options
    parser.add_argument("--mode", "--output_mode", type=int, default=1,
                          dest="output_mode", help="Output last step or all steps.")
    parser.add_argument("--use_init", "--use_init_matrix", type=int, default=1,
                          dest="use_init_matrix", help="Whether to use the init matrix as output.")
    parser.add_argument("--dims", "--output_dims", "--output_emotion_dims", type=int, default=0,
                          dest="output_emotion_dims", help="Whether to output valence and arousal.")
    parser.add_argument("--headless", type=int, default=0,
                          dest="headless", help="Whether to use a dense classifier on all timesteps in parallel.")

    # normalisation of inputs and model layers
    parser.add_argument("--scale", "--scale_inputs", type=float, default=255.0,
                        dest="scale_inputs", help="Scaling Factor for Input Images.")
    parser.add_argument("--normalize", "--normalize_inputs", type=int, default=0,
                        dest="normalize_inputs", help="Whether to normalize the input images.")
    parser.add_argument("--bn", "--use_batch_norm", type=int, default=1,
                        dest="use_batch_norm", help="Whether to use batch normalization.")
    parser.add_argument("--do", "--dropout", type=float, default=0,
                        dest="dropout", help="Whether to use dropout (dropout precentage).")

    # loss options
    parser.add_argument("--loss", "--weighted_loss", "--use_weighted_loss", type=int, default=1,
                        dest="use_weighted_loss", help="Whether to use class weights for emotions.")
    parser.add_argument("--cost", "--localisation_cost_factor", type=float, default=1.0,
                        dest='localisation_cost_factor', help="Scales the location cost.")

    args = parser.parse_args()


    list_args(**vars(args))
    main(**vars(args))

