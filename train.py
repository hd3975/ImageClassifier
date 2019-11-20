# Import Required Libraries Here

import numpy as np
import torch
import json
import argparse
from torchvision import datasets, transforms, models
from model_utils import build_classifier, build_network, do_deep_learning, validation_on_testset

# Command Line Argument Parser for train.py
def parse_train_cmd():
    parser = argparse.ArgumentParser('Command line argument parser for train.py')
    parser.add_argument('data_dir',
                    action = 'store',
                    default = '.',
                    help = 'Provide the directory [Default = Current Directory] where the Image files for traininare, validation and test are stored')
    parser.add_argument('--save_dir',
                    action = 'store',
                    dest = 'save_dir',
                    default = '.',
                    help = 'Provide the directory [Default = Current Directory] where the checkpoint file for the trained model will be saved')
    parser.add_argument('--arch',
                    action = 'store',
                    default = 'vgg16',
                    dest = 'arch',
                    help = 'Provide the pre-trained model architecture [Default = vgg16] from torchvision to be used')
    parser.add_argument('--learning_rate',
                    action = 'store',
                    default = 0.001,
                    type = float,
                    dest = 'lr',
                    help = 'Provide the learning rate [Default = 0.001] for the model training')
    parser.add_argument('--hidden_units',
                    action = 'append',
                    nargs = '*',
                    type = int,
                    dest = 'hidden_units',
                    help = 'Provide the hidden layers [Default = None] to be used for training the model')
    parser.add_argument('--epochs',
                    action = 'store',
                    default = 4,
                    type = int,
                    dest = 'epochs',
                    help = 'Provide the number of epochs [Default = 4] to run for training the model')
    parser.add_argument('--gpu',
                    action = 'store_true',
                    dest = 'gpu',
                    help = 'Provide the platform [Default = cpu] where to run the model')

    args = parser.parse_args()

    #print(args)

    if args.hidden_units == None:
        hidden_layers = []
    else:
        num_hidden_units = len(args.hidden_units)
        print(args.hidden_units)
        print(num_hidden_units)

        if num_hidden_units == 1:
            hidden_layers = args.hidden_units[0]
        else:
            hidden_layers = []
            for i in range(num_hidden_units):
                hidden_layers.append(args.hidden_units[i][0])
        #print(hidden_layers)

    print(args, hidden_layers)
    return args, hidden_layers

def main():
    # Initialize user defined variables
    # Image Size as required by the pre-trained network
    img_size = 224 # Width of the image in pixels

    # Parameters for reading images
    batchsize = 32 # Batch size for no. of images to be read

    # Following parameters are used for data augmentation during transformation
    img_resize = 256 # Used for RandomResize of the image
    drop_rate = 0.2 # Probability of dropping the image for training the model
    flip_rate = 0.15 # Probability for flipping the image horizontally
    rot_angle = 45 # Rotation Angle of image

    # Get the user provided parameters from the command line
    train_args, hidden_layers = parse_train_cmd()
    
    # Select here the pre-trained pytorch model you want to use
    model_arch = train_args.arch # Possible values are vgg16, vgg19, alexnet, densenet121, densenet161
    if model_arch == 'vgg16':
        input_size = 25088
    elif model_arch == 'vgg19':
        input_size = 25088
    elif model_arch == 'alexnet':
        input_size = 9216
    elif model_arch == 'densenet121':
        input_size = 1024
    elif model_arch == 'densenet161':
        input_size = 2208
    else:
        print("Wrong selection of pytorch model")
        exit(1)    

    # Parameters for training the model
    epoch = train_args.epochs # No of epochs for training the model
    print_every = 16
    lr = train_args.lr # Learning Rate

    
    # Select device to run the model
    device = 'cpu'
    if train_args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #device = 'cpu'
    print(device)
    #print(hidden_layers)

    # Create name of the checkpoint file based on the model to accomodate multiple models
    checkpoint_file = train_args.save_dir + '/' + model_arch + '_chkpt.pth'
    print("Checkpoint File = ", checkpoint_file)
    
    train_dir = train_args.data_dir + '/train'
    valid_dir = train_args.data_dir + '/valid'
    test_dir = train_args.data_dir + '/test'
    
    # Build the image transforms for training and validation data
    train_transform = transforms.Compose([transforms.RandomRotation(rot_angle),
                                                transforms.RandomResizedCrop(img_size),
                                                transforms.RandomHorizontalFlip(flip_rate),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    valid_transform = transforms.Compose([transforms.Resize(img_resize),
                                                transforms.CenterCrop(img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # Load the datasets after applying the transformations on images
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_data = datasets.ImageFolder(test_dir, transform=valid_transform)

    # Print the number of images in each data set
    print("No of images in trainig data set = ", len(train_data))
    print("No of images in validation data set = ", len(valid_data))
    print("No of images in test data set = ", len(test_data))

    # Define the data loaders for loading images for training, validation and testing
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batchsize, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = batchsize, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batchsize)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    # Get the output size for the classifier variables
    output_size = len(cat_to_name)
    
    # Call build_network to build the network ready for training & validation
    print("Selected pre-trained model = ", model_arch)
    print("Input Size for this model = ", input_size)
    print("Hidden Layers selected  = ", hidden_layers)
    
    # Build the network for the model
    model, criterion, optimizer = build_network(model_arch, input_size, output_size, hidden_layers, drop_rate, lr)
    
    print(criterion)
    print(optimizer)
    print(model)
    # Run the network to train the model
    model = do_deep_learning(model, criterion, optimizer, train_loader, valid_loader, epoch, print_every, device)
    
    # Use the Trained model to measure accuracy of prediction on test dataset
    correct_pred = validation_on_testset(model, test_loader, device)

    print("Accuracy of the network on Test Images = {:.2f}....".format(correct_pred * 100))
   
    # Save the model architecture and state dictionary of the trained model into a file
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'model_name': model_arch,
                  'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hidden_layers,
                  'drop_rate': drop_rate,
                  'lr': lr,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, checkpoint_file)

#####################################################################################################
# Code to run the Main Program
#####################################################################################################

if __name__ == '__main__':
    main()