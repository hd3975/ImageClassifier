# Import Required Libraries Here

import numpy as np
import torch
import json
import argparse
import torch.nn.functional as nnF
from torchvision import transforms, models
from model_utils import build_network, load_checkpoint
from PIL import Image

# Command Line Argument Parser for predict.py
def parse_predict_cmd():
    parser = argparse.ArgumentParser('Command line argument parser for predict.py')
    parser.add_argument('path_to_img',
                    action = 'store',
                    help = 'Provide the full path to the image file you want to predict using trained model')
    parser.add_argument('checkpoint',
                    action = 'store',
                    help = 'Provide the full path to the trained model checkpoint file')
    parser.add_argument('--top_k',
                    action = 'store',
                    default = 5,
                    type = int,
                    dest = 'top_k',
                    help = 'Provide the number of probabilities [Default = 5] to include in the output')
    parser.add_argument('--category_names',
                    action = 'store',
                    dest = 'cat_names',
                    default = 'cat_to_name.json',
                    help = 'Provide the file name with full path [Default = ./cat_to_name.json] to map the classes to real names')
    parser.add_argument('--gpu',
                    action = 'store_true',
                    dest = 'gpu',
                    help = 'Provide the platform [Default = cpu] where to run the model')

    return(parser.parse_args())

# FuUnction to process a PIL image to scale, crop and normalize to be used by the pytorch model
# Input: Image
# Ouput: transformed tensor of the image
def process_image(image_file):
    img = Image.open(image_file)
    img_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # Process a PIL image for use in a PyTorch model
    return(img_transform(img))    

# Function to predict the possible classes for the image using trained model and return associated probabilities
# Input: Image File, Trained Model, Mapping List of id to real names, top_k
# Output: Probabilities and real names of the classes
def predict(image_file, model, cat_to_name, idx_to_class, device, top_k=5):
    model.to(device)
    # Make sure model is in evaluation mode
    model.eval()
    
    # Get the transformed image from image file
    image = process_image(image_file)
    
    # Convert 2D Image torch tensor to 1D vector with inplace transform
    image = image.unsqueeze_(0)
    # convert the image torch tensor to float data type 
    image = image.float()
    
    # Run the model on image to get the prediction
    with torch.no_grad():
        output = model.forward(image.cuda())
    
    # Get the prediction probabilities
    pred_probs = nnF.softmax(output.data, dim = 1)
    
    # Get the probabilities for the topk prediction
    probs, indices = pred_probs.topk(top_k)
    
    # Convert to probabilities and indices to Numpy array
    probs = probs.cpu().numpy()[0]
    indices  = indices.cpu().numpy()[0]
    
    # Convert indices to class ids to map further to class names using cat_to_name
    #idx_to_class = {v:k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]
    classes_names = [cat_to_name[idx_to_class[x]] for x in indices]
    
    # Return the probabilites of prediction along with class names
    return probs, classes_names


def main():
    
    # Get the user provided parameters from the command line
    predict_args = parse_predict_cmd()
    
    print(predict_args)
    # Select device to run the model
    device = 'cpu'
    if predict_args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create a dictionary "cat_to_name" which contains category_id and category_nameof flowers
    # by reading the data from json format file cat_to_name.json
    with open(predict_args.cat_names, 'r') as f:
        cat_to_name = json.load(f)

    # Loade the trained model from the checkpoint file    
    trained_model = load_checkpoint(predict_args.checkpoint)
    idx_to_class = {v:k for k, v in trained_model.class_to_idx.items()}
    print(trained_model)
    
    # Use trained model to predict the class of the image
    probs, flower_classes = predict(predict_args.path_to_img, trained_model, cat_to_name, idx_to_class, device, predict_args.top_k)
    
    
    print("The trained model predicts the provided image to be " + flower_classes[0] + " with a probability of - " + str(probs[0] * 100))
    print("Here is the top " + str(predict_args.top_k) + " probabilities of the provided image")
    for i in range(predict_args.top_k):
        print("#" + str(i+1) + " for " + flower_classes[i] + " = {:.4f}....".format(probs[i] * 100))
        
#####################################################################################################
# Code to run the Main Program
#####################################################################################################

if __name__ == '__main__':
    main()        
    
    