import torch
from torch import nn, optim
import torch.nn.functional as nnF
from torchvision import models

# Function to create classifer with user provided hyper parameters
# Input: input features, hidden layers, output features, drop rate
# Output: Classifier

def build_classifier(inp_size, hidden_layers, out_size, p_drop):
    if hidden_layers == None:
        hidden_layers = []
    num_hid_layers = len(hidden_layers)
    classifier = nn.Sequential()
    if num_hid_layers == 0:
        classifier.add_module('fc0', nn.Linear(inp_size, out_size))
    else:
        classifier = nn.Sequential()
        classifier.add_module('fc0', nn.Linear(inp_size, hidden_layers[0]))
        classifier.add_module('relu0', nn.ReLU())
        classifier.add_module('drop0', nn.Dropout(p_drop))
        if num_hid_layers > 1:
            for i in range(num_hid_layers - 1):
                module_name = str(i + 1)
                classifier.add_module('fc' + module_name, nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                classifier.add_module('relu' + module_name, nn.ReLU())
                classifier.add_module('drop' + module_name, nn.Dropout(p_drop))
        module_name = str(num_hid_layers)
        classifier.add_module('fc' + module_name, nn.Linear(hidden_layers[num_hid_layers - 1], out_size))
    classifier.add_module('output', nn.LogSoftmax(dim = 1))
    return classifier

# Function builds a network using user provided model name, input & output features of the model, hidden layers, drop rate and
# learning rate. Assumes the default drop rate of 0.5 and learning rate f 0.001 if nor provided by user
# Input: Model Name, input features, output features, hidden layers, drop rate(optional), learning rate (optional)
# Output: model, criterion and optimizer
def build_network(model_name, input_size, output_size, hidden_layers, drop_rate = 0.5, lr = 0.001):
    if model_name == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained = True)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained = True)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained = True)
    elif model_name == 'densenet161':
        model = models.densenet161(pretrained = True)
    else:
        print("Wrong selection of pytorch model")
        exit(1)
    
    # print(model)# Freeze Parameters of the pretrained model
    for param in model.parameters():
        param.requires_grad = False
    
    # Use your own classifier for the model
    myclassifier = build_classifier(input_size, hidden_layers, output_size, drop_rate)
    model.classifier = myclassifier
    print(model.classifier)

    # Define Criterion
    criterion = nn.NLLLoss()

    # Define Optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    return model, criterion, optimizer

# Define a function for deep learning which will train the network and calculate accuracy on validation data
# Function tracks the loss and accuracy on validation dataset to determine best model weights
# Input: model, criterion, train loader, validation loader, epochs, print every, device (optional)
# Ouput: trained model
def do_deep_learning(model, criterion, optimizer, t_loader, v_loader, epoch = 4, print_every = 16, device = 'cpu'):
    
    # Initialize variable to train the netwrok
    best_accuracy = 0
    len_valid_loader = len(v_loader)
    #model_wts = copy.deepcopy(model.state_dict())

    # Move the model to available device default is cpu
    model.to(device)

    for ep in range(epoch):
        #    print("ep = ", ep, "epoch = ", epoch)
        # Kepp track of running loss for every epoch
        running_loss = 0
    
        # Get image and label from train loader i.e. training dataset
        for idx1, (train_input, train_label) in enumerate(t_loader):
            # Move the training images and labels to GPU if available        
            train_input, train_label = train_input.to(device), train_label.to(device)

            # Initialize Optimizer with zero gradients
            optimizer.zero_grad()

            # Forward pass, loss function and backward pass of the network
            train_output = model.forward(train_input)
            loss = criterion(train_output, train_label)
            loss.backward()
            optimizer.step()
        
            # Calculate running loss for this epoch
            running_loss += loss.item()

            # Check if validation phase needs to run
            if (idx1 + 1) % print_every == 0:
                model.eval()
                valid_loss = 0
                valid_accuracy = 0
                for idx2, (valid_input, valid_label) in enumerate(v_loader):
                    valid_input, valid_label = valid_input.to(device), valid_label.to(device)
                    model.to(device)
                    with torch.no_grad():
                        valid_output = model.forward(valid_input)
                        valid_loss = criterion(valid_output, valid_label)
                        ps = torch.exp(valid_output)
                        equality = (valid_label.data == ps.max(dim=1)[1])
                        valid_accuracy += equality.type_as(torch.FloatTensor()).mean()
                valid_loss = valid_loss / len_valid_loader
                valid_accuracy = valid_accuracy / len_valid_loader
            
                print("Epoch: {}/{}.... ".format(ep+1, epoch), "Training Loss: {:.4f}.... ".format(running_loss/print_every),
                  "Validation Loss: {:.4f}.... ".format(valid_loss), "Validation Accuracy: {:.2f}".format(valid_accuracy * 100))
                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                running_loss = 0
                model.train()
            
    print("Training and Validation Complete")        
    print("Best Accuracy achieved = {:.4f}..... ".format(best_accuracy * 100))

    return model

# Function to validate the trained model on test data set
# Input: Trained Model, Data Loader for the test data set, device
# Ouput: Percentage Correct Prediction by the model
def validation_on_testset(model, dataloader, device = 'cpu'):
    correct = 0.0
    total = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for test_input, test_label in dataloader:
            test_input, test_label = test_input.to(device), test_label.to(device)
            test_output = model(test_input)
            _, predicted = torch.max(test_output.data, 1)
            total += test_label.size(0)
            correct += (predicted == test_label).sum().item()
    model.train()
    return (correct / total)

# Funtion to rebuild the model and load it with trained weights from a file
# Input: checkpoint file of the model
# Output: trained model
def load_checkpoint(chkpt_file):
    checkpoint = torch.load(chkpt_file)
    model_name = checkpoint['model_name']
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    hidden_layers = checkpoint['hidden_layers']
    drop_rate = checkpoint['drop_rate']
    lr = checkpoint['lr']
    model,_,_ = build_network(model_name, input_size, output_size, hidden_layers, drop_rate, lr)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model