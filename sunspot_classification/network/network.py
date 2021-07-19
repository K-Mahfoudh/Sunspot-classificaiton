from torch import nn, optim
import torch.nn.functional as F
import torch
import numpy as np
import torchvision.models as models
import time
import sys

# This variable is used to solve unbalance in data, we give weights to each class and add it to our criterion



# Defining Neural network class
class Network(nn.Module):
    def __init__(self, lr, model_save_path):
        super().__init__()

        # Loading a pretrained model (If it's lunched for the first time, it will download model parameters first)
        self.model = models.resnet152(pretrained=True)

        # Train only classification layer, other layers won't be trained (This is used to disable Autograd
        for param in self.model.parameters():
            param.requires_grad = False

        # Learning rate
        self.lr = lr

        # Checking if cuda exists in device in order to train in GPU, otherwise in CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # These 2 lines are used in order to balance data
        #self.CLASS_WEIGHTS = torch.FloatTensor([0.49, 0.31, 1]).to(self.device)
        #self.criterion = nn.NLLLoss(weight=self.CLASS_WEIGHTS)  # NLLLoss Criterion to calculate our loss

        # NLLLoss Criterion to calculate our loss
        self.criterion = nn.NLLLoss()

        # Optimizer (SGD or Adam)
        self.optimizer = None

        # Minimum validation loss for saving model each time we get a lower validation loss
        self.min_valid_loss = np.inf

        # Path for saving model
        self.model_save_path = model_save_path

        # Defining classifier (fully connected layers that we're going to train)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 3),
            nn.LogSoftmax(dim=1)
        )

        # Changing the fully connected layer of resnet with the one that we created
        self.model.fc = self.fc

    # Displaying model layers
    def get_model_details(self):
        print(self.model)

    # Defining forward propagation method
    def forward(self, data):
        x = self.model(data)
        return x

    # Training neural network method
    def train_network(self, trainset, validset, epochs):
        # Setting the optimizer
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=self.lr)

        # Global loss and accuracy list for Visualiaztion
        train_loss_list = []
        train_accuracy_list = []
        valid_loss_list = []
        valid_accuracy_list = []

        # Sending model to GPU if CUDA is available
        self.to(self.device)

        # Training the network
        for epoch in range(epochs):
            # Measuring training time
            epoch_start_time = time.time()

            # Init epoch loss and accuracy for both validation and training
            train_loss = 0
            train_accuracy = 0
            valid_loss = 0
            valid_accuracy = 0

            # Enable training mode if not enabled (This will enable normalization and dropout)
            if not self.train():
                self.train()

            # Looping through each batch of our data loader
            for index, (images, labels, paths) in enumerate(trainset):
                # Sending data to GPU if CUDA is available
                images, labels = images.to(self.device), labels.to(self.device)

                # Resetting gradient, otherwise it will accumulate over each batch and we get false results
                self.optimizer.zero_grad()

                # Performing forward pass
                logits = self.forward(images)

                # Calculating loss
                loss = self.criterion(logits, labels)
                train_loss += loss

                # Backward propagation
                loss.backward()

                # Updating weights and biases
                self.optimizer.step()

                # Calculating accuracy
                preds = F.softmax(logits, dim=1)  # Calculating prediction of each class using softmax
                _, top_class = preds.topk(1, dim=1)  # Getting top predictions and classes with the highest prediction
                compare = top_class == labels.view(*top_class.shape)  # Comparing predictions to real values (if prediction is correct or not)
                accuracy = torch.mean(compare.type(torch.FloatTensor))  # Calculating accuracy of Our model
                train_accuracy += accuracy

                # Printing train loss (This part is only for printing values on console)
                sys.stdout.write(
                    'Batch :{}/{} ---- Train loss: {:.3f}\r'.format(index, len(trainset), loss))
                sys.stdout.flush()

            # Evaluating the model on validation set
            # Switching to evaluation mode to disable Normalization and dropout
            if not self.eval():
                self.eval()

            # Using no_grad in order to disable autograd (Reducing memory usage and more speed in calculations)
            with torch.no_grad():
                for index, (images, labels, paths) in enumerate(validset):
                    # Sending images and labels to GPU if CUDA is Available
                    images, labels = images.to(self.device), labels.to(self.device)

                    # Forward pass
                    logits = self.forward(images)

                    # Calculating loss
                    loss = self.criterion(logits, labels)
                    valid_loss += loss

                    # getting the accuracy
                    preds = F.softmax(logits, dim=1)
                    _, top_class = preds.topk(1, dim=1)
                    compare = top_class == labels.view(*top_class.shape)
                    accuracy = torch.mean(compare.type(torch.FloatTensor))
                    valid_accuracy += accuracy
                    # Printing train loss
                    sys.stdout.write(
                        'Batch :{}/{} ---- Validation loss: {:.3f}\r'.format(index, len(validset),
                                                                             loss))
                    sys.stdout.flush()

            # Getting final epoch accuracy and loss
            train_loss = train_loss / len(trainset)
            train_accuracy = train_accuracy / len(trainset) * 100
            valid_loss = valid_loss / len(validset)
            valid_accuracy = valid_accuracy / len(validset) * 100

            # Appending loss and accuracy to visualization lists
            train_loss_list.append(train_loss)
            train_accuracy_list.append(train_accuracy)
            valid_loss_list.append(valid_loss)
            valid_accuracy_list.append(valid_accuracy)

            # Checking if validation loss decreased in order to save the model
            if valid_loss < self.min_valid_loss:
                print('Validation loss decreased from {:.3f} =======> {:.3f}\r'.format(self.min_valid_loss, valid_loss))
                self.min_valid_loss = valid_loss
                print('Saving model')
                torch.save({
                    'state_dict': self.state_dict(),
                    'min_valid_loss': self.min_valid_loss,
                    'layers': [2048, 512, 128, 3],
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'input_size': 224
                }, self.model_save_path)

            print(
                'Epoch: {}-{:.3f} =====>  Train Accuracy: {:.3f} ------Train Loss: {:.3f} ------ Valid Accuracy: {:.3f} ------ Valid Loss: {:.3f} \r'.format(
                    epoch,
                    time.time() - epoch_start_time,
                    train_accuracy,
                    train_loss,
                    valid_accuracy,
                    valid_loss))

        return train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list

    # Defining prediction method
    def predict(self, testset):
        ground_truth_list = []
        prediction_list = []
        # Setting to evaluation mode to disable Normalization and dropout
        if not self.eval():
            self.eval()
        self.to(self.device)
        # Disable Gradients (no need for gradients in evaluation)
        with torch.no_grad():
            loss, accuracy = 0, 0
            for index, (images, labels, paths) in enumerate(testset):
                # Appending labels in order to calculate F1 score later
                ground_truth_list.append(labels)

                # Sending data to GPU if CUDA is available
                images, labels = images.to(self.device), labels.to(self.device)

                # forward propagation
                logits = self.forward(images)

                # Calculating loss
                loss += self.criterion(logits, labels)

                # Calculating predictions
                preds = F.softmax(logits, dim=1)


                # Getting top classes
                _, top_class = preds.topk(1, dim=1)

                # appending predictions to a list in order to use it with F1 score later
                prediction_list.append(top_class)

                # Comparing predictions to labels
                compare = top_class == labels.view(*top_class.shape)

                # Calculating accuracy
                acc = torch.mean(compare.type(torch.FloatTensor))
                accuracy += acc
                print('Batch accuracy is: {}'.format(acc))
            accuracy = accuracy / len(testset) * 100
            loss = loss / len(testset)
        return loss, accuracy, ground_truth_list, prediction_list

    def load_model(self, model_path):
        print('Loading model from path: {}'.format(model_path))
        model_dict = torch.load(model_path)
        self.min_valid_loss = model_dict['min_valid_loss']
        self.load_state_dict(model_dict['state_dict'])
        print(self.min_valid_loss)

