from torchvision import datasets, transforms
from customImageLoader import CustomImageFolder
from torch.utils.data import SubsetRandomSampler, DataLoader
import numpy as np


class Data:
    """
    Class for managing and preparing data for training, validation and testing phases.
    """
    def __init__(self, train_path, test_path, batch_size):
        """

        :param train_path: string representing train set path
        :param test_path: string representing test set path
        :param batch_size: batch size to be used in training
        """
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.train_transform = None
        self.test_transform = None

    def get_train_valid(self, validation_size):
        """
        spliting training data into train and validation sets based on a given validation size (as class attribute).

        :param validation_size: float value between 0 and 1 representing size of validation set
        :return: dataloader tuple of training and validation sets
        """
        # Defining transformations
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop((224,224)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        data = CustomImageFolder(self.train_path, transform=self.train_transform)

        # Getting Data size
        data_size = len(data)

        # Getting index list of data
        index_list = list(range(data_size))

        # Shuffling data
        np.random.shuffle(index_list)

        # Creating splitter
        splitter = int(np.floor(data_size * validation_size))

        # Creating Train and validation sublists
        train_index_list, valid_index_list = index_list[splitter:], index_list[:splitter]

        # Creating samples
        train_sampler, valid_sampler = SubsetRandomSampler(train_index_list), SubsetRandomSampler(valid_index_list)

        # getting data loaders
        train_loader = DataLoader(data, batch_size=self.batch_size, sampler=train_sampler)
        valid_loader = DataLoader(data, batch_size=self.batch_size, sampler=valid_sampler)

        return train_loader, valid_loader

    def get_test(self):
        """
        loading test set on dataloader object.

        :return: dataloader object containing test set
        """
        # Defining transformations
        self.test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        # Gettin dataset
        data = CustomImageFolder(self.test_path, transform=self.test_transform)

        # Getting Data Loader
        test_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        return test_loader
