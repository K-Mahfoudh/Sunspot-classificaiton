import os
from data import Data
from network import Network
import argparse
from sklearn import metrics


def f1_score(ground_truth_list, prediction_list):
    # flattening lists and converting items to cpu (they are originally in GPU )
    ground_truth_list = [item.to('cpu') for sublist in ground_truth_list for item in sublist]
    prediction_list = [item.to('cpu') for sublist in prediction_list for item in sublist]

    # Getting f1 Score for each class
    class_score = metrics.f1_score(ground_truth_list, prediction_list, average=None)

    # Getting Macro f1 score
    macro_score = metrics.f1_score(ground_truth_list, prediction_list, average='macro')

    return class_score, macro_score


parser = argparse.ArgumentParser(description="Sunspot classification")
parser.add_argument("-m", "--mode", help="TRAIN for training and TEST for testing", type=str, default='TEST')
parser.add_argument("-tp", "--train_path", help="Path of your training set, can be relative or full, default points to continuum dataset", type=str, default='dataset/continuum/train')
parser.add_argument("-ts", "--test_path", help="Path of your testing set, can be relative or full, default points to continuum dataset", type=str, default='dataset/continuum/test')
parser.add_argument("-b", "--batch_size", help="size of batches", type=int, default=32)
parser.add_argument("-e", "--epochs", help="Number of epochs", type=int, default=100)
parser.add_argument("-l", "--load_path", help="path to your model", type=str, default='models/model.pth')
parser.add_argument("-s", "--save_path",help="path for saving model", type=str, default='models/model.pth')
parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=0.0003)
parser.add_argument("-v", "--validation_size", help="validation size between 0 and 1", type=float, default=0.2)


def main(args):

    # Creating an instance of the neural network
    network = Network(args.lr, args.save_path)

    # Creating Data instance
    data = Data(args.train_path, args.test_path, args.batch_size)

    if os.path.exists(args.load_path):
        network.load_model(args.load_path)

    if args.mode == 'TRAIN':
        # Loading training and validation data
        train_data, validation_data = data.get_train_valid(args.validation_size)

        # Loading checkpoint model if exists


        train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list = network.train_network(train_data, validation_data, args.epochs)
        with open('test.txt', 'w+') as test_log:
            test_log.write('Train_lostt_list: {}\nTrain_accuracy_list: {}\nValidation_loss_list: {}\nvalidation_accuracy_list: {}'.format(train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list))
        # Storing training data in txt file

    elif args.mode == 'TEST':
        # Loading test data
        test_data = data.get_test()
        #test_data, validation_data = data.get_train_valid(args.validation_size)
        loss, accuracy, ground_truth_list, prediction_list = network.predict(test_data)
        print("Model's Loss is : {} \nModel's Accuracy is : {}".format(loss, accuracy))
        print('Calculating F1 score')
        class_score, macro_score = f1_score(ground_truth_list, prediction_list)
        print('Classes F1 score is respectively: {}\nAverage F1 score (macro) is {}'.format(class_score, macro_score))

        # Saving data in txt file
        with open('test.txt', 'w+') as test_log:
            test_log.write('Accuracy: {}\nLoss: {}\nClass F1 score: {}\nMacro F1 score: {}'.format(accuracy, loss, class_score, macro_score))

    else:
        print('Wrong mode, write TEST or TRAIN instead')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
