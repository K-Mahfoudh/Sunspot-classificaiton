# Sunspots classification

This project is a computer vision task that aims to perform sunspot groups classification using deep learning
algorithms, 2 types of images are used to train the models: Magnetogram and continuum images. Each type is
split into 3 classes: Alpha, Beta and BetaX.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required packages

```bash
pip3 install -r requirements.txt
```
If you're not placed in the root folder, use full/relative path to requirements.txt instead.

**Important**: Make sure to install the required packages, otherwise, you may 
have some package related errors (especially CUDA issues) while running the program.

## Data
The data used to train the model was unfortunately deleted by its original author, 
so I can't provide it. However, if you have your own data, and you want to test the model
on it, you need to have the following structure:

```
-dataset
 |
 |--Continuum
	|
	|-----Train
	|	|
	|	|
	|	|---- Alpha
	|	|
	|	|---- Beta
	|	|
	|	|---- Betax
	|
	|
	|-----test
	|	|
	|	|
	|	|---- Alpha
	|	|
	|	|---- Beta
		|
		|---- Betax
```

## Usage
You can run the program using the following command:
```
python3 classifier.py
```

This command will use the default parameters, if you want to change any parameter
, you will need to pass it as an argument when running the program.
For more information about arguments, type the following command:
```
python3 classifier.py -h
```
## Models
You can download the models from this [link](https://drive.google.com/file/d/10yGkL4EnOj4v61OhUV1AAz0SXoURK6CA/view?fbclid=IwAR1xt1VnOlT8XEfy_Mxb_z_9xwrrbgjbI_K3Pn-HlMf9DrcT48YgljYiP-w)
. You will find 2 models, ``resnet2.pth`` for magnetogram images, and ``resnetContinuum.pth`` for solar continuum images.

## Important
1. By default, if the program finds an existent model-name that is passed as argument, it will load it and use it as checkpoint to continue the training based on it.
So if you want to train your model from scratch, just enter a nonexistent model name or an empty string.

2. Note also that if the model saving path specified in arguments exists already, and you train your model, it will override it. So make sure to always save your model with a new name
or keep copies of your it.

## Useful link
For more information about sunpot magnetic classificaiton, check this [link](https://www.spaceweatherlive.com/en/help/the-magnetic-classification-of-sunspots.html)
## Contributing
All are welcome to contribute, but before making any changes, please make sure
to check the project's style. It is best if you follow the "fork-and-pull" Git workflow.

1. Fork the repo on GitHub
2. Clone the project to your own machine
3. Commit changes to your own branch
4. Push your work back up to your fork
5. Submit a Pull request.

**Important**: Do not forget to keep your cloned project up to date, and to open
a new issue in case of any major changes.

## License
[MIT](https://github.com/K-Mahfoudh/Solar-storm-classification/blob/main/LICENSE.md)
