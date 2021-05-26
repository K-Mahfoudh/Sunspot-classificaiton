Before you run your program, read this carefully please:

=============================== Installing packages =====================================================================================================================

A file named requirements.txt is included with the project, run the following command to install packages:


pip install requirements.txt






=============================== Training and testing =====================================================================================================================

##training and test sets should be inside dataset folder, the structure is defined already, 
you just need to copy your classes (Alpha, Beta, Betax) an paste them inside the corresponding folder, it should look like this


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

## Model should be downloader from the google drive link provided, copy them inside models folder



## In order to test your model, you can use the following commands on project's command line:

# testing--------------

python classifier.py TEST  path/to/magnetorgram/trainset path/to/magnetogram/testset 32 100 models/modelName.pth models/modelSaveName.pth 0.0003 0.2
python classifier.py TEST  path/to/continuum/trainset path/to/magnetogram/continuum 32 100 models/modelName.pth models/modelSaveName.pth 0.0003 0.2



# Training-------------

python classifier.py TRAIN  path/to/magnetorgram/trainset path/to/magnetogram/testset 32 100 models/modelName.pth models/modelSaveName.pth 0.0003 0.2
python classifier.py TRAIN  path/to/continuum/trainset path/to/continuum/testset 32 100 models/modelName.pth models/modelSaveName.pth 0.0003 0.2



You will find all the details about the used command line's arguments inside classifier.py, in parser help.






=============================== About training =====================================================================================================================
#By default, if the program finds an existant model name in your arguement,it will load it and use it a checkpoint to continue training based on it
so if you want to train from 0, just enter an invalid model name or an empty string

# Note also that if the model saving path specified in arguments exists already, and you train your model, it will override it, somake sure to always save with a new name
or keep copies of your models.
======================================================================================================================================================================

If you have any questions contact me :)

