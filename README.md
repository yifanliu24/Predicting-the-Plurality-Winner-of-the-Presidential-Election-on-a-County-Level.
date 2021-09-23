Before running this program, ensure that the pandas, scipy.special, random, sklearn.model_selection, numpy, matplotlib, and str2bool packages are installed in your python interpreter.
If they are not, you may install them using the following commands from your command prompt:
python -m pip install pandas
python -m pip install scipy.special
python -m pip install sklearn.model_selection
python -m pip install random
python -m pip install numpy
python -m pip install matplotlib
python -m pip install str2bool

This program accepts 5 arguments in the following order:
1) boolean <hyper_flag> is a flag that determines if you want to search for the best network architecture and learning rate.
2) int <hidden_1> is an integer that has two interpretations depending on the value of hyper_flag:
	2.a.) if hyper_flag is True, this integer represents m = <hidden_1>+1 maximal nodes to evaluate in your first hidden layer
	2.b.) if hyper_flag is False, this integer represents the known amount of hidden nodes in the first hidden layer of the optimal network architecture
3) int <hidden_2> is an integer that has two interpretations depending on the value of hyper_flag:
	3.a.) if hyper_flag is True, this integer represents n = <hidden_2>+1 maximal nodes to evaluate in your first hidden layer
	3.b.) if hyper_flag is False, this integer represents the known amount of hidden nodes in the second hidden layer of the optimal network architecture
4) float <lr_start> is a float with a value between 0 and 1. This has two interpretations depending on the value of hyper_flag:
	4.a.) if hyper_flag is True, this float represents the lower bound of the learning rates to search
	4.b.) if hyper_flag is False, this float represents the known best learning rate of the optimal network architecture
5) float <lr_end> is a float with a value between 0 and 1. If hyper_flag is True, the program will evaulate a 20 equally spaced points between <lr_start> and <lr_end>

WARNING: This program can take a substantial amount of time to run. It evaluates m*n*20*k neural networks (where k was selected as 10 for 10-fold cross validation). 
Searching for the best architecture in m=41, n=41, k=10 took approximately 10 hours.

The program predicts the winner of the plurality vote in each county in America for the 2020 presidential election using 10-fold CV combined with neural networks. 
It can optionally find the best network architecture if given enough time and will evaulate a given or found architecture over 100 training epochs. 


Running:
To run the program, first change your directory to where the file and dataset are stored. Then type "python -m main <hyper_flag> <hidden_1> <hidden_2> <lr_start> <lr_end>"  into your command prompt.
	Example 1:
	To find the best hyperparameters in a network that can have between 1-50 nodes per hidden layer and a learning rate between 0.2-0.4, enter "python -m main 1 49 49 0.2 0.4"
	Example 2: 
	To evaulate a known architecture with 10 hidden nodes in hidden layer 1 and 5 in hidden layer 2 with a learning rate of 0.31, enter "python -m main 0 10 5 0.31 0.31"
