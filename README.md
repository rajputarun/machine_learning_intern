# machine_learning_intern
Machine Learning Internship Challenge to build a model that automatically classifies a product’s pattern attribute based on its image.

<h1>Approach:</h1>
I initially started the project with just tensorflow but it was cumbersome to use, so I switched to keras.
Keras is a high level and powerful API that makes the machine learning implementation very easy. 
I referred to "https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html" to kick start things.
But above blog provides tutorial for binary classification(cats or dogs)
So to solve the multi-class classification task like given in the intern challenge, some modifications were required.
General idea remains the same, use of  Transfer Learning to get the pre-trained network(here VGG16) and imagenet weights, and remove final fully-connected layers from that model. 
Then use the remaining portion of the model as a feature extractor for the fashion dataset. These extracted features aka. "Bottleneck Features" - the last activation maps before the fully-connected layers in the original model.
We then train a small fully-connected network on those extracted bottleneck features in order to get the classes we need as outputs for our problem.

<h2>Implementation summary :</h2>

Given dataset contains Train and Test folders. 
Train folder contains 12 sub-directories. The names of the sub-directories will be the names of our classes.
It is a good practice to split the dataset into Training data and Validation data. We roughly use division ratio of  2/3 and 1/3 respectively.
Training data will be used for training of model and validation data to tune the hyperparameters to improve the accuracy and minimize the loss function.
<h3>Steps to build model:</h3>
    1.Extract and Save the bottleneck features from the VGG16 model.(use of Train folder)
    2.Train a small fully-connected ANN using the saved bottleneck features to classify our classes, and save the model so we can use it for prediction later 
    3.Use both the VGG16 model along with the top model to make predictions and save predictions to a CSV file.(use of Test folder)
Note: Other relevant details are commented in the code.
I used batch_size =16, because batch_size > 64 crashes my system.
In just 50 epochs, model accuracy shoot to 94 %.


<h4>How to run the code(from scratch):</h4>
There are two .py files
1.  "keras_bottleneck_multiclass.py"
      -execute the save_bottlebeck_features() only once. It will generate two npy files 'bottleneck_features_train.npy' and  'bottleneck_features_validation.npy'
      -This file extract features from train and validation directory, and generate class labels
      -train_top_model() can be executed multiple times to train model with different parameters.
      -Training takes less time, for 50 epoch it was roughly 10 minutes
      -It will generate 'class_indices.npy' and save weights for future prediction 'bottleneck_fc_model.h5'
2. "bottleneck_predict.py"
      -I takes inputs generate from above file, 'bottleneck_fc_model.h5' and  'class_indices.npy' 
      -here we only need to pass the image_path
      - It will print the class_id, predicted label and filename.
      -Filenames and respective labels will be saved to a CSV file.


<h5>Improvements:</h5>
Generally, 50 epochs are less to train a model but further tweaking the hyperparameters, and use of Image Augmentation will definitely increasing the validation accuracy.



<p>
References:
[Credit goes to Francois Chollet for his ingenious work]
"Using the bottleneck features of a pre-trained network: 90% accuracy in a minute"
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

Research paper and studies=>
"Dropout: A Simple Way to Prevent Neural Networks from Overfitting".

Which Optimizer to choose?
https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/

Impact of No. of Neurons
http://www.chioka.in/how-does-the-number-of-hidden-neurons-affect-a-neural-networks-performance/

</p>
