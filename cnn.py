"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

from PIL import Image
import glob
from multiprocess import Pool
import gc
import time

import subprocess

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def print_cm(cm, labels, hide_labels=True, file=None, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    #TODO: implementar hide_labels para saida em arquivo
    columnwidth = max([len(x) for x in labels]+[0]) # 5 is value length

    if hide_labels:
        columnwidth = 3

    empty_cell = " " * columnwidth
    # Print header
    if file is None:
        print "    " + empty_cell,
    else:
        file.write("    " + empty_cell)
    i = 0
    for label in labels:
        if file is None:
            if hide_labels:
                print "%{0}d".format(columnwidth) % i,
            else:
                print "%{0}s".format(columnwidth) % label,
        else:
            file.write("%{0}s".format(columnwidth) % label)
        i+=1
    if file is None:
        print
    else:
        file.write("\n")
    # Print rows
    for i, label1 in enumerate(labels):
        if file is None:
            if hide_labels:
                print "    %{0}d".format(columnwidth) % i,
            else:
                print "    %{0}s".format(columnwidth) % label1,
        else:
            file.write("    %{0}s".format(columnwidth) % label1)
        for j in range(len(labels)):
            cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            if file is None:
                print cell,
            else:
                file.write(cell)
        if file is None:
            print
        else:
            file.write("\n")

    for i, l in enumerate(labels):
        print "%d: %s" % (i, l)

def load_image(work):
    i = open(work)
    img = Image.open(i)
    #img = img.convert('RGB')
    out = np.asarray(img, dtype='float32') / 256.0
    out = np.reshape(out, (-1, out.shape[0], out.shape[1]))
    img.close()
    i.close()
    return out

def load_dataset(file_list):
    out = []
    for f in file_list:
        out.append(load_image(f))
    return np.asarray(out)


# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_mlp(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out


def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2,
                     drop_hidden=.5):
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a network in Python code can be a lot more
    # flexible than a configuration file. Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(
                network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
    return network


def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 256, 16),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 64 kernels of size 5x5, stride of 1. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(5,5), stride=1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2), stride=2)

    # Another convolution with 64 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(5,5), stride=1,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2), stride=2)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    # network = lasagne.layers.DenseLayer(
    #         lasagne.layers.dropout(network, p=.5),
    #         num_units=dense_layer_neurons,
    #         nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=512,
        nonlinearity=lasagne.nonlinearities.rectify # maybe softmax?
    )

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    # network = lasagne.layers.DenseLayer(
    #         lasagne.layers.dropout(network, p=.5),
    #         num_units=out_units,
    #         nonlinearity=lasagne.nonlinearities.softmax)

    network = lasagne.layers.DenseLayer(
	lasagne.layers.dropout(network, p=.5),
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax
    )

    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def get_fns(input_var, target_var):
    print("Building model and compiling functions...")

    network = build_cnn(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    test_fn = theano.function([input_var], [test_prediction])

    print("...done!")

    return train_fn, val_fn, test_fn, network

def reset_weights(network):
    params = lasagne.layers.get_all_params(network, trainable=True)
    for v in params:
        val = v.get_value()
        if(len(val.shape) < 2):
            v.set_value(lasagne.init.Constant(0.0)(val.shape))
        else:
            v.set_value(lasagne.init.GlorotUniform()(val.shape))

def load_meta(meta_file):
    with open(meta_file, 'r') as f:
        content = f.readlines()

    names = []
    labels_text = []

    for track in content:
        d = track.split("\t")
        names.append(d[0])
        labels_text.append(d[1].strip())
    
    label_codes = {
        'blues': 0,
        'classical': 1,
        'country': 2,
        'disco': 3,
        'hiphop': 4,
        'jazz': 5,
        'metal': 6,
        'pop': 7,
        'reggae': 8,
        'rock': 9
    }

    labels = []
    for i in labels_text:
        labels.append(label_codes[i])    

    names = np.array(names)
    labels = np.array(labels, dtype='int32')
    
    return names, labels

def get_class_names():
    return ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def main(model='mlp', num_epochs=80, meta_slices_file="meta_jgtzan100_slices_z120.txt", 
    meta_full_file="meta_jgtzan100_z120.txt", batch_size=500, slices_per_track=50):


    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    
    train_fn, val_fn, test_fn, network = get_fns(input_var, target_var)

    #load meta_file(s)

    slices_names, slices_labels = load_meta(meta_slices_file)
    full_names, full_labels = load_meta(meta_full_file)

    skf = StratifiedKFold(n_splits=10)

    #print(labels)

    k = 0

    for train_idx_f, test_idx_f in skf.split(full_names, full_labels):

        train_idx = []
        test_idx = []

        for i in train_idx_f:
            train_idx.extend( (i * slices_per_track) + np.arange(slices_per_track) )

        for i in test_idx_f:
            test_idx.extend( (i * slices_per_track) + np.arange(slices_per_track))

        k+=1 

	print("Loading fold %d" % (k))
        train_data = load_dataset(slices_names[train_idx])
        test_data = load_dataset(slices_names[test_idx])


        print("...done!")
        
        for epoch in range(num_epochs):

            start_time = time.time()
            train_err = 0
            train_batches = 0

            for batch_data, batch_labels in iterate_minibatches(train_data, slices_labels[train_idx], batchsize=batch_size, shuffle=True):
                train_err += train_fn(batch_data, batch_labels)
                train_batches+=1

            val_err = 0
            val_acc = 0
            val_batches = 0

            for batch_data, batch_labels in iterate_minibatches(test_data, slices_labels[test_idx], batchsize=batch_size, shuffle=True):
                err, acc = val_fn(batch_data, batch_labels)
                val_err+=err
                val_acc+=acc
                val_batches+=1

            if(epoch % 10 == 0 or epoch == (num_epochs - 1)):
                # Then we print the results for this epoch:
                print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, num_epochs, time.time() - start_time))
                print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
                print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
                print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

        y_true = []
        y_predicted = []
        

        print("Testing FOLD %d..." % (k))
        #print(test_idx_f)
        for i in xrange(len(test_idx_f)):
            #print("sample %d" % (i))
            x = test_data[ (i * slices_per_track) + np.arange(slices_per_track) ]
            y = test_fn(x)
            y = np.array(y)[0]
            s = y.sum(axis=0)
            #print(s, s.shape)
            prediction = s.argmax()
            y_predicted.append(prediction)
            y_true.append(full_labels[test_idx_f[i]])
            #print("true: %d, predicted: %d" % (y_true[-1], y_predicted[-1]))

        print("Results for FOLD %d:" % (k))
        print("Accuracy: %.2f" % (accuracy_score(y_true, y_predicted)) )
        print(classification_report(y_true, y_predicted, target_names=get_class_names()))
        print("Confusion Matrix")
        print_cm(confusion_matrix(y_true, y_predicted), get_class_names())

	sys.stdout.flush()        

        #print ("FINAL TEST SET STATS FOR FOLD %d:" % (k))
        #print ("validation loss:\t\t{:.6f}".format(val_err / val_batches))
        #print ("validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

        train_data = None
        test_data = None
        
        reset_weights(network)

    # # Finally, launch the training loop.
    # print("Starting training...")
    # # We iterate over epochs:
    # for epoch in range(num_epochs):
    #     # In each epoch, we do a full pass over the training data:
    #     train_err = 0
    #     train_batches = 0
    #     start_time = time.time()
    #     for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
    #         inputs, targets = batch
    #         train_err += train_fn(inputs, targets)
    #         train_batches += 1

    #     # And a full pass over the validation data:
    #     val_err = 0
    #     val_acc = 0
    #     val_batches = 0
    #     for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
    #         inputs, targets = batch
    #         err, acc = val_fn(inputs, targets)
    #         val_err += err
    #         val_acc += acc
    #         val_batches += 1

    #     # Then we print the results for this epoch:
    #     print("Epoch {} of {} took {:.3f}s".format(
    #         epoch + 1, num_epochs, time.time() - start_time))
    #     print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    #     print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    #     print("  validation accuracy:\t\t{:.2f} %".format(
    #         val_acc / val_batches * 100))

    # # After training, we compute and print the test error:
    # test_err = 0
    # test_acc = 0
    # test_batches = 0
    # for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
    #     inputs, targets = batch
    #     err, acc = val_fn(inputs, targets)
    #     test_err += err
    #     test_acc += acc
    #     test_batches += 1
    # print("Final results:")
    # print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    # print("  test accuracy:\t\t{:.2f} %".format(
    #     test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on JGTZAN100 using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODELS:")
        #print("       'mlp' for a simple Multi-Layer Perceptron (MLP),")
        #print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        #print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        #print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)
