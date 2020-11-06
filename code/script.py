import theano
import theano.tensor as T
import theano.tensor.nlinalg as Tla
import lasagne       # the library we're using for NN's
from lasagne.nonlinearities import leaky_rectify, softmax, linear, tanh, rectify, sigmoid
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
from numpy.random import *
from matplotlib import pyplot as plt
import sys

sys.path.append('lib')  # support files (mathematical tools, mostly)

from generativeModel import *       # Class file for generative models.
from recognitionModel import *      # Class file for recognition models
from SGVB import *                  # The meat of the algorithm - define the ELBO and initialize Gen/Rec model


def generate(gendict, xDim, yDim, samples):
    gen_nn = lasagne.layers.InputLayer((None, xDim))
    gen_nn = lasagne.layers.DenseLayer(gen_nn, yDim, nonlinearity=linear, W=lasagne.init.Orthogonal())
    NN_XtoY_Params = dict([('network', gen_nn)])

    gendict['NN_XtoY_Params'] = NN_XtoY_Params


    # Instantiate a PLDS generative model:
    true_model = PLDS(gendict, xDim, yDim, srng = RandomStreams(seed=20150503), nrng = np.random.RandomState(20150503))

    # Now, we can sample from it:
    Tt = samples # How many samples do we want?
    [x_data, y_data] = true_model.sampleXY(Tt) # sample from the generative model
    print(true_model.evaluateLogDensity(x_data[:100], y_data[:100]).eval())

    return x_data, y_data, true_model


def recognise(recdict, Y, xDim, yDim, y_data, srng, nrng):
    # Describe network for mapping into means
    NN_Mu = lasagne.layers.InputLayer((None, yDim))
    NN_Mu = lasagne.layers.DenseLayer(NN_Mu, 25, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    # --------------------------------------
    # let's initialize the first layer to have 0 mean wrt our training data
    W0 = np.asarray(NN_Mu.W.get_value(), dtype=theano.config.floatX)
    NN_Mu.W.set_value((W0 / np.dot(y_data, W0).std(axis=0)).astype(theano.config.floatX))
    W0 = np.asarray(NN_Mu.W.get_value(), dtype=theano.config.floatX)
    b0 = (-np.dot(y_data, W0).mean(axis=0)).astype(theano.config.floatX)
    NN_Mu.b.set_value(b0)
    # --------------------------------------
    NN_Mu = lasagne.layers.DenseLayer(NN_Mu, xDim, nonlinearity=linear, W=lasagne.init.Normal())
    NN_Mu.W.set_value(NN_Mu.W.get_value() * 10)
    NN_Mu = dict([('network', NN_Mu)])

    ########################################
    # Describe network for mapping into Covariances
    NN_Lambda = lasagne.layers.InputLayer((None, yDim))
    NN_Lambda = lasagne.layers.DenseLayer(NN_Lambda, 25, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    # --------------------------------------
    # let's initialize the first layer to have 0 mean wrt our training data
    W0 = np.asarray(NN_Lambda.W.get_value(), dtype=theano.config.floatX)
    NN_Lambda.W.set_value((W0 / np.dot(y_data, W0).std(axis=0)).astype(theano.config.floatX))
    W0 = np.asarray(NN_Lambda.W.get_value(), dtype=theano.config.floatX)
    b0 = (-np.dot(y_data, W0).mean(axis=0)).astype(theano.config.floatX)
    NN_Lambda.b.set_value(b0)
    # --------------------------------------
    NN_Lambda = lasagne.layers.DenseLayer(NN_Lambda, xDim * xDim, nonlinearity=linear, W=lasagne.init.Orthogonal())
    NN_Lambda.W.set_value(NN_Lambda.W.get_value() * 10)
    NN_Lambda = dict([('network', NN_Lambda)])

    recdict['NN_Lambda'] = NN_Lambda
    recdict['NN_Mu'] = NN_Mu

    rec_model = SmoothingLDSTimeSeries(recdict, Y, xDim, yDim, srng, nrng)

    return rec_model





def train_recmodel(training_obs, training_latents, PLDS=PLDS, SmoothingLDSTimeSeries=SmoothingLDSTimeSeries):
    y_data = training_obs
    x_data = training_latents
    # Instantiate an SGVB class:
    sgvb = SGVB(initGenDict, PLDS, recdict, SmoothingLDSTimeSeries, xDim = xDim, yDim = yDim)

    ########################################
    # Define a bare-bones thenao training function
    batch_y = T.matrix('batch_y')

    ########################################
    # choose learning rate and batch size
    learning_rate = 1e-2
    batch_size = 100

    ########################################
    # use lasagne to get adam updates
    updates = lasagne.updates.adam(-sgvb.cost(), sgvb.getParams(), learning_rate=learning_rate)

    ########################################
    # Finally, compile the function that will actually take gradient steps.
    train_fn = theano.function(
             outputs=sgvb.cost(),
             inputs=[theano.In(batch_y)],
             updates=updates,
             givens={sgvb.Y: batch_y},
        )

    ########################################
    # set up an iterator over our training data
    yiter = DatasetMiniBatchIndexIterator(y_data, batch_size=batch_size, randomize=True)

    ########################################
    # Iterate over the training data for the specified number of epochs
    n_epochs = 20
    cost = []
    for ie in np.arange(n_epochs):
        print('--> entering epoch %d' % ie)
        for y, _ in yiter:
            cost.append(train_fn(y)) #cost = elbo cost
    #        print cost[-1]


    #Trained GEN & REC MODELS
    #########################
    # Since the model is non-identifiable, let's find the best linear projection from the
    # learned posterior mean into the 'true' training-data latents
    pM = sgvb.mrec.postX.eval({sgvb.Y: y_data})
    wgt = np.linalg.lstsq(pM-pM.mean(), x_data-x_data.mean())[0]

    return pM, wgt, sgvb



def visualise(pM, wgt, sgvb, true_model):
    #########################
    # Plot posterior means
    nT = 200  # number of timepoints to visualize
    # Let's simulate some new test data from our original generative model
    [x_test, y_test] = true_model.sampleXY(nT)

    #########################
    # sample from the trained recognition model
    rtrain_samp = sgvb.mrec.getSample()

    plt.figure()
    #plt.hold('on')

    #########################
    # plot 25 samples from the posterior
    for idx in np.arange(25):  # plot multiple samples from the posterior
        xs = rtrain_samp.eval({sgvb.Y: y_test})
        plt.plot(np.dot(xs, wgt), 'k')
    # and now plot the posterior mean
    pMtest = sgvb.mrec.postX.eval({sgvb.Y: y_test})
    plt_post = plt.plot(np.dot(pMtest, wgt), 'r', linewidth=3, label='posterior mean')

    plt_true = plt.plot(x_test, 'g', linewidth=3, label='\"true\" mean')

    plt.legend(handles=plt_post + plt_true)
    plt.xlabel('time')
    plt.title('samples from the trained approximate posterior')
    plt.show()


if __name__ == '__main__':


    #user config file
    xDim = 1
    yDim = 20

    gendict = dict([('A'     , 0.8*np.eye(xDim)),         # Linear dynamics parameters
                    ('QChol' , 2*np.diag(np.ones(xDim))), # innovation noise
                    ('Q0Chol', 2*np.diag(np.ones(xDim))),
                    ('x0'    , np.zeros(xDim)),
    #                ('RChol', np.ones(yDim)),             # observation covariance
                    ('NN_XtoY_Params', None),    # neural network output mapping
                    ('output_nlin' , 'softplus')  # for poisson observations
                    ])





    recdict = dict([('A'     , .9*np.eye(xDim)),
                    ('QinvChol',  np.eye(xDim)), #np.linalg.cholesky(np.linalg.inv(np.array(tQ)))),
                    ('Q0invChol', np.eye(xDim)), #np.linalg.cholesky(np.linalg.inv(np.array(tQ0)))),
                    ('NN_Mu' ,None),
                    ('NN_Lambda',None),
                    ])


    #Generate training data (latents & observations)
    #Train recognition model with observations

    Y = T.matrix() #matrix of observations
    x_data, y_data, true_model = generate(gendict, xDim, yDim, 10000) #x = latent state, y= observations
    rec_model = recognise(recdict, Y, xDim, yDim, y_data, srng = RandomStreams(seed=20150503), nrng = RandomStreams(seed=20150503))


    # initialize training with a random generative model (that we haven't generated data from):
    initGenDict = dict([
                 ('output_nlin' , 'softplus')
                     ])

    pM, wgt, sgvb = train_recmodel(y_data, x_data)
    visualise(pM, wgt, sgvb, true_model)
