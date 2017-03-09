from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn.base
from scipy.stats import pearsonr
from joblib import delayed, Parallel
import sys

import tensorflow as tf

sns.set_style("white")

tf.set_random_seed(420)

def generate_timeseries(m=1., N=100, sigma=1.0, step=0.1):
    '''
    y(t) = m * t + N(0, sigma)
    '''
    np.random.seed(420)
    x = np.arange(N*step, step=step)
    signal = m*np.sin(x)
    #signal += m*x
    y = signal + np.random.normal(0, sigma, N)
    plt.plot(y)
    plt.title("Sinusoidal - Noise %1.3f" % sigma)
    plt.xlabel("$t$")
    plt.ylabel("$y_t$")
    plt.savefig("figures/sin_%1.3f.pdf" % sigma)
    plt.close()
    return y, signal

def read_eurchf():
    import pandas as pd
    df = pd.read_csv("exchange.csv")
    return df['EUR/CHF Close'].values

def read_sp500():
    import pandas as pd
    df = pd.read_csv("sp500.csv", index_col=0)
    ts = df['Close']
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    ts.plot(title="S&P 500 - Since 1950")
    plt.subplot(1,2,2)
    ts.ix[-365*3:].plot(title='S&P 500 - Last 3 Years')
    plt.savefig("figures/sp500-ts.pdf")
    plt.close()
    return ts.values

# helper function to make features and labels from time series
def _build_data(y, memory):
    features, labels = [], []
    for t in range(memory+1, len(y)):
        features.append(y[t-memory:t])
        labels.append(y[t])
    features = np.array(features)
    labels = np.array(labels)
    assert np.any(labels != features[:,-1])
    return features, labels

def ann_job(model, X, y):
    return model.fit(X, y)

class MackayCheapCheerful(object):
    def __init__(self, memory, hidden_layers=[32,32], learning_rate=0.001, 
                N=10):
        '''
        memory: the number of steps we remember
        hidden layers: size of hidden layers in neural network
        learning rate: learning rate for neural network (exponentially decays)
        N: Number of NNs to finetune with noise
        '''
        self.memory = memory
        self.N = N
        self.base_mlp = MLPRegressor(hidden_layer_sizes=hidden_layers,
                                    learning_rate_init=learning_rate,
                                    max_iter=5000, learning_rate='constant',
                                    warm_start=True, alpha=0, solver='adam')

    def train(self, y):
        X, y = _build_data(y, self.memory)
        self.base_mlp.fit(X,y)
        yhat = self.base_mlp.predict(X)
        sigma = np.sqrt(np.sum((y - yhat)**2) / (len(y) - self.memory - 1))

        # generate new models
        self.models = [sklearn.base.clone(self.base_mlp) for k in range(self.N)]
        jobs = []
        for k in range(self.N):
            noise = np.random.normal(0, sigma, y.shape[0])
            jobs.append(delayed(ann_job)(self.models[k], X, y+noise))
        self.models = Parallel(n_jobs=10)(jobs)

    def predict_stepwise(self, init_state, steps):
        yhat = init_state.tolist()
        ysamples = np.empty((steps, self.N))
        for step in range(0, steps):
            for j, model in enumerate(self.models):
                x_  = np.asarray(yhat[-self.memory:])[np.newaxis,:]
                ysamples[step,j] = model.predict(x_)[0]
            yhat.append(ysamples[step].mean())

        return ysamples

    def predict_independent(self, init_state, steps):
        assert len(init_state) == self.memory
        y_samples = np.empty((steps+init_state.shape[0], self.N))
        y_samples[:init_state.shape[0]] = init_state[:, np.newaxis]
        for i in range(self.N):
            y = y_samples[:self.memory,i]
            for j in range(self.memory, steps+self.memory):
                y = y_samples[j-self.memory:j,i][np.newaxis,:]
                y_samples[j,i] = self.models[i].predict(y)[0] 
        return y_samples[self.memory:]

class GalDropout(object):
    def __init__(self, memory, hidden_layers=[64,64], learning_rate=0.001,
                dropout_rate=0.20, epochs=1000, mc_runs=20):
        self.memory = memory
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.graph = tf.Graph()
        self.epochs = epochs
        self.mc_runs = mc_runs

        self._build_graph()

    def _build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=(None, self.memory))
        self.y = tf.placeholder(tf.float32, shape=(None,1))
        self.global_step = tf.Variable(1, dtype=tf.float32)
        with tf.device("/gpu:0"):
            h = self.x
            h = tf.nn.dropout(h, 1-self.dropout_rate) * (1 - self.dropout_rate)
            #h = tf.multiply(self.x, tf.random_normal(shape=(1, self.memory), mean=1.0, stddev=1.0,
            #                                         dtype=tf.float32))
            for h_layer in self.hidden_layers:
                h = tf.contrib.layers.fully_connected(h, num_outputs=h_layer, activation_fn=tf.nn.relu)
                h = tf.nn.dropout(h, 1-self.dropout_rate) * (1-self.dropout_rate)
                #h = tf.multiply(h, tf.random_normal(shape=(1, h_layer), mean=1.0, stddev=1.00,
                #                                    dtype=tf.float32))
            self.yhat = tf.contrib.layers.fully_connected(h, num_outputs=1, activation_fn=None)
            self.loss = 0.5*tf.reduce_mean(tf.square(self.y - self.yhat)) #tf.losses.mean_squared_error(self.y, self.yhat)
            opt = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer = opt.minimize(self.loss, global_step=self.global_step)

    def train(self, y, sess):
        x, y = _build_data(y, self.memory)
        feed_dict = {self.x: x, self.y: y[:,np.newaxis]}
        for epoch in range(self.epochs+1):
            res = sess.run([self.optimizer, self.loss, self.yhat], feed_dict=feed_dict)
            if epoch % 100 == 0:
                print "(Training) Interation: %i, Loss: %f, yhat mean: %f" % (epoch, res[1], np.mean(res[2]))

    def predict_stepwise(self, init_state, steps, sess):
        y = init_state.tolist()
        y_samples = np.empty((steps, self.mc_runs))
        for step in range(steps):
            feed_dict = {self.x: np.array(y[-self.memory:])[np.newaxis,:]}
            y_samples[step, :] =  [sess.run(self.yhat, feed_dict=feed_dict)[0][0] for j in range(self.mc_runs)]
            y.append(y_samples[step].mean())
        return y_samples

    def predict_independent(self, init_state, steps, sess):
        assert len(init_state) == self.memory
        y_samples = np.empty((steps+init_state.shape[0], self.mc_runs))
        y_samples[:init_state.shape[0]] = init_state[:, np.newaxis]
        for i in range(self.mc_runs):
            y = y_samples[:self.memory,i]
            for j in range(self.memory, steps+self.memory):
                y = y_samples[j-self.memory:j,i][np.newaxis,:]
                feed_dict = {self.x: y}
                y_samples[j,i] = sess.run(self.yhat, feed_dict=feed_dict)
        return y_samples[self.memory:]

def make_figure(y_samples, y_signal, y_noisy, title=None, show=False):
    sd = np.std(y_samples, axis=1).mean()
    corr = pearsonr(y_samples.mean(axis=1), y_signal)[0]
    rmse = np.mean((y_samples.mean(axis=1) - y_signal)**2)**0.5
    info = "Std: %0.3f, Corr: %0.2f, RMSE: %0.3f" % (sd, corr, rmse)
    plt.annotate(info, xy=(0.05, 0.95), xycoords='axes fraction')
    plt.plot(y_samples, alpha=0.1, color='blue', lw=0.2)
    plt.plot(y_noisy, lw=0.5, label="noisy signal", color='black')
    plt.plot(y_samples.mean(axis=1), label="predicted", color='blue', lw=0.8)
    plt.plot(y_signal, label="signal", color='black', ls='--', lw=0.8)
    plt.legend()
    plt.title(title)
    plt.ylabel('f(t)')
    plt.xlabel('t')
    if show:
        plt.show()

def finance_figure(y_samples, y_signal, y_past, title=None, show=False):
    sd = np.std(y_samples, axis=1).mean()
    info = "Std: %2.3f" % (sd)
    plt.annotate(info, xy=(0.05, 0.95), xycoords='axes fraction')
    x_past = range(len(y_past))
    x_test = range(len(y_past), len(y_past) + len(y_signal))
    x_sample = range(len(y_past), len(y_past) + len(y_samples))

    sample_low = np.percentile(y_samples.flatten(), 2)
    sample_high = np.percentile(y_samples.flatten(), 98)
    y_low = min([sample_low, min(y_past), min(y_signal)])
    y_high = max([sample_high, max(y_signal), max(y_past)])

    plt.plot(x_past, y_past, label="signal", color='black', ls='-', lw=0.8)
    plt.plot(x_sample, y_samples, alpha=0.1, color='blue', lw=0.2)
    plt.plot(x_sample, y_samples.mean(axis=1), label="predicted", color='blue', lw=0.8)
    plt.plot(x_test, y_signal, color='black', ls='-', lw=0.8)
    plt.ylim([y_low, y_high])
    plt.legend()
    plt.title(title)
    plt.ylabel('f(t)')
    plt.xlabel('t')
    if show:
        plt.show()

def run_gal(noise=0.05, N=1000, test_n=300, memory=20, mc_runs=20, which='stepwise'):
    hidden_layers = [128,128]
    y, signal = generate_timeseries(N=N, sigma=noise, step=0.1)
    ytrain, ytest = y[:-test_n], y[-test_n:]

    gal = GalDropout(memory, epochs=20000, dropout_rate=0.10, learning_rate=1e-2,
                     mc_runs=mc_runs, hidden_layers=hidden_layers)
    with tf.Session().as_default() as sess:
        tf.global_variables_initializer().run()
        gal.train(ytrain, sess)
        if which == 'stepwise':
            y_samples = gal.predict_stepwise(ytrain[-memory:], test_n, sess)
        else:
            y_samples = gal.predict_independent(ytrain[-memory:], test_n, sess)

    title = "MC-Dropout -- Noise: %1.3f"  % noise
    make_figure(y_samples, signal[-test_n:], ytest, title=title)
    plt.savefig("figures/%s_galdropout_%1.3f.pdf" % (which, noise))
    #plt.show()
    plt.close()

def run_mackay(noise=0.05, N=1000, test_n=300, memory=20, N_models=20):
    hidden_layers = [128,128]
    y, signal = generate_timeseries(N=N, sigma=noise, step=0.1)
    ytrain, ytest = y[:-test_n], y[-test_n:]

    mackay = MackayCheapCheerful(memory=memory, N=N_models, hidden_layers=hidden_layers)
    mackay.train(ytrain)
    if which == 'stepwise':
        yhat = mackay.predict_stepwise(ytrain[-memory:], steps=test_n)
    else:
        yhat = mackay.predict_independent(ytrain[-memory:], steps=test_n)
    title = "Mackay Cheap+Cheerful -- Noise: %1.3f"  % noise
    make_figure(yhat, signal[-test_n:], ytest, title=title)
    plt.savefig("figures/%s_mackay_%1.3f.pdf" % (which, noise))
    plt.close()

def gal_finance(which='sp500', memory=90, mc_runs=30,
                dropout_rate=0.2, hidden_layers=[256, 256]):
    if which=='eurchf':
        y = read_eurchf()
    elif which=='sp500':
        y = read_sp500()
    test_n = 365
    ytrain, ytest = y[:-test_n], y[-test_n:]

    gal = GalDropout(memory, epochs=2000, dropout_rate=dropout_rate, learning_rate=1e-2,
                     mc_runs=mc_runs, hidden_layers=hidden_layers)
    with tf.Session().as_default() as sess:
        tf.global_variables_initializer().run()
        gal.train(ytrain, sess)
        y_samples = gal.predict_independent(ytrain[-memory:], int(test_n), sess)
        title = "MC-Dropout -- S&P500 (Independent)"
        finance_figure(y_samples, ytest, ytrain[-2*365:], title=title)
        plt.savefig("figures/galdropout_independent_finance.pdf")
        plt.close()

        y_samples = gal.predict_stepwise(ytrain[-memory:], int(test_n), sess)
        title = "MC-Dropout -- S&P500 (Stepwise)"
        finance_figure(y_samples, ytest, ytrain[-2*365:], title=title)
        plt.savefig("figures/galdropout_stepwise_finance.pdf")
        plt.close()

def mackay_finance(which='sp500', memory=90, models=30,
                dropout_rate=0.2, hidden_layers=[256, 256]):
    y = read_sp500()
    test_n = 365
    ytrain, ytest = y[:-test_n], y[-test_n:]

    mackay = MackayCheapCheerful(memory=memory, N=models, hidden_layers=hidden_layers)
    print "Training Mackay Finance"
    mackay.train(ytrain)

    print "Inferring finance"
    ysamples = mackay.predict_stepwise(ytrain[-memory:], steps=test_n)
    title = "Mackay -- S&P500 (Independent)"
    finance_figure(ysamples, ytest, ytrain[-2*365:], title=title)
    plt.savefig("figures/mackay_independent_finance.pdf")
    plt.close()

    ysamples = mackay.predict_independent(ytrain[-memory:], steps=test_n)
    title = "Mackay -- S&P500 (Stepwise)"
    finance_figure(ysamples, ytest, ytrain[-2*365:], title=title)
    plt.savefig("figures/mackay_stepwise_finance.pdf")
    plt.close()

if __name__ == "__main__":
    mackay_finance()
    gal_finance()
    for which in ['stepwise', 'independent']:
        for noise in [0.1, 1.]:
            run_gal(noise)
            print "Cheap and Cheerful Training Mackay"
            run_mackay(noise)
