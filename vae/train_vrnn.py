from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import glob
import time
from datetime import datetime
import os
import pickle

from model_vrnn import VRNN

from matplotlib import pyplot as plt

'''
TODOS:
    - parameters for depth and width of hidden layers
    - implement predict function
    - separate binary and gaussian variables
    - clean up nomenclature to remove MDCT references
    - implement separate MDCT training and sampling version
'''
def denormalize(normArray):
    def denormalize_one_dim(data,maxV=1, minV=0, high=1, low=-1):
        return ((((data - high) * (maxV - minV))/(high - low)) + maxV)

    Array = normArray.copy()

    # MinMax Values for different dataset
    # GooG
    maxV =  [23,16500,1,1,942,150,942,942,3000,3000]
    minV = [0,0,0,0,916,0,916,916,1,1]

    for i in range(Array.shape[2]):
        Array[:,:,i] = denormalize_one_dim(normArray[:,:,i],maxV=maxV[i],minV=minV[i])

    return Array

def normalize(normArray):
    def normalize_one_dim(array, maxV=1, minV=0, high=1, low=-1):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i,j] < minV:
                    array[i,j] = minV
                if array[i,j] > maxV:
                    array[i,j] = maxV
        return (high - (((high - low) * (maxV - array)) / (maxV - minV)))

    Array = normArray.copy()

    #MinMax Values for different dataset
    # GooG
    maxV =  [23,16500,1,1,942,150,942,942,3000,3000]
    minV = [0,0,0,0,916,0,916,916,1,1]

    for i in range(Array.shape[2]):
        Array[:,:,i] = normalize_one_dim(normArray[:,:,i],maxV=maxV[i],minV=minV[i])
    return Array


def next_batch(args):
    #t0 = np.random.randn(args.batch_size, 1, (2 * args.chunk_samples))
    #mixed_noise = np.random.randn(
    #    args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1
    #x = t0 + mixed_noise + np.random.randn(
    #    args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1
    #y = t0 + mixed_noise + np.random.randn(
    #    args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1
    #x = np.sin(2 * np.pi * (np.arange(args.seq_length)[np.newaxis, :, np.newaxis] / 10. + t0)) + np.random.randn(
    #    args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1 + mixed_noise*0.1
    #y = np.sin(2 * np.pi * (np.arange(1, args.seq_length + 1)[np.newaxis, :, np.newaxis] / 10. + t0)) + np.random.randn(
    #    args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1 + mixed_noise*0.1

    #y[:, :, args.chunk_samples:] = 0.
    #x[:, :, axrgs.chunk_samples:] = 0.
    #return x, y
    try:
        idx = np.random.randint(0, data.shape[0])
    except:
        data = np.load(args.data_path + 'data.npy', mmap_mode='r')
        idx = np.random.randint(0, data.shape[0])

    orderStreams_train = normalize(np.squeeze(data[idx].copy()))
    orderStreams_train = np.concatenate((orderStreams_train,np.zeros(orderStreams_train.shape)),axis=-1)
    return orderStreams_train,orderStreams_train


def train(args, model):
    dirname = 'save-vrnn'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    ckpt = tf.train.get_checkpoint_state(dirname)
    n_batches = 100
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('logs/' + datetime.now().isoformat().replace(':', '-'), sess.graph)
        check = tf.add_check_numerics_ops()
        merged = tf.summary.merge_all()
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Loaded model")
        start = time.time()
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            state = model.initial_state_c, model.initial_state_h
            for b in range(n_batches):
                x, y = next_batch(args)
                feed = {model.input_data: x, model.target_data: y}
                train_loss, _, cr, summary, sigma, mu, input, target= sess.run(
                        [model.cost, model.train_op, check, merged, model.sigma, model.mu, model.flat_input, model.target],
                                                             feed)
                summary_writer.add_summary(summary, e * n_batches + b)
                if (e * n_batches + b) % args.save_every == 0 and ((e * n_batches + b) > 0):
                    checkpoint_path = os.path.join(dirname, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * n_batches + b)
                    print("model saved to {}".format(checkpoint_path))
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.6f}, time/batch = {:.1f}, std = {:.3f}" \
                    .format(e * n_batches + b,
                            args.num_epochs * n_batches,
                            e, args.chunk_samples * train_loss, end - start, sigma.mean(axis=0).mean(axis=0)))
                start = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=5,
                        help='size of RNN hidden state')
    parser.add_argument('--latent_size', type=int, default=5,
                        help='size of latent space')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=21,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1.,
                        help='decay of learning rate')
    parser.add_argument('--chunk_samples', type=int, default=10,
                        help='number of samples per mdct chunk')
    parser.add_argument('--data_path', type=str, default='./',
                        help='number of samples per mdct chunk')

    args = parser.parse_args()

    model = VRNN(args)
    train(args, model)
