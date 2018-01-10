from __future__ import print_function, division
import os 

import numpy as np
import tensorflow as tf
from keras.backend import learning_phase

import conv_model
import sampler_utils
import data_utils


def train_model(model, dataset_path='top_dataset.h5', input_size=40, 
                batch_size=64, epochs=30, vol_coeff=1.0, 
                iter_sampler=sampler_utils.uniform_sampler(),
                summary_prefix='summary', save_prefix='trained_models'):
        
    # main graph
    input_data = tf.placeholder(tf.float32, (None, input_size, input_size, 2), name='input_data')
    output_true = tf.placeholder(tf.float32, (None, input_size, input_size, 1), name='output_true')
    learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
    
    with tf.variable_scope('topopt_model'):
        output_pred = model(input_data)
        
    # loss
    conf_loss = tf.losses.log_loss(output_true, output_pred, reduction=tf.losses.Reduction.MEAN)
    vol_loss = tf.square(tf.reduce_mean(output_true - output_pred))
    loss = conf_loss + vol_coeff * vol_loss
    
    # metrics
    correct_prediction = tf.equal(tf.round(output_true), tf.round(output_pred))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # train
    adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
    minimize_op = adam.minimize(loss)
    
    # summary
    tf.summary.scalar('conf_loss', conf_loss)
    tf.summary.scalar('vol_loss', vol_loss)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all() 
    
    summary_path = os.path.join(summary_prefix, 'VOL_COEFF={}'.format(vol_coeff))
    writer = tf.summary.FileWriter(summary_path, graph=tf.get_default_graph())
    
    # training_process
    sess = tf.Session()
    saver = tf.train.Saver()
    
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        
        print('Training...')
        global_step = 0
        
        for epoch in range(epochs):
            for x, y in data_utils.DatasetIterator(dataset_path, batch_size, iter_sampler):
                
                _, summary = sess.run([minimize_op, summary_op], 
                                      feed_dict={
                                          input_data: x, 
                                          output_true: y,
                                          learning_rate: 0.001 / (1 + epochs // 15),
                                          learning_phase(): 1
                                      })
                
                writer.add_summary(summary, global_step=global_step)
                global_step += 1
         
        print('Training is finished')
        save_path = os.path.join(save_prefix, 'VOL_COEFF={}'.format(vol_coeff))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saver.save(sess, save_path)
        print('Graph is saved: {}'.format(save_path))



if __name__ == '__main__':
    # parsing
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--dataset-path', type=str, dest='dataset_path', 
                        help='path to `.h5` dataset', required=True)

    parser.add_argument('--input-size', type=int, dest='input_size', 
                        help='size of the input tensor', default=40)

    parser.add_argument('--batch-size', type=int, dest='batch_size', 
                        help='size of a minibatch', default=64)

    parser.add_argument('--epochs', type=int, dest='epochs', 
                        help='number of training epochs', default=30)

    parser.add_argument('--vol-coeff', type=float, dest='vol_coeff', 
                        help='volume constraint coefficient in total loss', 
                        default=1.0)

    parser.add_argument('--iter-sampler', type=str, dest='iter_sampler', 
                        help='iteration sampler. Either "uniform" or "poisson_LAM"\n'
                        'LAM: Lambda parameter in Poisson distribution', 
                        default='uniform')

    parser.add_argument('--summary-prefix', type=str, dest='summary_prefix', 
                        help='root folder to save the summary', 
                        default='summary')

    parser.add_argument('--save-prefix', type=str, dest='save_prefix', 
                        help='root folder to save the model', 
                        default='trained_models')

    options = parser.parse_args()

    # training
    model = conv_model.build()
    
    train_model(model, 
                dataset_path=options.dataset_path, 
                input_size=options.input_size, 
                batch_size=options.batch_size, 
                epochs=options.epochs, 
                vol_coeff=options.vol_coeff, 
                iter_sampler=sampler_utils.parse_sampler(options.iter_sampler),
                summary_prefix=os.path.join(options.summary_prefix, options.iter_sampler), 
                save_prefix=os.path.join(options.save_prefix, options.iter_sampler))

