from experimental_2.hyper_view_data_loader_dynamic import DataGenerator
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
import argparse
import os
from math import ceil
import numpy as np
import csv
from experimental_2.hyper_view_model_model_selector import SpatioMultiChannellModel
import matplotlib.pyplot as plt
import keras.backend as K
import pandas as pd

np.random.seed(1)
tf.random.set_seed(2)


parser = argparse.ArgumentParser(description='HyperView')

parser.add_argument('-m', '--model-type', default=3, type=int, metavar='MT', help='0: X,  1: Y, 2: Z,')
parser.add_argument('-c', '--channel-type', default=5, type=int, metavar='CT', help='0: X,  1: Y, 2: Z,')
parser.add_argument('--start-epoch', default=0, type=int, metavar='SE', help='start epoch (default: 0)')
parser.add_argument('--num-epochs', default=3, type=int, metavar='NE', help='number of epochs to train (default: 120)')
parser.add_argument('--num-workers', default=4, type=int, metavar='NW', help='number of workers in training (default: 8)')
parser.add_argument('-b','--batch-size', default=1, type=int, metavar='BS', help='number of batch size (default: 32)')
parser.add_argument('-w','--width', default=128, type=int, metavar='BS', help='number of widthxheight size (default: 32)')
parser.add_argument('-l','--learning-rate', default=0.01, type=float, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--weights-dir', default='None', type=str, help='Weight Directory (default: modeldir)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate the model (it requires the wights path to be given')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true', help='pretrained or not')
parser.add_argument('--cuda', default='all', type=str, help=' cuda devices (default: 0)')

parser.add_argument('--train-dir', default='train_data/train_data/', type=str, help='path to the data directory')
parser.add_argument('--label-dir', default='train_data/train_gt.csv', type=str, help='path to the data directory')
parser.add_argument('--eval-dir', default='test_data/', type=str, help='path to the data directory')


parser.add_argument('--out-dir', default='modeldir/', type=str, help='Out Directory (default: modeldir)')
parser.add_argument('--log-file', default='performance-logs.csv', type=str, help='path to log dir (default: logdir)')

args = parser.parse_args()


def main():
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    image_shape = (args.width, args.width)
    dataset = DataGenerator(args.train_dir, args.label_dir, args.eval_dir,
                            valid_size=0.20,
                            image_shape=image_shape,
                            batch_size=args.batch_size)

    experiment_log = '{}/m_{}_c_{}_b_{}_e_{}_lr_{}_p_{}_w_{}'.format(args.out_dir, args.model_type, args.channel_type,
                                                                     args.batch_size, 159,
                                                                     args.learning_rate, args.pretrained, args.width) #args.num_epochs,


    model = SpatioMultiChannellModel(args.model_type,args.channel_type, dataset.image_shape, dataset.label_shape, pretrained=args.pretrained)
    #train_model(model, dataset, experiment_log, warmup=True)
    model.load_weights('{}_model_best.h5'.format(experiment_log))
    train_model(model, dataset, experiment_log, warmup=False)
    model.load_weights('{}_model_best.h5'.format(experiment_log))
    evaluate_model(model, dataset)
    create_submission(model, dataset.eval_reader,experiment_log)


def train_model(model, dataset, log_args, warmup=True):
    #strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    #with strategy.scope():
        if warmup:
            print('\n\nWARM-UP SESSION STARTED!\n\n')
            #for idx in range(len(model.submodules)):
            #    if 'backbone_model' in model.submodules[idx].name:
            #        model.submodules[idx].trainable=False
            #        for idy in range(len(model.submodules[idx].layers)): model.submodules[idx].layers[idy].trainable = False

            learning_rate = args.learning_rate / 1
            num_epochs = ceil(args.num_epochs / 15)

        else:
            print('\n\nTRAINING SESSION STARTED!\n\n')
            for idx in range(len(model.submodules)):
                if 'backbone_model' in model.submodules[idx].name:
                    model.submodules[idx].trainable = True
                    for idy in range(len(model.submodules[idx].layers)): model.submodules[idx].layers[idy].trainable = True
            model.trainable=True
            learning_rate = args.learning_rate
            num_epochs = args.num_epochs


        #maximal_learning_rate=learning_rate*100
        #clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=learning_rate,
        #                                  maximal_learning_rate= maximal_learning_rate,
        #                                  scale_fn=lambda x: 1 / (2. ** (x - 1)),
        #                                  step_size=250
        #                                  )

        optimizer = Adam(learning_rate=learning_rate)
        #moving_avg_optimizer = tfa.optimizers.SWA(optimizer)


        mse_total = CustomMSE()
        mse0 = CustomMSE(idx=0)
        mse1 = CustomMSE(idx=1)
        mse2 = CustomMSE(idx=2)
        mse3 = CustomMSE(idx=3)
        # mse = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        # lossWeights = {"total": 1, "P": 0 / 1100, "K": 0 / 2500, "Mg": 0 / 2000, "pH": 0 / 3}

        losses = {"total": mse_total, "P": mse0,"K": mse1,"Mg": mse2,"pH": mse3}
        lossWeights = {"total": 1, "P": 0.0 , "K": 0.0 , "Mg": 0.0 , "pH": 0 }
        model.compile(optimizer=optimizer, loss=losses,loss_weights=lossWeights, run_eagerly=False)

        callbacks = [
                ReduceLROnPlateau(verbose=1, patience=6),
                EarlyStopping(patience=25),
                ModelCheckpoint(#update_weights=True,
                    filepath='{}_model_best.h5'.format(log_args),
                    monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True),
                ]

        history = model.fit(dataset.train_reader,
                                epochs=num_epochs,
                                workers=args.num_workers,
                                callbacks=callbacks,
                                use_multiprocessing=True,
                                shuffle=True,
                                validation_data=dataset.valid_reader)

        loss_log = '{}_total_loss.jpg'.format(log_args)
        print_history(history, 'loss', loss_log)
        loss_log = '{}_P_loss.jpg'.format(log_args)
        print_history(history, 'P_loss', loss_log)
        loss_log = '{}_K_loss.jpg'.format(log_args)
        print_history(history, 'K_loss', loss_log)
        loss_log = '{}_Mg_loss.jpg'.format(log_args)
        print_history(history, 'Mg_loss', loss_log)
        loss_log = '{}_pH_loss.jpg'.format(log_args)
        print_history(history, 'pH_loss', loss_log)

        #return model

def evaluate_model(model, generators, logging=True):

    print('\n\nEVALUATION SESSION STARTED!\n\n')
    tr_loss = challenge_eval(model,generators.train_reader)
    val_loss = challenge_eval(model,generators.evalid_reader)

    print('TOTAL LOSS:  Training: {}, Validation: {}, Test: {}'.format(tr_loss[0],val_loss[0],te_loss[0]))
    #tr_loss = model.evaluate(generators.train_reader)
    #val_loss = model.evaluate(generators.valid_reader)
    if logging:
        header = ['out_dir','m','c','b','e','l','p','wxh', 'train_loss', 'valid_loss', 'P','P_val','K','K_val', 'Mg','Mg_val','pH', 'pH_val']
        info = [args.out_dir, args.model_type,args.channel_type,args.batch_size,args.num_epochs,args.learning_rate,args.pretrained,args.width,
                tr_loss[0], val_loss[0], tr_loss[1], val_loss[1],tr_loss[2], val_loss[2], tr_loss[3], val_loss[3],tr_loss[4], val_loss[4]]
        if not os.path.exists(args.out_dir+'/'+args.log_file):
            with open(args.out_dir+'/'+args.log_file, 'w') as file:
                logger = csv.writer(file)
                logger.writerow(header)
                logger.writerow(info)
        else:
            with open(args.out_dir+'/'+args.log_file, 'a') as file:
                logger = csv.writer(file)
                logger.writerow(info)

def create_submission(model, reader,log_args):
    print('\n\nSUBMISSION SESSION STARTED!\n\n')
    predictions = []
    files = []
    for X, Y, file_name  in reader:
        y_pred = model.predict(X)
        y_pred = y_pred[0]
        #y_pred = np.concatenate(y_pred,axis=1)
        if len(predictions)==0:
            predictions=y_pred
            files=file_name.numpy()
        else:
            predictions=np.concatenate((predictions,y_pred),axis=0)
            files=np.concatenate((files,file_name.numpy()))

    predictions = predictions * np.array([325.0, 625.0, 400.0, 7.8])

    sample_index = np.expand_dims(np.array([int(os.path.basename(f.decode('utf-8')).replace(".npz", "")) for f in files]),1)
    predictions = np.concatenate((sample_index, predictions), axis=1)


    submission = pd.DataFrame(data=predictions, columns=['temp_index',"P", "K", "Mg", "pH"])
    submission=submission.sort_values(by='temp_index',ascending=True)
    submission=submission.drop(columns='temp_index')
    submission.to_csv('{}_submission.csv'.format(log_args), index_label="sample_index")

def challenge_eval(model, reader):
    predictions = []
    ground_truth = []
    y_base = np.array([121764.2 / 1731.0, 394876.1 / 1731.0, 275875.1 / 1731.0, 11747.67 / 1731.0]) /np.array([325.0, 625.0, 400.0, 7.8])
    for X, Y  in reader:

        y_pred = model.predict(X)
        y_pred = y_pred[0]
        #y_pred = np.concatenate(y_pred,axis=1)
        if len(predictions)==0:
            predictions = y_pred
            ground_truth=Y.numpy()
        else:
            ground_truth =np.concatenate((ground_truth,Y.numpy()),axis=0)
            predictions=np.concatenate((predictions,y_pred),axis=0)


    mse = np.mean((ground_truth - predictions) ** 2, axis=0)
    mse_b = np.mean((ground_truth-y_base) ** 2, axis=0)
    scores = mse / mse_b #np.array([1100.0, 2500.0, 2000.0, 3.0])
    # Calculate the final score
    final_score = np.mean(scores)

    return np.concatenate((np.array([final_score]),scores),axis=0)



def print_history(history, type, file_name):
    fig = plt.figure()
    plt.plot(history.history['{}'.format(type)])
    plt.plot(history.history['val_{}'.format(type)])
    plt.title('model {}'.format(type))
    plt.ylabel('{}'.format(type))
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.grid(True)
    fig.savefig(file_name, dpi=fig.dpi)


class CustomMSE(tf.keras.losses.Loss):

    def __init__(self, idx=None):
        super().__init__()
        self.idx = idx
        y_base_fact = np.array([121764.2 / 1731.0, 394876.1 / 1731.0, 275875.1 / 1731.0, 11747.67 / 1731.0]) / np.array([325.0, 625.0, 400.0, 7.8])
        self.y_base = tf.constant(y_base_fact, dtype=tf.float32)
        if self.idx is not None:
            self.y_base = self.y_base[self.idx]

    def call(self, y_true, y_pred):
        if self.idx is not None:
            y_true = y_true[:, self.idx]

        loss_raw = K.mean(K.square(y_true - y_pred), 0)
        loss_base = K.mean(K.square(y_true - self.y_base), 0)
        loss = tf.math.divide(loss_raw, loss_base)
        return loss





if __name__ == '__main__':
    main()