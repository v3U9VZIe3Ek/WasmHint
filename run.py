# !/usr/bin/env python
# encoding: utf-8

import os
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import logging
from datetime import datetime
from data_loader import load_datasets_and_vocabs
from trainer import trainer
from model import WasmTypeRestorer


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2024,  # 42
                        help='random seed for initialization')

    # ################ Fixed Parameters ###########################
    parser.add_argument('--base_dataset_path', type=str, default='./dataset',
                        help='The base dataset path')

    parser.add_argument('--cached_dataset_path', type=str, default='./cache',
                        help='The cached dataset path')

    parser.add_argument('--result_path', type=str, default='./results',
                        help='The result path')

    parser.add_argument('--model_save_path', type=str, default='./model',
                        help='Path to models.')

    parser.add_argument('--max_label_length', type=int, default=7,
                        help='max number of output sequences')

    parser.add_argument('--teacher_forcing_fraction', type=float, default=0.5,
                        help='fraction of batches that will use teacher forcing during training')

    parser.add_argument('--scheduled_teacher_forcing', type=bool, default=True,
                        # action='store_true',
                        help='Linearly decrease the teacher forcing fraction '
                             'from 1.0 to 0.0 over the specified number of epochs')

    parser.add_argument('--save_interval', type=int, default=50,  # 50 for classification,
                        help='Save every ... update steps')

    parser.add_argument('--use_reinforcement_learning', type=bool, default=False,
                        # action='store_true',
                        help='Control whether use the reinforcement learning')

    parser.add_argument('--reward_type', type=str, default='accuracy',
                        # action='store_true',
                        help='Using Accuracy as the reward function')

    # ################ Fixed Parameters ###########################

    # ################ Hyper-parameters ###########################

    parser.add_argument('--loss', type=str, default='single',
                        help='Prediction type of the model', choices=['single', 'double'])

    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for training.")

    parser.add_argument("--test_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for testing.")

    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument('--epochs', type=int, default=50,
                        help='Total number of training epochs to perform')

    parser.add_argument('--infer_type', type=str, default="param",
                        help='Prediction type of the model', choices=['param', 'return'])

    parser.add_argument('--module', type=str, default="classification",
                        help='The current module of the model', choices=['classification', 'generation'])

    parser.add_argument('--with_stack', type=bool, default=True,
                        help='Whether to use stack')

    parser.add_argument('--input_stack_dim', type=int, default=101,
                        help='The init dimension of stack vectors')

    parser.add_argument('--with_cfg', type=bool, default=True,
                        help='Whether to use cfg')

    parser.add_argument('--with_dcmp', type=bool, default=False,
                        help='Whether to use dcmp')

    parser.add_argument('--dcmp_dim', type=int, default=768,
                        help='The init dimension of dcmp vectors')

    parser.add_argument('--with_consts', type=bool, default=True,
                        help='Whether to use consts')

    parser.add_argument('--seq2seq_model', type=str, default="wasmhint",
                        help='Prediction model', choices=['wasmhint'])

    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='The max length of input sequences')

    parser.add_argument('--input_node_dim', type=int, default=126,
                        help='Dimension of node embeddings in cfg')

    parser.add_argument('--input_dim', type=int, default=102,
                        help='Dimension of embeddings')

    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Dimension of hidden layers')

    parser.add_argument('--transformer_n_layers', type=int, default=6,
                        help='Number of Transformer encoder layers')

    parser.add_argument('--transformer_n_heads', type=int, default=8,
                        help='Number of Heads in Transformers')

    parser.add_argument('--beam_width', type=int, default=10,
                        help='Max width of each beam')

    parser.add_argument('--test_only', type=bool, default=False,
                        help='Top k output')

    parser.add_argument('--topk', type=int, default=5,
                        help='Top k output')

    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout rate')

    # parser.add_argument("--max_grad_norm", default=1.0, type=float,
    #                     help="Max gradient norm.")
    #
    # parser.add_argument('--keep_prob', type=float, default=1.0,
    #                     help='Probablity of keeping an element in the dropout step.')

    # ################ Hyper-parameters ###########################

    return parser.parse_args()


def check_args(args):
    logger.info(vars(args))


def main():
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # Parse args
    args = parse_args()

    # Setup CUDA, GPU training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logger.info('Device is %s', args.device)

    # Set seed
    set_seed(args)

    # Print args
    check_args(args)

    # Load datasets
    start_time = datetime.now()

    train_dataset, test_dataset, token_idx2vec, ty2idx, idx2ty, token2idx, idx2token, const2idx, idx2const, path2idx, idx2path = \
        load_datasets_and_vocabs(args)

    end_time = datetime.now()
    logger.info('Load dataset spends %s' % str((end_time - start_time).seconds))

    # Build Model
    model = WasmTypeRestorer(token_idx2vec, token2idx, ty2idx, const2idx, args)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(args.device)

    logger.info('Build model successfully! ')

    if args.scheduled_teacher_forcing:
        schedule = np.arange(1.0, 0.0, -1.0 / args.epochs)
    else:
        schedule = np.ones(args.epochs) * args.teacher_forcing_fraction

    # Train
    trainer(args, model, train_dataset, test_dataset, token2idx, idx2token, ty2idx, idx2ty, idx2path, schedule)
    print('Done! ')

    # base_path = os.path.join(args.result_path, args.language, args.module)
    # if not os.path.exists(base_path):
    #     os.makedirs(base_path)

    # save_path = os.path.join(base_path, 'results.pkl')
    #
    # save_data = (indicators, internal_time / args.test_dataset_length, args.test_dataset_length)
    #
    # save_or_append_pkl(save_path, save_data)

    # print('Save results successfully! ')


if __name__ == '__main__':
    main()

