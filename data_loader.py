# !/usr/bin/env python
# encoding: utf-8

import os
from tqdm import tqdm
import numpy as np
import pickle
import logging
import torch
import random
from torch.utils.data import Dataset
from data_builder import build_data, build_idx2vocab, opcode2vec, build_token_idx2vec
from torch_geometric.data import Data
import networkx as nx
from collections import Counter
from utils import load_glove_model, calculate_instr_freq, sentence_to_vec
import nltk
from dcmp_vec import run_dcmp_vec, load_embeddings


nltk.download('punkt')


logger = logging.getLogger(__name__)


def load_datasets_and_vocabs(args):
    # ##############################  Cache Data ########################################

    cached_path_prefix = args.cached_dataset_path
    if not os.path.exists(cached_path_prefix):
        os.makedirs(cached_path_prefix)

    cached_data_path = os.path.join(cached_path_prefix, 'cached_data_%s_%s_ml_%s.pkl' %
                                    (args.infer_type, args.module, str(args.max_seq_length)))

    if os.path.exists(cached_data_path):
        with (open(cached_data_path, 'rb') as fr):
            train_dataset, test_dataset, token_idx2vec, ty2idx, idx2ty, token2idx, idx2token, const2idx, idx2const, path2idx, idx2path = pickle.load(fr)

            logger.info('Load data from cache file %s. The number of train/test dataset is %d/%d'
                        % (cached_data_path, len(train_dataset), len(test_dataset)))

        return train_dataset, test_dataset, token_idx2vec, ty2idx, idx2ty, token2idx, idx2token, const2idx, idx2const, path2idx, idx2path
    # ##############################  Cache Data ########################################

    # If not cached, build dataset from scratch
    dataset_path = args.base_dataset_path
    dataset, token_vocab, const_vocab, path_vocab = build_data(dataset_path, args.infer_type)

    ty2idx, idx2ty, token2idx, idx2token, const2idx, idx2const, path2idx, idx2path = build_idx2vocab(
        token_vocab, const_vocab, path_vocab, args.infer_type)

    # Build vectorized data
    glove_model = load_glove_model('embeds/glove.6B.100d.txt')
    print("Load Glove model successfully!")
    args.glove_model = glove_model

    op2vec = opcode2vec(glove_model)

    # print(len(op2vec))
    token_idx2vec = build_token_idx2vec(idx2token, op2vec, glove_model)

    # Build dcmp vectors
    saved_path = f'embeds/dcmp_vecs_{args.infer_type}.pkl'
    if not os.path.exists(saved_path):
        cached_file_path = f'cache/raw_cached_data_%s.pkl' % args.infer_type
        run_dcmp_vec(cached_file_path, path2idx, saved_path)

    func_id2dcmp_vec = load_embeddings(saved_path)

    # # TODO !!! For local testing
    # dataset = dataset[:1000]

    # shuffle the dataset
    random.shuffle(dataset)

    # split train and test dataset
    total_cnt = len(dataset)
    split_cnt = (total_cnt // 5) * 4
    train = dataset[:split_cnt]
    random.shuffle(train)

    test = dataset[split_cnt:]

    train_dataset = WASMDataset(train, func_id2dcmp_vec, token2idx, ty2idx, const2idx, path2idx, args)
    test_dataset = WASMDataset(test, func_id2dcmp_vec, token2idx, ty2idx, const2idx, path2idx, args)

    logger.info('Build Train/Test finished! The total number is %d/%d' % (len(train_dataset), len(test_dataset)))

    # ##############################  Cache Data ########################################
    with open(cached_data_path, 'wb') as fw:
        pickle.dump((train_dataset, test_dataset, token_idx2vec, ty2idx, idx2ty,
                     token2idx, idx2token, const2idx, idx2const, path2idx, idx2path), fw)

    logger.info('Save data to caches successfully. The total number of train/test dataset is %d/%d' %
                (len(train_dataset), len(test_dataset)))
    # ##############################  Cache Data ########################################

    # param: 113512/28381   30063/7611
    # return: 55260/13816   843/226
    # new statistics

    # Param: 74896/18893 -- 30124/7365
    # Return: 52316/13090 -- 812/195
    return train_dataset, test_dataset, token_idx2vec, ty2idx, idx2ty, token2idx, idx2token, const2idx, idx2const, path2idx, idx2path


class WASMDataset(Dataset):
    def __init__(self, dataset, func_id2dcmp_vec, token2idx, ty2idx, const2idx, path2idx, args):
        self.args = args

        self.dataset = dataset
        self.func_id2dcmp_vec = func_id2dcmp_vec

        self.token2idx = token2idx
        self.ty2idx = ty2idx
        self.const2idx = const2idx
        self.path2idx = path2idx

        self.max_seq_length = args.max_seq_length
        self.max_label_length = args.max_label_length

        self.glove_model = args.glove_model

        self.vectorize_data = self._convert_features()

    def __len__(self):
        return len(self.vectorize_data)

    def __getitem__(self, idx):
        item = self.vectorize_data[idx]

        instr_feature = torch.from_numpy(item['instrs']).to(torch.long)
        stack_feature = torch.from_numpy(item['stacks']).to(torch.float)
        const_feature = torch.from_numpy(item['consts']).to(torch.long)
        graph = item['cfg']
        dcmp_feature = torch.from_numpy(item['dcmp']).to(torch.float)
        label = torch.tensor(item['label'], dtype=torch.long)
        func_ident = torch.tensor(item['func_ident'], dtype=torch.long)

        return instr_feature, stack_feature, const_feature, dcmp_feature, graph, label, func_ident

    def _convert_features(self):
        vectorize_data = []
        for item in tqdm(self.dataset, desc='Convert data to features'):
            # convert the data to model input format
            single_inst = self._convert_single_instance(item)
            if single_inst is not None:
                vectorize_data.append(single_inst)

        print('Convert data to features successfully! A total of %d instances' % len(vectorize_data))
        return vectorize_data

    def _convert_single_instance(self, item):

        # deal with cfg
        func_cfg = item['func_cfg']
        graph = self._proc_cfg(func_cfg)

        # deal with dcmp using CodeBERT
        # func_dcmp = item['func_dcmp']
        dcmps = self.func_id2dcmp_vec[self.path2idx[item['func_ident']]]  # 768-dim vector

        instr_stack_ty_info = item['item']

        # deal with wasm_type and instrs, combine them together
        # Similarly, deal with the stack and type label
        instrs, stacks, consts, label = self._proc_param_return(instr_stack_ty_info, infer_type=self.args.infer_type)

        # print(instrs.shape)
        # print(consts.shape)
        # print(label)
        # print(label.shape)
        # print(stacks.shape)
        #  'return_instrs': return_idx2infos,
        # 'param_instrs': param_idx2infos,  # dict, param_idx => [opcode, stack, const_val]

        if self.args.module == 'generation':
            # Remove the samples without multiple labels
            temp_label = label[:3]
            if temp_label[-1] == self.ty2idx['<TY_EOS>']:
                return None
            else:
                return {
                    'instrs': instrs,
                    'stacks': stacks,
                    'consts': consts,
                    'cfg': graph,
                    'dcmp': dcmps,
                    'label': label,
                    'func_ident': self.path2idx[item['func_ident']],
                }
        else:
            if label[2] == self.ty2idx['<TY_EOS>']:
                return {
                    'instrs': instrs,
                    'stacks': stacks,
                    'consts': consts,
                    'cfg': graph,
                    'dcmp': dcmps,
                    'label': label,
                    'func_ident': self.path2idx[item['func_ident']],
                }

    def _proc_cfg(self, func_cfg):
        # Features to represent a node (basic block)
        # 1. Statistical features: (1) Number of instructions
        # 2. Instruction-based features: (1) Frequency of each instruction type based on the groups in opcodes.py (13)
        #                                (2) Contain function calls? (1)
        # 3. Control Flow features: Type of terminator instruction  (7)
        # 4. Data Flow features: (1) Number of defined constants; (2) Number of used local variables;
        #                        (3) Number of used global variables;
        #                        (4) Number of used memory-related variables (load/store)
        # 5. Semantic features: Using Doc2Vec for the instruction sequence
        # print(func_cfg)

        # Node --> 126-dim vector
        edge_list = torch.tensor(list(func_cfg.edges), dtype=torch.long).t().contiguous()
        node_features = torch.tensor([self._parse_basic_block(node[1]['instrs'])
                                      for node in func_cfg.nodes(data=True)], dtype=torch.float)

        g = Data(x=node_features, edge_index=edge_list)
        return g

    def _parse_basic_block(self, bb_instrs):

        semantic_features = sentence_to_vec(' '.join(bb_instrs), self.glove_model, dim=100)

        instr_freqs = calculate_instr_freq(bb_instrs)

        # Initialize features
        features = {
            # Statistical features
            'num_instrs': len(bb_instrs),

            # Instruction-based features
            'freq_parametric': instr_freqs['freq_parametric'],
            'freq_logical_i32': instr_freqs['freq_logical_i32'],
            'freq_logical_i64': instr_freqs['freq_logical_i64'],
            'freq_logical_f32': instr_freqs['freq_logical_f32'],
            'freq_logical_f64': instr_freqs['freq_logical_f64'],
            'freq_arithmetic_i32': instr_freqs['freq_arithmetic_i32'],
            'freq_bitwise_i32': instr_freqs['freq_bitwise_i32'],
            'freq_arithmetic_i64': instr_freqs['freq_arithmetic_i64'],
            'freq_bitwise_i64': instr_freqs['freq_bitwise_i64'],
            'freq_arithmetic_f32': instr_freqs['freq_arithmetic_f32'],
            'freq_arithmetic_f64': instr_freqs['freq_arithmetic_f64'],
            'freq_conversion': instr_freqs['freq_conversion'],
            'freq_unsupported': instr_freqs['freq_unsupported'],

            # Memory-related instructions
            'freq_local_variable': instr_freqs['freq_local_variable'],
            'freq_global_variable': instr_freqs['freq_global_variable'],
            'freq_memory': instr_freqs['freq_memory'],
            'freq_constant': instr_freqs['freq_constant'],

            'has_function_call': instr_freqs['has_function_call'],

            # Control flow features: block, loop, if, br, br_if, return, unreachable
            'is_end_with_block': 1 if len(bb_instrs) > 0 and bb_instrs[-1] == 'block' else 0,
            'is_end_with_loop': 1 if len(bb_instrs) > 0 and bb_instrs[-1] == 'loop' else 0,
            'is_end_with_if': 1 if len(bb_instrs) > 0 and bb_instrs[-1] == 'if' else 0,
            'is_end_with_br': 1 if len(bb_instrs) > 0 and bb_instrs[-1] == 'br' else 0,
            'is_end_with_br_if': 1 if len(bb_instrs) > 0 and bb_instrs[-1] == 'br_if' else 0,
            'is_end_with_return': 1 if len(bb_instrs) > 0 and bb_instrs[-1] == 'return' else 0,
            'is_end_with_unreachable': 1 if len(bb_instrs) > 0 and bb_instrs[-1] == 'unreachable' else 0,

            # Semantic features
            'semantic_features': semantic_features
        }

        vector = [
            # Statistical features
            features['num_instrs'],

            # Instruction-based features
            features['freq_parametric'],
            features['freq_logical_i32'],
            features['freq_logical_i64'],
            features['freq_logical_f32'],
            features['freq_logical_f64'],
            features['freq_arithmetic_i32'],
            features['freq_bitwise_i32'],
            features['freq_arithmetic_i64'],
            features['freq_bitwise_i64'],
            features['freq_arithmetic_f32'],
            features['freq_arithmetic_f64'],
            features['freq_conversion'],
            features['freq_unsupported'],

            # Memory-related instructions
            features['freq_local_variable'],
            features['freq_global_variable'],
            features['freq_memory'],
            features['freq_constant'],

            features['has_function_call'],

            # Control flow features
            features['is_end_with_block'],
            features['is_end_with_loop'],
            features['is_end_with_if'],
            features['is_end_with_br'],
            features['is_end_with_br_if'],
            features['is_end_with_return'],
            features['is_end_with_unreachable'],

        ]

        vector.extend(features['semantic_features'])
        vector = np.array(vector)

        return vector

    def _proc_param_return(self, instr_stack_ty_info, infer_type):
        # (param_pos, cur_item, raw_label)
        # print(instr_stack_ty_info)

        combined_num = 3 if infer_type == 'param' else 2

        pos, cur_item, raw_label = instr_stack_ty_info

        instr_features = []
        stack_features = []
        consts = []

        if infer_type == 'param':
            for it in cur_item:
                cur = []
                cur.append(self.token2idx['<SOS>'])
                cur_stack = []
                cur_stack.append(np.zeros(101))

                wasm_ty, ops = it[0], it[1]

                # if not wasm_ty:
                #     wasm_ty = 'None'
                if wasm_ty == 'None' or wasm_ty == '' or wasm_ty == '<NONE>':
                    cur.append(self.token2idx['<NONE>'])
                else:
                    cur.append(self.token2idx[wasm_ty])
                cur_stack.append(np.zeros(101))

                cur.append(self.token2idx['<SEP>'])
                cur_stack.append(np.zeros(101))

                for op_info in ops:
                    op = op_info[0]
                    stacks = op_info[1]
                    const = op_info[2]
                    if op in self.token2idx:
                        cur.append(self.token2idx[op])
                    else:
                        cur.append(self.token2idx['<UNK>'])

                    # deal with stack
                    cur_stack_vec = self._proc_stack(stacks)  # 101-dim vector
                    # print(stack)
                    cur_stack.append(cur_stack_vec)

                    if const and const in self.const2idx:
                        consts.append(self.const2idx[const])

                if len(cur) >= self.args.max_seq_length - 1:
                    cur = cur[:self.args.max_seq_length - 1]
                    cur.append(self.token2idx['<EOS>'])
                    cur_stack = cur_stack[:self.args.max_seq_length - 1]
                    cur_stack.append(np.zeros(101))
                else:
                    cur.append(self.token2idx['<EOS>'])
                    cur = cur + [self.token2idx['<PAD>']] * (self.args.max_seq_length - len(cur))
                    cur_stack.append(np.zeros(101))
                    # cur_stack + [np.zeros(101)] * (self.args.max_seq_length - len(cur_stack))
                    cur_stack.extend([np.zeros(101) for _ in range(self.args.max_seq_length - len(cur_stack))])

                instr_features.append(cur)
                stack_features.append(cur_stack)
        else:  # For return, reverse the order
            cur_it = cur_item[0]  # The first one is the return-related instruction
            cur = []
            cur.append(self.token2idx['<SOS>'])
            cur_stack = []
            cur_stack.append(np.zeros(101))

            wasm_ty, ops = cur_it[0], cur_it[1]

            if wasm_ty == 'None' or wasm_ty == '' or wasm_ty == '<NONE>':
                cur.append(self.token2idx['<NONE>'])
            else:
                cur.append(self.token2idx[wasm_ty])
            cur_stack.append(np.zeros(101))

            cur.append(self.token2idx['<SEP>'])
            cur_stack.append(np.zeros(101))

            temp_cur = []
            temp_cur_stack = []
            for op_info in reversed(ops):
                op = op_info[0]
                stacks = op_info[1]
                const = op_info[2]
                if op in self.token2idx:
                    temp_cur.append(self.token2idx[op])
                else:
                    temp_cur.append(self.token2idx['<UNK>'])

                # deal with stack
                cur_stack_vec = self._proc_stack(stacks)  # 101-dim vector
                # print(stack)
                temp_cur_stack.append(cur_stack_vec)

                if const and const in self.const2idx:
                    consts.append(self.const2idx[const])

            cur.extend(reversed(temp_cur))
            cur_stack.extend(reversed(temp_cur_stack))

            if len(cur) >= self.args.max_seq_length - 1:
                # print('Large Large Large Large Large Large Large Large Large ')
                cur = cur[:self.args.max_seq_length - 1]
                cur.append(self.token2idx['<EOS>'])
                cur_stack = cur_stack[:self.args.max_seq_length - 1]
                cur_stack.append(np.zeros(101))
            else:
                # print('Less Less Less Less Less Less Less Less ')
                cur.append(self.token2idx['<EOS>'])
                cur = cur + [self.token2idx['<PAD>']] * (self.args.max_seq_length - len(cur))
                cur_stack.append(np.zeros(101))
                # cur_stack + [np.zeros(101)] * (self.args.max_seq_length - len(cur_stack))
                cur_stack.extend([np.zeros(101) for _ in range(self.args.max_seq_length - len(cur_stack))])

            instr_features.append(cur)
            stack_features.append(cur_stack)

            if len(cur_item) == 2:
                cur_it = cur_item[1]

                cur = []
                cur.append(self.token2idx['<SOS>'])
                cur_stack = []
                cur_stack.append(np.zeros(101))

                wasm_ty, ops = cur_it[0], cur_it[1]

                # if not wasm_ty:
                #     wasm_ty = 'None'
                if wasm_ty == 'None' or wasm_ty == '' or wasm_ty == '<NONE>':
                    cur.append(self.token2idx['<NONE>'])
                else:
                    cur.append(self.token2idx[wasm_ty])
                cur_stack.append(np.zeros(101))

                cur.append(self.token2idx['<SEP>'])
                cur_stack.append(np.zeros(101))

                for op_info in ops:
                    op = op_info[0]
                    stacks = op_info[1]
                    const = op_info[2]
                    if op in self.token2idx:
                        cur.append(self.token2idx[op])
                    else:
                        cur.append(self.token2idx['<UNK>'])

                    # deal with stack
                    cur_stack_vec = self._proc_stack(stacks)  # 101-dim vector
                    # print(stack)
                    cur_stack.append(cur_stack_vec)

                    if const and const in self.const2idx:
                        consts.append(self.const2idx[const])

                if len(cur) >= self.args.max_seq_length - 1:
                    # print('Large Large Large Large Large Large Large Large Large ')
                    cur = cur[:self.args.max_seq_length - 1]
                    cur.append(self.token2idx['<EOS>'])
                    cur_stack = cur_stack[:self.args.max_seq_length - 1]
                    cur_stack.append(np.zeros(101))
                else:
                    # print('Less Less Less Less Less Less Less Less ')
                    cur.append(self.token2idx['<EOS>'])
                    cur = cur + [self.token2idx['<PAD>']] * (self.args.max_seq_length - len(cur))
                    cur_stack.append(np.zeros(101))
                    # cur_stack + [np.zeros(101)] * (self.args.max_seq_length - len(cur_stack))
                    cur_stack.extend([np.zeros(101) for _ in range(self.args.max_seq_length - len(cur_stack))])

                instr_features.append(cur)
                stack_features.append(cur_stack)

        if len(consts) > self.args.max_seq_length:
            consts = consts[:self.args.max_seq_length]
        else:
            consts = consts + [self.const2idx['<PAD>']] * (self.args.max_seq_length - len(consts))

        label = self._proc_label(raw_label)

        while len(instr_features) < combined_num:
            instr_features = instr_features + [[self.token2idx['<PAD>']] * self.args.max_seq_length]
            # stack_features = stack_features + [[np.zeros(101)] * self.args.max_seq_length]
            stack_features.append([np.zeros(101) for _ in range(self.max_seq_length)])
        #
        # for item in stack_features:
        #     print(np.array(item).shape)
        return np.array(instr_features), np.array(stack_features), np.array(consts), np.array(label)

    def _proc_label(self, raw_label):
        # '<TY_SOS>', '<TY_EOS>', '<TY_PAD>', '<TY_UNK>', '<NONE>'
        if raw_label == '' or raw_label == 'NONE':
            labels = [self.ty2idx['<TY_SOS>'], self.ty2idx['<TY_NONE>'], self.ty2idx['<TY_EOS>']]
            labels = labels + [self.ty2idx['<TY_PAD>']] * (self.max_label_length - len(labels))
            return labels

        raw_label = raw_label.strip().split()
        labels = [self.ty2idx['<TY_SOS>']]
        for lb in raw_label:
            # print(lb)
            if lb in self.ty2idx:
                labels.append(self.ty2idx[lb])
            else:
                raise NotImplementedError

        if len(labels) >= self.max_label_length - 1:
            labels = labels[:self.max_label_length - 1]
            labels.append(self.ty2idx['<TY_EOS>'])
        else:
            labels.append(self.ty2idx['<TY_EOS>'])
            labels = labels + [self.ty2idx['<TY_PAD>']] * (self.max_label_length - len(labels))

        return labels

    def _proc_stack(self, stack_info):
        # Current stack size
        # Semantic features: treat each stack element as a sequence
        # (offset, cur_opcode, cur_value, cur_source)
        if len(stack_info) == 0:
            return np.zeros(101)
        stack_vecs = []
        for stack in stack_info:
            str_stack = ' '.join(stack)
            # print(stack)
            stack_vec = sentence_to_vec(str_stack, self.glove_model, dim=100)
            stack_vecs.append(stack_vec)

        stack_vecs = np.array(stack_vecs).mean(axis=0)
        stack_vecs = np.concatenate([np.array([len(stack_info)]), stack_vecs], axis=0)

        # offset, opcode, const_val = stack_info
        return stack_vecs
