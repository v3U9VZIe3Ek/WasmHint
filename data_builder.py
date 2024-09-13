import os
import json
import pickle
from datetime import datetime
import networkx as nx
import logging

import numpy as np
from tqdm import tqdm
from opcodes import opcodes as OPCODES
from utils import sentence_to_vec

import torch
from transformers import AutoTokenizer, AutoModel


logger = logging.getLogger(__name__)


def build_networkx_graph(cfg_dict, is_print=False):

    def parse_instr(instr):  # (offset: opcode operands)
        return instr.strip().split()[1].replace('(', '').replace(')', '').strip()

    graph = nx.DiGraph()

    # Add nodes with attributes and edges to the graph
    for node, info in cfg_dict.items():
        node_id = int(node)

        if 'instructions' in info:
            graph.add_node(node_id, instrs=[parse_instr(instr) for instr in info['instructions']])
        else:
            graph.add_node(node_id)

        # Add edges from this node to its neighbors
        for neighbor in info['successors']:
            graph.add_edge(node_id, neighbor)

    if is_print:
        # Print out the nodes and their attributes
        for node in graph.nodes(data=True):  # The data=True parameter tells it to return node attributes
            print(f"Node {node[0]}: {node[1]}")  # node[0] is the node ID, node[1] are the attributes

        # Print all edges with their attributes
        for edge in graph.edges(data=True):
            # edge[0] --> edge[1] with edge[2] are the attributes
            print(f"Edge from {edge[0]} to {edge[1]}: {edge[2]}")

    return graph


def opcode2vec(glove_model):
    op2vec = {}
    for opcode, info in OPCODES.items():
        pops = info[1]
        pushs = info[2]
        desc = info[3]
        desc_vec = sentence_to_vec(desc, glove_model, dim=100)

        op2vec[opcode] = np.concatenate([np.array([pops, pushs]), desc_vec], axis=0)

    return op2vec


def build_token_idx2vec(idx2token, op2vec, glove_model):

    tk_idx2vec = {}
    for idx, token in idx2token.items():
        if token in op2vec:
            tk_idx2vec[idx] = op2vec[token]
        elif token == '<PAD>':
            tk_idx2vec[idx] = np.zeros(102)
        elif token in ['<SOS>', '<EOS>', '<UNK>', '<SEP>', '<NONE>']:
            tk_idx2vec[idx] = np.random.rand(102)
        else:
            tk_idx2vec[idx] = np.random.rand(102)

    tk_idx2vec['i32'] = np.concatenate((glove_model['i32'], np.zeros(2)), axis=0) \
        if 'i32' in glove_model else np.random.rand(102)

    tk_idx2vec['i64'] = np.concatenate((glove_model['i64'], np.zeros(2)), axis=0) \
        if 'i64' in glove_model else np.random.rand(102)

    tk_idx2vec['f32'] = np.concatenate((glove_model['f32'], np.zeros(2)), axis=0) \
        if 'f32' in glove_model else np.random.rand(102)

    tk_idx2vec['f64'] = np.concatenate((glove_model['f64'], np.zeros(2)), axis=0) \
        if 'f64' in glove_model else np.random.rand(102)

    return tk_idx2vec

def build_type_vocab(infer_type):
    vocabs = ['<TY_SOS>', '<TY_EOS>', '<TY_PAD>', '<TY_UNK>', '<TY_NONE>']
    with open(f'./vocab/type/vocab_{infer_type}_type.txt', 'r') as fr:
        for line in fr.readlines():
            ts, n = line.strip().split('\t')
            for t in ts.split():
                vocabs.append(t.strip())
    # assert len(vocabs) == len(list(set(vocabs)))

    vocabs = list(set(vocabs))

    idx2ty = {idx: ty for idx, ty in enumerate(vocabs)}
    ty2idx = {ty: idx for idx, ty in enumerate(vocabs)}

    return ty2idx, idx2ty


def build_token_vocab(token_vocab):
    token_vocab = ['<PAD>', '<SOS>', '<EOS>', '<UNK>', '<SEP>',
                   'i32', 'i64', 'f32', 'f64', '<NONE>'] + token_vocab
    idx2token = {idx: token for idx, token in enumerate(token_vocab)}
    token2idx = {token: idx for idx, token in enumerate(token_vocab)}

    return token2idx, idx2token


def build_const_vocab(const_vocab):
    vocabs = ['<PAD>']
    for v in const_vocab:
        if not v.startswith('$'):
            vocabs.append(v)

    assert len(vocabs) == len(list(set(vocabs)))

    idx2const = {idx: token for idx, token in enumerate(vocabs)}
    const2idx = {token: idx for idx, token in enumerate(vocabs)}

    return const2idx, idx2const


def build_path_vocab(path_vocab):
    assert len(path_vocab) == len(list(set(path_vocab)))

    idx2path = {idx: token for idx, token in enumerate(path_vocab)}
    path2idx = {token: idx for idx, token in enumerate(path_vocab)}

    return path2idx, idx2path


def build_idx2vocab(token_vocab, const_vocab, path_vocab, infer_type):
    ty2idx, idx2ty = build_type_vocab(infer_type)
    token2idx, idx2token = build_token_vocab(token_vocab)
    const2idx, idx2const = build_const_vocab(const_vocab)
    path2idx, idx2path = build_path_vocab(path_vocab)
    return ty2idx, idx2ty, token2idx, idx2token, const2idx, idx2const, path2idx, idx2path


def parse_stack(stack_item):
    sf_stack_item = stack_item[len('(offset: instr) ->'):].strip()
    # print(sf_stack_item)
    offset = sf_stack_item[: sf_stack_item.find(':')].strip()
    cur_instr = sf_stack_item[sf_stack_item.find(':') + 1: sf_stack_item.find(', value:')].strip()
    cur_opcode = cur_instr.strip().split()[0].replace('(', '').replace(')', '').strip()
    cur_value = sf_stack_item[sf_stack_item.find(', value:') + 8: sf_stack_item.find(', source:')].strip()
    cur_source = sf_stack_item[sf_stack_item.find(', source:') + 9:].strip()
    # 11: local.get 1, value: param_1, source: param_1
    # print(offset, '----', cur_opcode, '-----', value, '-----', source)
    return (offset, cur_opcode, cur_value, cur_source)


def parse_instr_with_stack(info):
    # print(info)
    # offset = info['offset']

    # parse the instruction
    instr = info['instr']
    # print(instr)
    const_val = ''
    # store_local_offset = ''
    opcode = instr.strip().split()[0].replace('(', '').replace(')', '').strip()
    # print(opcode)
    # print('====================')
    if 'const' in instr:
        const_val = instr.strip().split()[1].replace('(', '').replace(')', '').strip()

    # if 'store' in opcode and 'offset' in instr:
    #     store_local_offset = 'local_' + instr.strip().split()[1].replace('(', '').replace(')', '').strip()[len('offset='):]
    # parse the stack
    raw_stack = info['stack']
    stack = [parse_stack(stack_item) for stack_item in raw_stack]

    # if store_local_offset != '':
    #     return (opcode, stack, const_val), (store_local_offset, stack, const_val)
    # else:
    return (opcode, stack, const_val)


def merge_return_instrs(return_instrs):
    # # offset => [dict{offset, instr, stack([xxx])}, ...]
    merged_instrs = {}
    for idx, infos in return_instrs.items():
        for item in infos:
            offset = item['offset']
            if offset not in merged_instrs:
                merged_instrs[offset] = {
                    'instr': item['instr'],
                    'stack': set(item['stack'])
                }
            else:
                for it in item['stack']:
                    merged_instrs[offset]['stack'].add(it)
    for offset, item in merged_instrs.items():
        merged_instrs[offset]['stack'] = list(item['stack'])

    return merged_instrs


def build_param_data(wasm_param, param_idx2infos, func_high_param):

    if not wasm_param:
        return []

    if len(wasm_param) == 0:
        return []

    if not param_idx2infos:
        return []

    if len(wasm_param) != len(param_idx2infos):
        return []

    if len(wasm_param) < len(func_high_param):
        return []

    if len(func_high_param) > 5:
        return []

    if list(wasm_param.keys()) != list([key[len('param_'):].strip() for key in list(param_idx2infos.keys())]):
        return []

    items = []

    combined_param_num = 3
    if len(wasm_param) <= combined_param_num:  # combine 3 param-related sequences
        cur_item = []
        for idx, (param_idx, details) in enumerate(param_idx2infos.items()):
            # list of (opcode, stack, const_val)
            cur_item.append((wasm_param[str(idx)], details))
        for param_pos in range(len(func_high_param)):
            raw_label = func_high_param['Param ' + str(param_pos + 1)]
            items.append((param_pos, cur_item, raw_label))
    else:
        # print(param_idx2infos)
        for param_pos in range(len(func_high_param)):
            raw_label = func_high_param['Param ' + str(param_pos + 1)]
            start = max(param_pos, 0)
            end = min(param_pos + combined_param_num, len(wasm_param))
            if end - start < combined_param_num:
                start = max(end - combined_param_num, 0)
            # cur_wasm_param = wasm_param[start:end]
            cur_wasm_param = [wasm_param[str(param_idx)] for param_idx in range(start, end)]
            cur_param_info = [param_idx2infos['param_' + str(param_idx)] for param_idx in range(start, end)]
            assert len(cur_wasm_param) == len(cur_param_info)
            cur_item = []
            for i in range(len(cur_wasm_param)):
                cur_item.append((cur_wasm_param[i], cur_param_info[i]))
            items.append((param_pos, cur_item, raw_label))

    return items


def build_return_data(wasm_param, wasm_result, param_infos, return_infos, func_high_return):

    if len(return_infos) == 0:
        return None

    # (opcode, stack, const_val)
    if return_infos[-1][0].replace(')', '').strip() != 'return':
        return None

    if wasm_result and isinstance(wasm_result, list) and len(wasm_result) != 1:
        return None

    cur_item = []
    if not wasm_result or wasm_result == '' or wasm_result == 'None' or wasm_result == 'null':
        cur_item.append(('<NONE>', return_infos))
    else:
        if isinstance(wasm_result, list) and len(wasm_result) == 1:
            cur_item.append((wasm_result[0], return_infos))

    if len(param_infos) > 0 and wasm_param and "0" in wasm_param:
        cur_item.append((wasm_param["0"], param_infos))

    if len(cur_item) == 0:
        return None

    return ('', cur_item, func_high_return)


def load_param_path():
    param_paths = {}
    data_path = 'vocab/path_vocab_param.txt'
    with open(data_path, 'r') as fr:
        for line in fr.readlines():
            param_paths[line.strip()] = ''
    return param_paths


def build_data(base_path, infer_type):

    # each file is a dict,
    # func_id => {
    #     'func_body': func_body,
    #     'func_cfg': func_cfg,
    #     'wasm_param': idx2param,
    #     'wasm_result': wret,
    #     'tainted_instrs_param': param_related_instrs,
    #     'return_instrs': return_related_instrs,
    #     'func_dcmp': dcmp_func_body,
    #     'func_high_param': func_high_param,
    #     'func_high_return': func_high_return,
    #     'Path': func_info['Path'] # Save the path information for easy debugging
    # }

    all_data_items = list()
    token_vocab = list()
    const_vocab = list()
    path_vocab = list()

    start_time = datetime.now()
    # ########################### Cache the data ###########################
    raw_cached_path_prefix = 'cache'
    if not os.path.exists(raw_cached_path_prefix):
        os.makedirs(raw_cached_path_prefix)

    raw_cached_data_path = os.path.join(raw_cached_path_prefix, 'raw_cached_data_%s.pkl' % infer_type)

    if os.path.exists(raw_cached_data_path):
        with (open(raw_cached_data_path, 'rb') as fr):
            all_data_items = pickle.load(fr)

        with open(os.path.join('vocab', 'token_vocab_%s.txt' % infer_type), 'r') as fr:
            for line in fr.readlines():
                token_vocab.append(line.strip())

        with open(os.path.join('vocab', 'const_vocab_%s.txt' % infer_type), 'r') as fr:
            for line in fr.readlines():
                const_vocab.append(line.strip())

        with open(os.path.join('vocab', 'path_vocab_%s.txt' % infer_type), 'r') as fr:
            for line in fr.readlines():
                path_vocab.append(line.strip())

        end_time = datetime.now()
        print("Load the cached raw data with a total of %d functions for %s, which spent %s(s)" %
              (len(all_data_items), infer_type, str((end_time - start_time).seconds)))

        return all_data_items, token_vocab, const_vocab, path_vocab
    # ########################### Cache the data ###########################

    print("Build the data for %s ........" % infer_type)

    all_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.json'):
                all_files.append(file)

    # # TODO for test
    # all_files = all_files[:1]
    print("Total files: ", len(all_files))

    param_paths = {}
    if infer_type == 'return':
        param_paths = load_param_path()

    for file in all_files:
        file_path = os.path.join(base_path, file)
        with open(file_path, 'r') as fr:
            all_raw_data = json.load(fr)

        for func_name, details in tqdm(all_raw_data.items(), desc='Current file: %s' % file_path):
            # func_body = details['func_body']

            # Build cfg for each function
            func_cfg_json = details['func_cfg']  # CFG
            func_cfg = build_networkx_graph(func_cfg_json)

            # Using the dcmp content as a code sequence
            func_dcmp = details['func_dcmp']  # decompiled function body
            func_dcmp = func_dcmp.replace('\n', ' ')

            # Record the path information
            path = details['Path']
            func_ident = path + '_<FUNC_NAME>_' + func_name

            wasm_param = details['wasm_param']  # dict, param type in wasm
            wasm_result = details['wasm_result']  # str, return type in wasm

            # param_idx ==> [offset, instr, stack([xxx])]
            # xxx => (offset: instr) -> 11: local.get 1, value: param_1, source: param_1
            if infer_type == 'param':
                param_idx2infos = {}
                tainted_instrs_param = details['tainted_instrs_param'][0]  # param related instructions
                for param_idx, infos in tainted_instrs_param.items():
                    # print(param_idx, instrs)
                    param_idx2infos[param_idx] = []
                    for info in infos:
                        instr_opcode_stack = parse_instr_with_stack(info)
                        # if len(instr_opcode_stack) == 2:
                        #     param_idx2infos[param_idx].append(instr_opcode_stack[0])
                        #     param_idx2infos[param_idx].append(instr_opcode_stack[1])
                        #
                        #     # Vocab
                        #     token_vocab.append(instr_opcode_stack[0][0])
                        #     token_vocab.append(instr_opcode_stack[1][0])
                        #     const_vocab.append(instr_opcode_stack[0][2])
                        #     const_vocab.append(instr_opcode_stack[1][2])
                        # else:
                        param_idx2infos[param_idx].append(instr_opcode_stack)

                        # Vocab
                        token_vocab.append(instr_opcode_stack[0])
                        const_vocab.append(instr_opcode_stack[2])

                func_high_param = details['func_high_param']  # high level param
                param_items = build_param_data(wasm_param, param_idx2infos, func_high_param)

                for item in param_items:
                    if item[2] == 'closure':
                        continue
                    path_vocab.append(func_ident)
                    all_data_items.append({
                        'func_ident': func_ident,
                        'func_cfg': func_cfg,
                        'func_dcmp': func_dcmp,
                        'item': item,  # param_pos, cur_item ([opcode, stack, const_val]), raw_label
                    })

            elif infer_type == 'return':

                if func_ident not in param_paths:
                    continue

                return_infos = []
                # offset => [dict{offset, instr, stack([xxx])}, ...]
                return_instrs = details['return_instrs']  # return related instructions
                # Merge the return-related instructions
                merged_return_instrs = merge_return_instrs(return_instrs)
                for offset, item in merged_return_instrs.items():
                    # return_idx2infos[offset] = []
                    instr_opcode_stack = parse_instr_with_stack(item)
                    # if len(instr_opcode_stack) == 2:
                    #     return_idx2infos[offset].append(instr_opcode_stack[0])
                    #     return_idx2infos[offset].append(instr_opcode_stack[1])
                    #
                    #     # Vocab
                    #     token_vocab.append(instr_opcode_stack[0][0])
                    #     token_vocab.append(instr_opcode_stack[1][0])
                    #     const_vocab.append(instr_opcode_stack[0][2])
                    #     const_vocab.append(instr_opcode_stack[1][2])
                    # else:
                    return_infos.append(instr_opcode_stack)

                    # Vocab
                    token_vocab.append(instr_opcode_stack[0])
                    const_vocab.append(instr_opcode_stack[2])

                raw_first_param_infos = []
                tainted_instrs_param = details['tainted_instrs_param'][0]  # param related instructions
                if len(tainted_instrs_param.keys()) > 0:
                    raw_first_param_infos = tainted_instrs_param[list(tainted_instrs_param.keys())[0]]

                first_param_infos = []
                for info in raw_first_param_infos:
                    instr_opcode_stack = parse_instr_with_stack(info)
                    first_param_infos.append(instr_opcode_stack)
                    token_vocab.append(instr_opcode_stack[0])
                    const_vocab.append(instr_opcode_stack[2])

                func_high_return = details['func_high_return']  # high level return
                item = build_return_data(wasm_param, wasm_result, first_param_infos, return_infos, func_high_return)
                if item is not None:
                    path_vocab.append(func_ident)
                    all_data_items.append({
                        'func_ident': func_ident,
                        'func_cfg': func_cfg,
                        'func_dcmp': func_dcmp,
                        'item': item,
                    })

    end_time = datetime.now()

    # Cache the data
    with open(raw_cached_data_path, 'wb') as fw:
        pickle.dump(all_data_items, fw)

    token_vocab = list(set(token_vocab))
    with open(os.path.join('vocab', 'token_vocab_%s.txt' % infer_type), 'w') as fw:
        for token in token_vocab:
            fw.write(token + '\n')

    const_vocab = list(set(const_vocab))
    with open(os.path.join('vocab', 'const_vocab_%s.txt' % infer_type), 'w') as fw:
        for const in const_vocab:
            fw.write(const + '\n')

    path_vocab = list(set(path_vocab))
    with open(os.path.join('vocab', 'path_vocab_%s.txt' % infer_type), 'w') as fw:
        for path in path_vocab:
            fw.write(path + '\n')

    print('Save the data to %s for caching, with a total of %d functions for %s, which spent %s(s)' %
          (raw_cached_data_path, len(all_data_items), infer_type, str((end_time - start_time).seconds)))

    # 290577 functions for param, which spent 409(s)  -> after filtering: 141893 functions
    # 290577 functions for return, which spent 627(s)  -> after filtering: 69076 functions
    return all_data_items, token_vocab, const_vocab, path_vocab


# if __name__ == '__main__':
#     build_data(base_path='./dataset/', infer_type='return')
