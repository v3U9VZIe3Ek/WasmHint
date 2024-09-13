import torch
import time
import numpy as np
from nltk.tokenize import word_tokenize
from torch.autograd import Variable


def to_np(x):
    return x.data.cpu().numpy()


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).contiguous().view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)

    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def load_glove_model(file_path):
    glove_model = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype='float32')
            glove_model[word] = vector
    return glove_model


def sentence_to_vec(sentence, glove_model, dim=100):
    # Tokenize the sentence
    tokens = word_tokenize(sentence.lower())
    embeds = [glove_model[token] for token in tokens if token in glove_model]

    if embeds:
        sent_embeds = np.mean(embeds, axis=0)
    else:
        # Handle the case where none of the words are in the GloVe model
        sent_embeds = np.zeros(dim)
    return sent_embeds


group2instrs = {
    'Parametric': ['drop', 'select'],
    'Local_var': ['local.get', 'local.set', 'local.tee'],
    'Global_var': ['global.get', 'global.set'],
    'Memory': ['i32.load', "i64.load","f32.load","f64.load","i32.load8_s","i32.load8_u","i32.load16_s","i32.load16_u","i64.load8_s","i64.load8_u","i64.load16_s","i64.load16_u","i64.load32_s","i64.load32_u","i32.store","i64.store","f32.store","f64.store","i32.store8","i32.store16","i64.store8","i64.store16","i64.store32","memory.size","memory.grow"],
    'Constant': ["i32.const", "i64.const", "f32.const", "f64.const"],
    'Logical_i32': ["i32.eqz","i32.eq","i32.ne", "i32.lt_s", "i32.lt_u","i32.gt_s","i32.gt_u","i32.le_s", "i32.le_u","i32.ge_s", "i32.ge_u"],
    'Logical_i64': ["i64.eqz","i64.eq", "i64.ne", "i64.lt_s","i64.lt_u", "i64.gt_s","i64.gt_u","i64.le_s", "i64.le_u","i64.ge_s","i64.ge_u"],
    'Logical_f32': ["f32.eq","f32.ne","f32.lt","f32.gt","f32.le","f32.ge"],
    'Logical_f64': ["f64.eq","f64.ne","f64.lt","f64.gt","f64.le","f64.ge"],
    'Arithmetic_i32': ["i32.clz","i32.ctz", "i32.popcnt", "i32.add", "i32.sub", "i32.mul","i32.div_s","i32.div_u","i32.rem_s", "i32.rem_u"],
    'Bitwise_i32': ["i32.and", "i32.or", "i32.xor", "i32.shl", "i32.shr_s", "i32.shr_u", "i32.rotl", "i32.rotr"],
    'Arithmetic_i64': ["i64.clz","i64.ctz", "i64.popcnt", "i64.add", "i64.sub", "i64.mul","i64.div_s","i64.div_u","i64.rem_s", "i64.rem_u"],
    'Bitwise_i64': ["i64.and", "i64.or", "i64.xor", "i64.shl", "i64.shr_s", "i64.shr_u", "i64.rotl", "i64.rotr"],
    'Arithmetic_f32': ["f32.abs", "f32.neg", "f32.ceil", "f32.floor", "f32.trunc", "f32.nearest", "f32.sqrt", "f32.add", "f32.sub", "f32.mul", "f32.div", "f32.min","f32.max","f32.copysign"],
    'Arithmetic_f64': ["f64.abs", "f64.neg", "f64.ceil", "f64.floor", "f64.trunc", "f64.nearest", "f64.sqrt", "f64.add", "f64.sub", "f64.mul", "f64.div", "f64.min","f64.max","f64.copysign"],
    'Conversion': ["i32.wrap_i64", "i32.trunc_f32_s", "i32.trunc_f32_u", "i32.trunc_f64_s","i32.trunc_f64_u", "i64.extend_i32_s",  "i64.extend_i32_u", "i64.trunc_f32_s", "i64.trunc_f32_u", "i64.trunc_f64_s", "i64.trunc_f64_u", "f32.convert_i32_s", "f32.convert_i32_u", "f32.convert_i64_s","f32.convert_i64_u", "f32.demote_f64", "f64.convert_i32_s", "f64.convert_i32_u", "f64.convert_i64_s", "f64.convert_i64_u","f64.promote_f32", "i32.reinterpret_f32", "i64.reinterpret_f64", "f32.reinterpret_i32", "f64.reinterpret_i64"],
    'Control': ["nop","block","loop","if","else","end", "br","br_if","br_table", "return", "call", "call_indirect"]
}

def calculate_instr_freq(bb_instrs):
    instr2group = {}
    for group, instrs in group2instrs.items():
        for instr in instrs:
            instr2group[instr] = group

    instr_freq = {
        # Instruction-based features
        'freq_parametric': 0,
        'freq_logical_i32': 0,
        'freq_logical_i64': 0,
        'freq_logical_f32': 0,
        'freq_logical_f64': 0,
        'freq_arithmetic_i32': 0,
        'freq_bitwise_i32': 0,
        'freq_arithmetic_i64': 0,
        'freq_bitwise_i64': 0,
        'freq_arithmetic_f32': 0,
        'freq_arithmetic_f64': 0,
        'freq_conversion': 0,
        'freq_unsupported': 0,

        # Memory-related instructions
        'freq_local_variable': 0,
        'freq_global_variable': 0,
        'freq_memory': 0,
        'freq_constant': 0,

        # func call
        'has_function_call': 0,
    }

    for instr in bb_instrs:
        if instr in instr2group:
            if instr2group[instr] == 'Parametric':
                instr_freq['freq_parametric'] += 1
            elif instr2group[instr] == 'Logical_i32':
                instr_freq['freq_logical_i32'] += 1
            elif instr2group[instr] == 'Logical_i64':
                instr_freq['freq_logical_i64'] += 1
            elif instr2group[instr] == 'Logical_f32':
                instr_freq['freq_logical_f32'] += 1
            elif instr2group[instr] == 'Logical_f64':
                instr_freq['freq_logical_f64'] += 1
            elif instr2group[instr] == 'Arithmetic_i32':
                instr_freq['freq_arithmetic_i32'] += 1
            elif instr2group[instr] == 'Bitwise_i32':
                instr_freq['freq_bitwise_i32'] += 1
            elif instr2group[instr] == 'Arithmetic_i64':
                instr_freq['freq_arithmetic_i64'] += 1
            elif instr2group[instr] == 'Bitwise_i64':
                instr_freq['freq_bitwise_i64'] += 1
            elif instr2group[instr] == 'Arithmetic_f32':
                instr_freq['freq_arithmetic_f32'] += 1
            elif instr2group[instr] == 'Arithmetic_f64':
                instr_freq['freq_arithmetic_f64'] += 1
            elif instr2group[instr] == 'Conversion':
                instr_freq['freq_conversion'] += 1
            # Memory-related instructions
            elif instr2group[instr] == 'Local_var':
                instr_freq['freq_local_variable'] += 1
            elif instr2group[instr] == 'Global_var':
                instr_freq['freq_global_variable'] += 1
            elif instr2group[instr] == 'Memory':
                instr_freq['freq_memory'] += 1
            elif instr2group[instr] == 'Constant':
                instr_freq['freq_constant'] += 1
            elif instr2group[instr] == 'Control':
                if instr == 'call' or instr == 'call_indirect':
                    instr_freq['has_function_call'] = 1
        else:
            instr_freq['freq_unsupported'] += 1

    return instr_freq


def convert2type_topk(output_ids, idx2ty):
    out_lists = []
    for per_output_ids in output_ids:
        per_out_list = []
        for ids in per_output_ids:
            tmp_out = ''
            for i in ids:
                token = idx2ty[i]
                if token == '<TY_SOS>':
                    continue
                elif token == '<TY_EOS>':
                    break
                else:
                    tmp_out += token + ' '
            tmp_out = tmp_out.strip()
            per_out_list.append(tmp_out)
        out_lists.append(per_out_list)
    return out_lists


def convert2type(output_ids, idx2ty):
    out_list = []
    for ids in output_ids:
        tmp_out = ''
        for i in ids:
            token = idx2ty[i]
            if token == '<TY_SOS>':
                continue
            elif token == '<TY_EOS>':
                break
            else:
                tmp_out += token + ' '
        tmp_out = tmp_out.strip()
        out_list.append(tmp_out)
    return out_list


def to_np(x):
    return x.data.cpu().numpy()


def trim_seqs_beam(seqs, ty2idx):
    trimmed_seqs = []
    for output_seq in seqs:
        trimmed_seq = []
        for per_seq in output_seq:
            per_trimmed_seq = []
            for idx in per_seq:
                per_trimmed_seq.append(idx)
                if idx == ty2idx['<TY_PAD>']:
                    break
            trimmed_seq.append(per_seq)

        trimmed_seqs.append(trimmed_seq)

    return trimmed_seqs