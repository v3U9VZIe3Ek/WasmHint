import logging
import operator
from queue import PriorityQueue
import torch
from torch.autograd import Variable
import math
from abc import ABC
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import timeit, to_one_hot, to_np
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool


# from utils import timeit, to_np, to_one_hot, generate_square_subsequent_mask, create_padding_mask


logger = logging.getLogger(__name__)


class WasmTypeRestorer(nn.Module):
    def __init__(self, token_idx2vec, token2idx, ty2idx, const2idx, args):
        super(WasmTypeRestorer, self).__init__()

        self.token_idx2vec = token_idx2vec
        self.token2idx = token2idx
        self.ty2idx = ty2idx
        self.const2idx = const2idx
        self.args = args
        self.module = self.args.module

        # print(ty2idx)

        self.embed_dim = self.args.input_dim
        self.hidden_size = self.args.hidden_size
        self.max_label_length = self.args.max_label_length

        # print(token_idx2vec.shape)

        # Initialize the embedding layer with the pretrained token embeddings
        self.embed = nn.Embedding(len(self.token_idx2vec.keys()),
                                  embedding_dim=self.embed_dim,
                                  padding_idx=self.token2idx['<PAD>'])
        self.embed.weight.data.copy_(torch.tensor(list(self.token_idx2vec.values()), dtype=torch.float))

        self.instr_proj = nn.Linear(self.embed_dim, self.args.hidden_size)
        self.stack_proj = nn.Linear(self.args.input_stack_dim, self.args.hidden_size)

        # Initialize the transformer encoder layer for instruction embeddings
        self.instr_encoder = SequenceAggregator(self.args)

        # Initialize the graph encoder layer for cfg embeddings
        self.cfg_encoder = CFGEncoder(self.args)

        # Initialize the constant embedding layer for constant embeddings
        self.const_embed = nn.Embedding(len(self.const2idx.keys()),
                                        embedding_dim=self.embed_dim,
                                        padding_idx=self.const2idx['<PAD>'])
        self.const_embed.weight.data.normal_(0, 1 / self.embed_dim ** 0.5)
        # Set the first row (corresponding to <PAD>) to all zeros
        self.const_embed.weight.data[self.const2idx['<PAD>']] = torch.zeros(self.embed_dim, dtype=torch.float)
        # Initialize the constant encoder layer for constant embeddings
        self.const_encoder = ConstNAU(self.args)

        if self.args.with_dcmp:
            # Initialize the dcmp encoder layer for dcmp embeddings
            self.dcmp_proj = nn.Linear(self.args.dcmp_dim, self.args.hidden_size)
            self.encoder_proj = nn.Linear(self.args.hidden_size * 4, self.args.hidden_size)
        else:
            self.encoder_proj = nn.Linear(self.args.hidden_size * 3, self.args.hidden_size)

        # Initialize the type decoder layer for type prediction
        self.decoder = TypeDecoder(self.ty2idx, self.args)

        self.encoder_outputs = None

        self.pre_out_attn = FeatureAttention(self.args.hidden_size)

    def get_encoder_outputs(self):
        return self.encoder_outputs

    def forward(self, instr, stack, const, dcmp,
                node_pos, x, edge_index,
                targets=None, teacher_forcing=None):

        instr_feature = self.embed(instr)
        instr_feature = self.instr_proj(instr_feature)

        if self.args.with_stack:
            stack_proj = self.stack_proj(stack)
            instr_feature = self.instr_encoder(instr_feature, stack_proj)
        else:
            instr_feature = self.instr_encoder(instr_feature)

        # Using graphsage network for embedding the cfg graph
        cfg_feature = self.cfg_encoder(x, edge_index, node_pos)  # bs x hidden_size

        # Embedding the constant values
        const = self.const_embed(const)
        const_feature = self.const_encoder(const)  # bs x hidden_size

        # print(instr_feature.shape)
        # print(cfg_feature.shape)
        # print(const_feature.shape)

        # Concatenate the embeddings
        if self.args.module == 'classification':
            # features = torch.cat([instr_feature, cfg_feature, const_feature], dim=-1)
            # features = self.encoder_proj(features)

            # Mean TODO testing it
            # features = torch.mean(torch.stack([instr_feature, cfg_feature, const_feature]), dim=0)

            # sum
            # features = instr_feature + cfg_feature + const_feature

            # weighted mean
            # features = self.pre_out_attn(instr_feature, cfg_feature, const_feature)

            if self.args.with_dcmp:
                dcmp_feature = self.dcmp_proj(dcmp)
                features = self.pre_out_attn(instr_feature, cfg_feature, const_feature, dcmp_feature)
            else:
                features = self.pre_out_attn(instr_feature, cfg_feature, const_feature)

            self.encoder_outputs = features  # Save the encoder outputs for similarity computation
            outputs = self.decoder(features)
        else:
            # print(instr_feature.shape)
            cfg_feature_expanded = cfg_feature.unsqueeze(1).repeat(1, self.args.max_seq_length, 1)
            const_feature_expanded = const_feature.unsqueeze(1).repeat(1, self.args.max_seq_length, 1)

            # sum
            # features = instr_feature + cfg_feature_expanded + const_feature_expanded

            if self.args.with_dcmp:
                dcmp_feature = self.dcmp_proj(dcmp)
                dcmp_feature_expanded = dcmp_feature.unsqueeze(1).repeat(1, self.args.max_seq_length, 1)

                features = self.pre_out_attn(instr_feature, cfg_feature_expanded,
                                             const_feature_expanded, dcmp_feature_expanded)

            else:
                features = self.pre_out_attn(instr_feature, cfg_feature_expanded, const_feature_expanded)

            self.encoder_outputs = features  # Save the encoder outputs for similarity computation
            outputs = self.decoder(features, targets=targets, teacher_forcing=teacher_forcing)

        # features = instr_feature + cfg_feature + const_feature

        return outputs

    def decode(self, instr, stack, const, dcmp, node_pos, x, edge_index, targets):
        # enc_output = self.encoder(dfs, values)

        # seq_length = enc_output.data.shape[1]
        #
        # s = torch.zeros((batch_size, 1)).to(self.args.device)
        # s[:, 0] = self.word2idx['<SEM>']
        # inputs = torch.cat([s, values], dim=1)

        # hidden = Variable(torch.zeros(1, batch_size, self.hidden_size)).to(self.args.device)
        batch_size = instr.shape[0]

        instr_feature = self.embed(instr)
        instr_feature = self.instr_proj(instr_feature)

        if self.args.with_stack:
            stack_proj = self.stack_proj(stack)
            instr_feature = self.instr_encoder(instr_feature, stack_proj)
        else:
            instr_feature = self.instr_encoder(instr_feature)

        # Using graphsage network for embedding the cfg graph
        cfg_feature = self.cfg_encoder(x, edge_index, node_pos)  # bs x hidden_size

        # Embedding the constant values
        const = self.const_embed(const)
        const_feature = self.const_encoder(const)  # bs x hidden_size

        cfg_feature_expanded = cfg_feature.unsqueeze(1).repeat(1, self.args.max_seq_length, 1)
        const_feature_expanded = const_feature.unsqueeze(1).repeat(1, self.args.max_seq_length, 1)

        if self.args.with_dcmp:
            dcmp_feature = self.dcmp_proj(dcmp)
            dcmp_feature_expanded = dcmp_feature.unsqueeze(1).repeat(1, self.args.max_seq_length, 1)

            features = self.pre_out_attn(instr_feature, cfg_feature_expanded,
                                         const_feature_expanded, dcmp_feature_expanded)

        else:
            features = self.pre_out_attn(instr_feature, cfg_feature_expanded, const_feature_expanded)

        # features = instr_feature + cfg_feature_expanded + const_feature_expanded

        hidden = Variable(torch.zeros(1, batch_size, self.hidden_size)).to(self.args.device)
        return self.beam_decode(hidden, features)

    @timeit
    def beam_decode(self, decoder_hiddens, encoder_outputs):
        # prev_idx, prev_hidden, encoder_outputs
        beam_width = self.args.beam_width
        topk = self.args.topk  # how many sentence do you want to generate
        max_length = self.args.max_label_length
        decoded_batch = []

        # decoding goes sentence by sentence
        for idx in range(encoder_outputs.size(0)):  # batch_size
            # if isinstance(decoder_hiddens, tuple):  # LSTM case
            #     decoder_hidden = (
            #         decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
            # else:
            #     decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
            encoder_output = encoder_outputs[idx, :, :].unsqueeze(0)

            # Start with the start of the sentence token
            decoder_input = torch.LongTensor([self.ty2idx['<TY_SOS>']]).to(self.args.device)

            selective_read = Variable(torch.zeros(1, 1, self.hidden_size)).to(self.args.device)

            # one_hot_input_seq = to_one_hot(inputs[idx].unsqueeze(0), self.vocab_size + seq_length)
            # if next(self.parameters()).is_cuda:
            #     selective_read = selective_read.to(self.args.device)
            #     one_hot_input_seq = one_hot_input_seq.to(self.args.device)
            #
            # # Start with the start of the sentence token
            # decoder_input_sos = torch.tensor([self.word2idx['<SOS>']], dtype=torch.long).to(self.args.device)

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            while True:
                if qsize > 1000:
                    break

                score, n = nodes.get()

                decoder_input = n.token_id
                decoder_input = decoder_input.view(1, decoder_input.size(0))
                decoder_hidden = n.h

                if n.token_id.item() == self.ty2idx['<TY_EOS>'] and n.prev_node is not None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # _, decoder_output, decoder_hidden, selective_read = self.decoder.step(decoder_input,
                #                                                                       decoder_hidden,
                #                                                                       encoder_output,
                #                                                                       selective_read,
                #                                                                       one_hot_input_seq)

                # print(encoder_output.shape)
                # exit()
                # decode for one step using decoder
                sampled_idx, probs, decoder_hidden = self.decoder.step(decoder_input,
                                                                       decoder_hidden,
                                                                       encoder_output)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(probs, beam_width, dim=1)
                nextnodes = []

                for new_k in range(beam_width):
                    # decoded_t = indexes[0][new_k].view(-1)
                    decoded_t = indexes[0][new_k].unsqueeze(0)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.log_prob + log_p, n.length + 1)
                    score = -node.eval()

                    if node.length <= max_length:
                        nextnodes.append((score, node))
                    # nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(to_np(n.token_id)[0])
                while n.prev_node is not None:
                    n = n.prev_node
                    utterance.append(to_np(n.token_id)[0])

                utterance = utterance[::-1][:max_length]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        return decoded_batch


class BeamSearchNode(object):
    def __init__(self, hidden, prev_node, token_id, log_prob, length):
        self.h = hidden
        self.prev_node = prev_node
        self.token_id = token_id
        self.log_prob = log_prob
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0
        return self.log_prob / float(self.length - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.length < other.length

    def __gt__(self, other):
        return self.length > other.length


class TypeDecoder(nn.Module):

    def __init__(self, ty2idx, args):
        super(TypeDecoder, self).__init__()

        self.ty2idx = ty2idx
        self.args = args

        if self.args.module == 'classification':
            self.decoder = nn.Linear(self.args.hidden_size, len(self.ty2idx.keys()))
        else:
            self.hidden_size = self.args.hidden_size
            self.embedding_dim = self.args.hidden_size
            self.ty2idx = ty2idx
            self.vocab_size = len(self.ty2idx.keys())

            self.embed = nn.Embedding(len(self.ty2idx.keys()),
                                      self.embedding_dim,
                                      padding_idx=self.ty2idx['<TY_PAD>'])

            self.embed.weight.data.normal_(0, 1 / self.embedding_dim ** 0.05)
            self.embed.weight.data[self.ty2idx['<TY_PAD>'], :] = 0.0

            self.attn_W = nn.Linear(self.hidden_size, self.hidden_size)

            # input = (context + selective read size + embedding)
            self.gru = nn.GRU(2 * self.hidden_size,
                              self.hidden_size, batch_first=True)

            self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, encoder_outputs,  # B x L x dim
                # final_encoder_hidden,  # 2 x B x dim
                targets=None, teacher_forcing=0.0, keep_prob=1.0):
        if self.args.module == 'classification':
            output = self.decoder(encoder_outputs)
            return output
        else:
            batch_size = encoder_outputs.size(0)  # B
            seq_length = encoder_outputs.size(1)  # L

            # 1 x B x dim
            hidden = Variable(torch.zeros(1, batch_size, self.hidden_size)).to(encoder_outputs.device)

            # every decoder output seq starts with <TY_SOS>
            # sos_output = Variable(torch.zeros((batch_size, self.vocab_size + seq_length)))  # B x (L + seq<64>)

            sos_output = Variable(torch.zeros(batch_size, self.vocab_size)).to(encoder_outputs.device)
            # index of the <TY_SOS> token, one-hot encoding
            sos_output[:, self.ty2idx['<TY_SOS>']] = 1.0

            decoder_outputs = [sos_output]
            # B x 1
            sampled_idxs = [Variable(torch.ones((batch_size, 1)).long()).to(encoder_outputs.device)]

            for step_idx in range(1, self.args.max_label_length):
                sampled_idx = sampled_idxs[-1]

                if targets is not None and teacher_forcing > 0.0 and step_idx < targets.size(1):
                    teacher_forcing_mask = (
                                torch.rand((batch_size, 1), device=encoder_outputs.device) < teacher_forcing)
                    sampled_idx = torch.where(teacher_forcing_mask, targets[:, step_idx - 1: step_idx], sampled_idx)

                sampled_idx, probs, hidden = self.step(sampled_idx, hidden, encoder_outputs)

                decoder_outputs.append(probs)
                sampled_idxs.append(sampled_idx)

            decoder_outputs = torch.stack(decoder_outputs, dim=1)
            sampled_idxs = torch.stack(sampled_idxs, dim=1)

            # if keep_prob < 1.0:
            #     dropout_mask = (Variable(torch.rand(
            #         batch_size, 1, 2 * self.hidden_size + self.embed.embedding_dim)) < keep_prob).float() / keep_prob
            # else:
            #     dropout_mask = None
            #
            # selective_read = Variable(torch.zeros(batch_size, 1, self.hidden_size))       # B x 1 x dim
            # one_hot_input_seq = to_one_hot(inputs, self.vocab_size + seq_length)   # B x (L + seq)
            # # one_hot_input_seq = to_one_hot(inputs, self.vocab_size)   # B x (L + seq)
            # if next(self.parameters()).is_cuda:
            #     selective_read = selective_read.cuda()
            #     one_hot_input_seq = one_hot_input_seq.cuda()
            #
            # for step_idx in range(1, self.args.max_length_output):
            #     if targets is not None and teacher_forcing > 0.0 and step_idx < targets.shape[1]:
            #         # replace some inputs with the targets (i.e. teacher forcing)
            #         # B x 1
            #         teacher_forcing_mask = Variable((torch.rand((batch_size, 1)) < teacher_forcing), requires_grad=False)
            #         if next(self.parameters()).is_cuda:
            #             teacher_forcing_mask = teacher_forcing_mask.cuda()
            #
            #         sampled_idx = sampled_idx.masked_scatter(teacher_forcing_mask, targets[:, step_idx - 1: step_idx])
            #
            #     sampled_idx, output, hidden, selective_read = self.step(
            #         sampled_idx, hidden, encoder_outputs, selective_read, one_hot_input_seq, dropout_mask=dropout_mask)
            #
            #     decoder_outputs.append(output)
            #     sampled_idxs.append(sampled_idx)
            #
            # decoder_outputs = torch.stack(decoder_outputs, dim=1)
            # sampled_idxs = torch.stack(sampled_idxs, dim=1)

            return decoder_outputs, sampled_idxs

    def step(self, prev_idx, prev_hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)

        transformed_hidden = self.attn_W(prev_hidden).view(batch_size, self.hidden_size, 1)
        attn_scores = torch.bmm(encoder_outputs, transformed_hidden)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)

        embedded = self.embed(prev_idx)
        # print(context.shape)
        # print(embedded.shape)
        rnn_input = torch.cat((context, embedded), dim=2)

        # print(rnn_input.shape)
        # print(prev_hidden.shape)
        output, hidden = self.gru(rnn_input, prev_hidden)

        logits = self.out(output.squeeze(1))
        probs = F.softmax(logits, dim=1)

        _, topi = probs.topk(1)
        # print(topi.shape)
        # print('=================')
        sampled_idx = topi.view(batch_size, 1)

        return sampled_idx, probs, hidden

class SequenceAggregator(nn.Module):
    def __init__(self, args):
        super(SequenceAggregator, self).__init__()

        self.args = args
        self.transformer = SequenceEncoder(self.args)

        self.instr_aggregator = GlobalAttention(self.args.hidden_size)
        # if self.args.model_type == 'param':
        #     self.attention_aggregator = WeightedAttentionAggregator(self.args)

        # self.attention_aggregator = WeightedAttentionAggregator(self.args)

    def forward(self, src, stack=None):
        # src shape: [batch_size, seq_num, seq_len, input_dim]
        # torch.Size([128, 3, 100, 512])

        aggregated_features = []
        for i in range(src.shape[1]):
            x = src[:, i, :, :]
            transformer_output = self.transformer(x)

            if self.args.with_stack:
                stack_i = stack[:, i, :, :]
                transformer_output = transformer_output + stack_i

            aggregated_features.append(transformer_output)

        # [seq_num, batch_size, seq_len, hidden_size]
        aggregated_features = torch.stack(aggregated_features, dim=0)
        # print('aggregated_features shape: ', aggregated_features.shape)
        # sum_features = aggregated_features.sum(dim=0)
        mean_features = aggregated_features.mean(dim=0)

        # sum_features = self.attention_aggregator(aggregated_features)
        # print('sum_features shape: ', sum_features.shape)
        # print('mean features shape: ', mean_features.shape)

        if self.args.module == 'classification':
            mean_features, _ = self.instr_aggregator(mean_features)

        return mean_features


class SequenceEncoder(nn.Module):
    def __init__(self, args):
        super(SequenceEncoder, self).__init__()

        self.args = args
        self.d_model = self.args.hidden_size
        self.positional_encoding = PositionalEncoding(self.d_model, self.args.max_seq_length)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.args.hidden_size,
            nhead=self.args.transformer_n_heads,
            dim_feedforward=self.args.hidden_size,
            dropout=self.args.dropout_rate,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=self.args.transformer_n_layers)

    def forward(self, src):
        # src shape: [seq_len, batch_size, input_dim]
        src = self.positional_encoding(src)
        return self.transformer_encoder(src)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension for broadcasting

    def forward(self, x):
        self.encoding = self.encoding.to(x.device)
        return x + self.encoding[:, :x.size(1)].detach()


class CFGEncoder(nn.Module):

    def __init__(self, args):
        super(CFGEncoder, self).__init__()

        self.args = args

        # GraphSAGE layers
        self.conv1 = SAGEConv(self.args.input_node_dim, self.args.hidden_size)
        self.conv2 = SAGEConv(self.args.hidden_size, self.args.hidden_size)
        self.fc = nn.Linear(self.args.input_node_dim, self.args.hidden_size)

        # # GAT layers
        # self.conv1 = GATConv(self.args.input_node_dim, self.args.hidden_size, heads=1)
        # self.conv2 = GATConv(self.args.hidden_size, self.args.hidden_size, heads=1)
        # self.fc = nn.Linear(self.args.input_node_dim, self.args.hidden_size)

    def forward(self, graph_x, edge_index, node_pos):

        x = graph_x

        # Check if the graph has only one node
        if x.size(0) == 1 or edge_index.numel() == 0:
            # If there's only one node, skip the graph convolutions
            # print(x.shape)
            x = self.fc(x)
        else:

            # GraphSAGE
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)

            # # GAT
            #
            # x = F.dropout(x, p=0.6, training=self.training)
            # x = F.elu(self.conv1(x, edge_index))
            # x = F.dropout(x, p=0.6, training=self.training)
            # x = self.conv2(x, edge_index)

        x = self.mean_by_batch(x, node_pos)
        return x

        # x, edge_index = graph_x, edge_index
        #
        #
        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        #
        # # print(x.shape)
        # # exit()
        # # x = self.attn_by_batch(x, node_pos, self.args)  # Modify to attention or mean
        # x = self.mean_by_batch(x, node_pos)
        # return x

    def mean_by_batch(self, cfg_nodes, node_pos):
        cfg = torch.zeros((len(node_pos) - 1, self.args.hidden_size)).to(self.args.device)
        for idx in range(len(node_pos) - 1):
            cur = cfg_nodes[node_pos[idx]: node_pos[idx + 1], :]
            cfg[idx, :] = torch.mean(cur, dim=0)

        return cfg

    def attn_by_batch(self, cfg_nodes, node_pos):
        cfg = torch.zeros((len(node_pos) - 1, self.args.hidden_size)).to(self.args.device)
        for idx in range(len(node_pos) - 1):
            cur = cfg_nodes[node_pos[idx]: node_pos[idx + 1], :]
            attn_cur, _ = self.attn_cfg(cur)
            cfg[idx, :] = attn_cur
        return cfg


class GraphLocalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 8, 1))

    def forward(self, encoder_outputs):

        energy = self.projection(encoder_outputs)
        energy = energy.squeeze(-1)

        weights = F.softmax(energy, dim=0)

        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=0)
        return outputs, weights


class ConstNAU(nn.Module):
    def __init__(self, args):
        super(ConstNAU, self).__init__()

        self.args = args

        self.linear = nn.Linear(self.args.input_dim, self.args.hidden_size)

        self.attn_const = GlobalAttention(self.args.hidden_size)

    def forward(self, x):
        x = self.linear(x)
        x, _ = self.attn_const(x)
        return x


class GlobalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 8, 1))

    def forward(self, encoder_outputs):

        energy = self.projection(encoder_outputs)
        energy = energy.squeeze(-1)

        weights = F.softmax(energy, dim=1)

        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class FeatureAttention(nn.Module):
    def __init__(self, feature_dim):
        super(FeatureAttention, self).__init__()
        self.feature_dim = feature_dim
        self.attention_weights = nn.Linear(feature_dim, 1)

    def forward(self, instr_feature, cfg_feature, const_feature, dcmp_feature=None):

        # Step 1: Stack the features together along a new dimension (resulting in a 3 x feature_dim matrix)
        if dcmp_feature is not None:
            combined_features = torch.stack([instr_feature, cfg_feature, const_feature, dcmp_feature], dim=0)
        else:
            combined_features = torch.stack([instr_feature, cfg_feature, const_feature], dim=0)

        # Step 2: Compute attention scores
        # Flatten from 3 x feature_dim to 3 x 1 to apply the linear layer
        attn_scores = self.attention_weights(combined_features)
        attn_weights = F.softmax(attn_scores, dim=0)  # Softmax over the first dimension to get weights

        # Step 3: Calculate the weighted sum of the features
        weighted_sum = torch.sum(attn_weights * combined_features, dim=0)

        return weighted_sum


# class DecoderBase(ABC, nn.Module):
#     def forward(self, encoder_outputs, inputs,
#                 final_encoder_hidden, targets=None, teacher_forcing=1.0):
#         raise NotImplementedError


class CustomDecoder(nn.Module):
    def __init__(self, ty2idx, args):
        super(CustomDecoder, self).__init__()

        self.args = args
        self.hidden_size = self.args.hidden_size
        self.embedding_dim = self.args.hidden_size
        self.ty2idx = ty2idx
        self.vocab_size = len(self.ty2idx.keys())

        self.embed = nn.Embedding(len(self.ty2idx.keys()),
                                  self.embedding_dim,
                                  padding_idx=self.ty2idx['<TY_PAD>'])

        self.embed.weight.data.normal_(0, 1 / self.embedding_dim ** 0.05)
        self.embed.weight.data[self.ty2idx['<TY_PAD>'], :] = 0.0

        self.attn_W = nn.Linear(self.hidden_size, self.hidden_size)

        # input = (context + selective read size + embedding)
        self.gru = nn.GRU(2 * self.hidden_size + self.embed.embedding_dim,
                          self.hidden_size, batch_first=True)

        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, encoder_outputs,    # B x L x dim
                targets=None, keep_prob=1.0, teacher_forcing=0.0):

        batch_size = encoder_outputs.size(0)  # B
        seq_length = encoder_outputs.size(1)  # L

        # 1 x B x dim
        hidden = Variable(torch.zeros(1, batch_size, self.hidden_size)).to(encoder_outputs.device)
        # if next(self.parameters()).is_cuda:
        #     hidden = hidden.cuda()
        # else:
        #     hidden = hidden

        # every decoder output seq starts with <TY_SOS>
        # sos_output = Variable(torch.zeros((batch_size, self.vocab_size + seq_length)))  # B x (L + seq<64>)

        sos_output = Variable(torch.zeros(batch_size, self.vocab_size)).to(encoder_outputs.device)
        # index of the <TY_SOS> token, one-hot encoding
        sos_output[:, self.word2idx['<TY_SOS>']] = 1.0

        decoder_outputs = [sos_output]
        # B x 1
        sampled_idxs = [Variable(torch.ones((batch_size, 1)).long()).to(encoder_outputs.device)]

        for step_idx in range(1, self.args.max_label_length):
            sampled_idx = sampled_idxs[-1]

            if targets is not None and teacher_forcing > 0.0 and step_idx < targets.size(1):
                teacher_forcing_mask = (torch.rand((batch_size, 1), device=encoder_outputs.device) < teacher_forcing)
                sampled_idx = torch.where(teacher_forcing_mask, targets[:, step_idx - 1: step_idx], sampled_idx)

            sampled_idx, hidden = self.step(sampled_idx, hidden, encoder_outputs)

            decoder_outputs.append(self.embed(sampled_idx))
            sampled_idxs.append(sampled_idx)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        sampled_idxs = torch.stack(sampled_idxs, dim=1)

        # if keep_prob < 1.0:
        #     dropout_mask = (Variable(torch.rand(
        #         batch_size, 1, 2 * self.hidden_size + self.embed.embedding_dim)) < keep_prob).float() / keep_prob
        # else:
        #     dropout_mask = None
        #
        # selective_read = Variable(torch.zeros(batch_size, 1, self.hidden_size))       # B x 1 x dim
        # one_hot_input_seq = to_one_hot(inputs, self.vocab_size + seq_length)   # B x (L + seq)
        # # one_hot_input_seq = to_one_hot(inputs, self.vocab_size)   # B x (L + seq)
        # if next(self.parameters()).is_cuda:
        #     selective_read = selective_read.cuda()
        #     one_hot_input_seq = one_hot_input_seq.cuda()
        #
        # for step_idx in range(1, self.args.max_length_output):
        #     if targets is not None and teacher_forcing > 0.0 and step_idx < targets.shape[1]:
        #         # replace some inputs with the targets (i.e. teacher forcing)
        #         # B x 1
        #         teacher_forcing_mask = Variable((torch.rand((batch_size, 1)) < teacher_forcing), requires_grad=False)
        #         if next(self.parameters()).is_cuda:
        #             teacher_forcing_mask = teacher_forcing_mask.cuda()
        #
        #         sampled_idx = sampled_idx.masked_scatter(teacher_forcing_mask, targets[:, step_idx - 1: step_idx])
        #
        #     sampled_idx, output, hidden, selective_read = self.step(
        #         sampled_idx, hidden, encoder_outputs, selective_read, one_hot_input_seq, dropout_mask=dropout_mask)
        #
        #     decoder_outputs.append(output)
        #     sampled_idxs.append(sampled_idx)
        #
        # decoder_outputs = torch.stack(decoder_outputs, dim=1)
        # sampled_idxs = torch.stack(sampled_idxs, dim=1)

        return decoder_outputs, sampled_idxs

    def step(self, prev_idx, prev_hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)

        transformed_hidden = self.attn_W(prev_hidden).view(batch_size, self.hidden_size, 1)
        attn_scores = torch.bmm(encoder_outputs, transformed_hidden)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)

        embedded = self.embed(prev_idx)
        rnn_input = torch.cat((context, embedded), dim=2)

        output, hidden = self.gru(rnn_input, prev_hidden)

        logits = self.out(output.squeeze(1))
        probs = F.softmax(logits, dim=1)

        _, topi = probs.topk(1)
        sampled_idx = topi.view(batch_size, 1)

        return sampled_idx, hidden


    # def step(self, prev_idx, prev_hidden, encoder_outputs,
    #          prev_selective_read, one_hot_input_seq, dropout_mask=None):
    #
    #     batch_size = encoder_outputs.shape[0]
    #     seq_length = encoder_outputs.shape[1]
    #     # vocab_size = len(self.word2idx.keys())
    #
    #     # ## Attention mechanism
    #     transformed_hidden = self.attn_W(prev_hidden)
    #     transformed_hidden = transformed_hidden.view(batch_size, self.hidden_size, 1)  # B x dim x 1
    #     # reduce encoder outputs and hidden to get scores.
    #     # remove singleton dimension from multiplication.
    #
    #     attn_scores = torch.bmm(encoder_outputs, transformed_hidden)   # B x L x dim * B x dim x 1 => B x L x 1
    #     attn_weights = F.softmax(attn_scores, dim=1)      # B x L x 1
    #     # [b, 1, hidden] weighted sum of encoder_outputs (i.e. values)
    #
    #     # B x 1 x L * B x L x dim => B x 1 x dim     <attn among all encoder inputs>
    #     context = torch.bmm(torch.transpose(attn_weights, 1, 2), encoder_outputs)
    #
    #     # ## Call the RNN
    #     # [b, 1] bools indicating which seqs copied on the previous step
    #     out_of_vocab_mask = prev_idx >= self.reserved_vocab_size  # > self.vocab_size
    #     unks = torch.ones_like(prev_idx).long() * self.word2idx['<UNK>']
    #     # replace copied tokens with <UNK> token before embedding
    #     prev_idx = prev_idx.masked_scatter(out_of_vocab_mask, unks)
    #     # embed input (i.e. previous output token)
    #     embedded = self.embed(prev_idx)
    #
    #     # B x 1 x dim | B x 1 x dim | B x 1 x dim
    #     rnn_input = torch.cat((context, prev_selective_read, embedded), dim=2)
    #     if dropout_mask is not None:
    #         if next(self.parameters()).is_cuda:
    #             dropout_mask = dropout_mask.cuda()
    #
    #         rnn_input *= dropout_mask
    #
    #     self.gru.flatten_parameters()
    #     output, hidden = self.gru(rnn_input, prev_hidden)  # B x 1 x dim
    #
    #     # ## Copy mechanism
    #     transformed_hidden_2 = self.copy_W(output).view(batch_size, self.hidden_size, 1)  # B x dim x 1
    #     # this is linear. add activation function before multiplying.
    #     copy_score_seq = torch.bmm(encoder_outputs, transformed_hidden_2)  # B x L x 1
    #     # [b, 1, vocab_size + seq_length] * B x L x 1
    #     copy_scores = torch.bmm(torch.transpose(copy_score_seq, 1, 2), one_hot_input_seq).squeeze(1)
    #     # tokens not present in the input sequence
    #     missing_token_mask = (one_hot_input_seq.sum(dim=1) == 0)
    #     # <PAD> tokens are not part of any sequence
    #     missing_token_mask[:, self.word2idx['<PAD>']] = 1
    #     copy_scores = copy_scores.masked_fill(missing_token_mask, -1000000.0)
    #
    #     # ## Generate mechanism
    #     gen_scores = self.out(output.squeeze(1))  # [b. vocab_size]
    #     gen_scores[:, self.word2idx['<PAD>']] = -1000000.0  # penalize <PAD> tokens in generate mode too
    #
    #     # ## Combine results from copy and generate mechanisms
    #     combined_scores = torch.cat((gen_scores, copy_scores), dim=1)
    #     probs = F.softmax(combined_scores, dim=1)
    #     # gen_probs = probs[:, :self.reserved_vocab_size]
    #     gen_probs = probs[:, :self.vocab_size]
    #
    #     gen_padding = Variable(torch.zeros(batch_size, seq_length))
    #     # gen_padding = Variable(torch.zeros(batch_size, self.vocab_size - self.reserved_vocab_size))
    #     if next(self.parameters()).is_cuda:
    #         gen_padding = gen_padding.cuda()
    #
    #     gen_probs = torch.cat((gen_probs, gen_padding), dim=1)  # [b, vocab_size + seq_length]
    #
    #     # copy_probs = probs[:, self.reserved_vocab_size:]
    #     copy_probs = probs[:, self.vocab_size:]
    #
    #     final_probs = gen_probs + copy_probs
    #
    #     log_probs = torch.log(final_probs + 10 ** -10)
    #
    #     _, topi = log_probs.topk(1)
    #     sampled_idx = topi.view(batch_size, 1)
    #
    #     # ## Create selective read embedding for next time step
    #     reshaped_idxs = sampled_idx.view(-1, 1, 1).expand(one_hot_input_seq.size(0), one_hot_input_seq.size(1), 1)
    #     pos_in_input_of_sampled_token = one_hot_input_seq.gather(2, reshaped_idxs)  # [b, seq_length, 1]
    #     selected_scores = pos_in_input_of_sampled_token * copy_score_seq
    #     selected_socres_norm = F.normalize(selected_scores, p=1)
    #
    #     selective_read = (selected_socres_norm * encoder_outputs).sum(dim=1).unsqueeze(1)
    #
    #     return sampled_idx, log_probs, hidden, selective_read