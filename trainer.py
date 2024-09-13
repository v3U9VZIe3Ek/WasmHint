import os
from tqdm import tqdm
import logging
import random
import numpy as np
import torch
from tqdm import trange
import torch.nn as nn
import torch.nn.functional as F
from utils import trim_seqs_beam, to_np, convert2type_topk, convert2type
from datetime import datetime
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score


logger = logging.getLogger(__name__)


def get_input_from_batch(batch, args):
    # instr_feature, stack_feature, const_feature, dcmp_feature, graph, label, func_ident
    instr_feature = batch[0].to(args.device)
    stack_feature = batch[1].to(args.device)
    const_feature = batch[2].to(args.device)
    dcmp_feature = batch[3].to(args.device)

    graph = batch[4].to(args.device)
    node_pos = graph.ptr.to(args.device)
    x = graph.x.to(args.device)
    edge_index = graph.edge_index.to(args.device)

    if args.module == 'generation':
        label = batch[5].to(args.device)
    else:
        label = batch[5][:, 1]
        label = label.to(args.device)
    func_ident = batch[6].to(args.device)

    return (instr_feature, stack_feature, const_feature, dcmp_feature, node_pos, x, edge_index, func_ident), label

    # if args.loss == 'single':
    #     return (instr_feature, stack_feature, const_feature, dcmp_feature, node_pos, x, edge_index, func_ident), label
    # else:
    #     inputs = (instr_feature, stack_feature, const_feature, dcmp_feature, node_pos, x, edge_index, func_ident)
    #
    #     # Create positive pairs for contrastive learning
    #     batch_size = instr_feature.size(0)
    #     pos_indices = torch.arange(batch_size, device=args.device)
    #     pos_indices = (pos_indices + torch.randint(1, batch_size, (batch_size,), device=args.device)) % batch_size
    #
    #     pos_inputs = tuple(feat[pos_indices] if isinstance(feat, torch.Tensor) else feat for feat in inputs)
    #
    #     # Create a mask for valid positive pairs (excluding self-pairs)
    #     pos_mask = torch.eye(batch_size, device=args.device, dtype=torch.bool)
    #     pos_mask = ~pos_mask
    #
    #     return inputs, label, pos_inputs, pos_mask


def generate_negative_pairs(batch_features, batch_targets):
    # Assuming batch_features and batch_targets are tensors of shape [batch_size, seq_len]
    # Negative samples for each item in the batch are selected by shuffling targets
    permuted_indices = torch.randperm(batch_targets.size(0))
    negative_targets = batch_targets[permuted_indices]
    return batch_features, negative_targets


def trainer(args, model, train_dataset, test_dataset, token2idx, idx2token, ty2idx, idx2ty, idx2path, schedule):

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    logger.info('The total number of train dataset is %d' % len(train_dataset))
    logger.info('The total number of test dataset is %d' % len(test_dataset))

    # Training model
    start_time = datetime.now()

    if not args.test_only:
        do_train(args, model, train_dataloader, ty2idx, schedule)

        end_time = datetime.now()
        logger.info('Model training spends %s' % str((end_time - start_time).seconds))

    # Test model
    with_cfg = 'w_cfg' if args.with_cfg else 'wo_cfg'
    with_stack = 'w_stack' if args.with_stack else 'wo_stack'
    with_const = 'w_const' if args.with_consts else 'wo_const'
    with_dcmp = 'w_dcmp' if args.with_dcmp else 'wo_dcmp'

    filename = f'model_{args.infer_type}_{args.module}_{with_cfg}_{with_stack}_{with_const}_{with_dcmp}_ml_{str(args.max_seq_length)}.pth'

    model_saved_path = os.path.join(args.model_save_path, filename)

    # To deal with multiple GPUs
    state_dict = torch.load(model_saved_path)

    # If your current model is wrapped in DataParallel or DistributedDataParallel, unwrap it
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    model.load_state_dict(state_dict)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(args.device)

    start_time = datetime.now()

    do_test(args, model, test_dataloader, idx2ty, ty2idx, idx2path)

    end_time = datetime.now()

    logger.info('Model testing spends %s' % str((end_time - start_time).seconds))

    # return indicators, internal_time


def do_train(args, model, train_dataset, ty2idx, schedule):
    model_save_path = args.model_save_path

    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    logger.info('****** Start training ******')

    model.zero_grad()

    best_loss = float('inf')
    global_step = 0  # Initialize global step counter

    for cur_epoch, teacher_forcing in enumerate(schedule):
        logger.info('============================= Current Epoch: %s =======================' % str(cur_epoch))

        total_loss = 0.0  # Initialize total loss for the epoch

        for batch in tqdm(train_dataset):

            model.train()

            # if args.loss == 'single':
            #     batch_features, labels = get_input_from_batch(batch, args)
            # else:
            #     batch_features, labels, pos_inputs, pos_mask = get_input_from_batch(batch, args)

            batch_features, labels = get_input_from_batch(batch, args)
            instr, stack, const, dcmp, node_pos, x, edge_index, func_ident = batch_features

            if args.module == 'generation':
                decoder_outputs, sampled_idxs = model(
                    instr, stack, const, dcmp, node_pos, x, edge_index, labels, teacher_forcing)
                batch_size = instr.shape[0]
                # print(decoder_outputs.shape)
                # print(decoder_outputs.shape)
                # print(labels.shape)
                # exit()
                flattened_outputs = decoder_outputs.view(batch_size * args.max_label_length, -1)
                loss = get_loss_gene(flattened_outputs, labels.contiguous().view(-1), ty2idx)

                if args.use_reinforcement_learning:
                    # Compute rewards
                    rewards = compute_reward(flattened_outputs, labels.view(-1))
                    # Compute loss and back-propagate
                    loss += -rewards   # Negative reward as we're doing gradient ascent

                # if args.loss != 'single':
                #     # contrastive learning
                #     _, neg_labels = generate_negative_pairs(batch_features, labels)
                #
                #     neg_decoder_outputs, _ = model(instr, stack, const, dcmp, node_pos, x, edge_index, neg_labels, teacher_forcing)
                #
                #     cl_loss = F.triplet_margin_loss(anchor=decoder_outputs, positive=decoder_outputs,
                #                                     negative=neg_decoder_outputs, margin=1.0)
                #     loss += cl_loss

                if args.loss == 'double':
                    encoder_outputs = model.get_encoder_outputs()
                    # mean_encoder_outputs = torch.mean(encoder_outputs, dim=1)
                    # mean_decoder_outputs = torch.mean(decoder_outputs, dim=1)
                    sim_loss = cosine_loss(encoder_outputs, decoder_outputs, args)
                    loss += sim_loss

            else:
                logits = model(instr, stack, const, dcmp, node_pos, x, edge_index, labels, teacher_forcing)
                loss = get_loss(logits, labels.view(-1))

                # if args.loss != 'simple':
                #     # contrastive learning
                #     _, neg_labels = generate_negative_pairs(batch_features, labels)
                #
                #     neg_logits = model(
                #         instr, stack, const, dcmp, node_pos, x, edge_index, neg_labels, teacher_forcing)
                #
                #     cl_loss = F.triplet_margin_loss(anchor=logits, positive=logits,
                #                                     negative=neg_logits, margin=1.0)
                #     loss += cl_loss

                if args.loss == 'double':
                    # print('Here using similarity loss!')
                    encoder_outputs = model.get_encoder_outputs()
                    # mean_encoder_outputs = torch.mean(encoder_outputs, dim=1)
                    # mean_decoder_outputs = torch.mean(logits, dim=1)
                    sim_loss = cosine_loss(encoder_outputs, logits, args)
                    loss += sim_loss

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Gradient clip, if needed
            optimizer.step()

            total_loss += loss.item()

            global_step += 1  # Increment global step

            # Save model conditionally at certain steps
            if global_step % args.save_interval == 0 and global_step > 0:
                # if global_step > 0:
                current_loss = total_loss / global_step
                if current_loss < best_loss:
                    current_loss = total_loss / global_step
                    logger.info(f"Saving model at step {global_step}, loss: {current_loss:.4f}")
                    best_loss = current_loss
                    os.makedirs(model_save_path, exist_ok=True)

                    # Take care of distributed/parallel training
                    # model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save = model

                    # print(model.keys())
                    with_cfg = 'w_cfg' if args.with_cfg else 'wo_cfg'
                    with_stack = 'w_stack' if args.with_stack else 'wo_stack'
                    with_const = 'w_const' if args.with_consts else 'wo_const'
                    with_dcmp = 'w_dcmp' if args.with_dcmp else 'wo_dcmp'

                    filename =f'model_{args.infer_type}_{args.module}_{with_cfg}_{with_stack}_{with_const}_{with_dcmp}_ml_{str(args.max_seq_length)}.pth'
                    save_path = os.path.join(model_save_path, filename)
                    torch.save(model_to_save.state_dict(), save_path)

        epoch_loss = total_loss / len(train_dataset)  # Calculate average loss for the epoch
        logger.info(f'Epoch {cur_epoch + 1}, Loss: {epoch_loss:.4f}')

    logger.info('Training complete. ')


def do_test(args, model, test_dataset, idx2ty, ty2idx, idx2path):
    logger.info('****** Running Testing *******')

    preds = None
    trues = None
    funcs = None

    for batch in tqdm(test_dataset):

        model.eval()

        with torch.no_grad():
            batch_features, labels = get_input_from_batch(batch, args)

            instr, stack, const, dcmp, node_pos, x, edge_index, func_ident = batch_features
            # if args.module == 'generation':
            #     decoder_outputs, sampled_idxs = model(
            #         instr, stack, const, dcmp, node_pos, x, edge_index, labels, teacher_forcing)
            #     batch_size = instr.shape[0]
            #     print(decoder_outputs.shape)
            #     flattened_outputs = decoder_outputs.view(batch_size * args.max_label_length, -1)
            #     loss = get_loss_gene(flattened_outputs, labels.contiguous().view(-1), ty2idx)
            #     print('Here!')
            # else:
            #     logits = model(instr, stack, const, dcmp, node_pos, x, edge_index, labels, teacher_forcing)
            #     loss = get_loss(logits, labels.view(-1))
            if args.module == 'generation':
                # decoder_outputs, sampled_idxs = model(
                #     instr, stack, const, dcmp, node_pos, x, edge_index)
                decoder_outputs = model.decode(instr, stack, const, dcmp, node_pos, x, edge_index, labels)

                # print('=================================================')
                # print(batch_outputs_ids)
                # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                # print(batch_targets_ids)
                # exit()
                batch_outputs_ids = trim_seqs_beam(decoder_outputs, ty2idx)
                batch_targets_ids = [list(seq[seq > 0]) for seq in list(to_np(labels))]

                batch_preds = convert2type_topk(batch_outputs_ids, idx2ty)
                batch_targets = convert2type(batch_targets_ids, idx2ty)

            else:
                logits = model(instr, stack, const, dcmp, node_pos, x, edge_index)
                batch_targets = labels.detach().cpu().numpy()
                _, batch_preds = torch.max(logits, dim=1)
                batch_preds = batch_preds.detach().cpu().numpy()

            if preds is None:
                preds = batch_preds
                trues = batch_targets
                funcs = func_ident.detach().cpu().numpy()
            else:
                # print(preds)
                # exit()
                preds = np.append(preds, batch_preds, axis=0)
                trues = np.append(trues, batch_targets, axis=0)
                funcs = np.append(funcs, func_ident.detach().cpu().numpy(), axis=0)

    # print(len(preds))
    # print(len(trues))

    logger.info('Testing complete. ')

    indicators = cal_indicators(preds, trues, ty2idx, args)

    print(indicators)

    if args.module == 'generation':
        # Save to TXT file
        with open('results/%s_true_pred_func_gene.txt' % args.infer_type, mode='w') as file:
            file.write('pred\ttrue\tfunc\n')
            for func, pred, true in zip(funcs, preds, trues):
                # # print(type(pred))
                # if isinstance(pred, list):
                #     temp_pred = [idx2ty[idx] for idx in pred]
                # elif isinstance(pred, np.ndarray):
                #     print(pred)
                #     exit()
                # else:  # numpy array
                #     temp_pred = idx2ty[pred]
                # if isinstance(true, list):
                #     temp_true = [idx2ty[idx] for idx in true]
                # else:
                #     temp_true = idx2ty[true]
                # # temp_true = [idx2ty[idx] for idx in true]
                temp_func = idx2path[func]
                temp_pred = ' <SEP> '.join(pred)
                file.write(f'{true} <PRED> {temp_pred} <FUNC> {temp_func}\n')
    else:
        # Save to TXT file
        with open('results/%s_pred_true_func.txt' % args.infer_type, mode='w') as file:
            file.write('pred\ttrue\tfunc\n')
            for func, pred, true in zip(funcs, preds, trues):
                # print(type(pred))
                if isinstance(pred, list):
                    temp_pred = [idx2ty[idx] for idx in pred]
                else: # numpy array
                    temp_pred = idx2ty[pred]
                if isinstance(true, list):
                    temp_true = [idx2ty[idx] for idx in true]
                else:
                    temp_true = idx2ty[true]
                # temp_true = [idx2ty[idx] for idx in true]
                temp_func = idx2path[func]
                file.write(f'{temp_pred}\t{temp_true}\t{temp_func}\n')


def cal_indicators(preds, trues, ty2idx, args):
    assert len(preds) == len(trues)

    if args.module == 'generation':

        top1_acc = cal_top_k_acc(preds, trues, topk=1)
        top3_acc = cal_top_k_acc(preds, trues, topk=3)
        top5_acc = cal_top_k_acc(preds, trues, topk=5)

        ret = {
            'top1_acc': str(top1_acc),
            'top3_acc': str(top3_acc),
            'top5_acc': str(top5_acc),
        }

    else:
        accuracy = accuracy_score(trues, preds)
        precision = precision_score(trues, preds, average='macro')
        recall = recall_score(trues, preds, average='macro')
        f1 = f1_score(trues, preds, average='macro')

        # weighted
        weighted_precision = precision_score(trues, preds, average='weighted')
        weighted_recall = recall_score(trues, preds, average='weighted')
        weighted_f1 = f1_score(trues, preds, average='weighted')

        ret = {
            'accuracy': str(accuracy),
            'precision': str(precision),
            'recall': str(recall),
            'f1': str(f1),
            'weighted_precision': str(weighted_precision),
            'weighted_recall': str(weighted_recall),
            'weighted_f1': str(weighted_f1)
        }

    return ret


def get_loss(logits, y_true):
    loss = nn.CrossEntropyLoss()
    output = loss(logits, y_true)
    return output


# def top_k_accuracy(true_labels, pred_labels, k):
#     top_k_correct = np.any(pred_labels == np.expand_dims(true_labels, axis=1), axis=1)
#     return np.mean(top_k_correct)

def get_loss_gene(logits, y_true, ty2idx):
    loss = nn.NLLLoss(ignore_index=ty2idx['<TY_PAD>'])
    # loss = nn.CrossEntropyLoss(ignore_index=ty2idx['<TY_PAD>'])
    output = loss(logits, y_true)

    return output


def cal_top_k_acc(preds, trues, topk):
    cnt = 0
    for i in range(len(preds)):
        topk_preds = preds[i][:topk]
        if trues[i] in topk_preds:
            cnt += 1

    acc = cnt / len(preds)
    return acc


def compute_reward(outputs, targets):
    predictions = outputs.argmax(dim=-1)
    correct = (predictions == targets).float()  # Convert to float tensor
    accuracy = correct.sum() / correct.numel()
    accuracy = torch.tensor(accuracy, requires_grad=True)
    return accuracy


def cosine_loss(encoder_outputs, decoder_outputs, args):
    # print(decoder_outputs.shape)
    # print(encoder_outputs.shape)

    if args.module == 'generation':
        bs = encoder_outputs.size(0)
        encoder_outputs = encoder_outputs.view(bs, -1)
        decoder_outputs = decoder_outputs.view(bs, -1)

    encoder_norm = F.normalize(encoder_outputs, p=2, dim=-1)
    decoder_norm = F.normalize(decoder_outputs, p=2, dim=-1)

    encoder_transposed = encoder_norm.transpose(1, 0)

    cos_sim_matrix = torch.matmul(encoder_transposed, decoder_norm)

    loss = (1 - cos_sim_matrix).mean()
    return loss


def contrastive_loss(anchors, positives, mask, temperature=0.1):
    # Compute similarity matrix
    sim_matrix = torch.matmul(anchors, positives.transpose(0, 1)) / temperature

    # Apply mask to exclude self-pairs
    sim_matrix = sim_matrix.masked_fill(~mask, -float('inf'))

    # Compute contrastive loss
    loss = -torch.log_softmax(sim_matrix, dim=1).diag().mean()

    return loss