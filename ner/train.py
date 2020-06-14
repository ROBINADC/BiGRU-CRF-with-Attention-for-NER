# -*- coding: utf-8 -*-

"""
Train logic for models
Created on 2020/5/27 
"""

__author__ = "Yihang Wu"

import os
import pickle

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from gamma import build_model, EarlyStopping, cal_f1score, DataLoader, load_embeddings


def run(arg):

    # Create necessary directories
    if not os.path.exists(arg.event_dir):
        os.makedirs(arg.event_dir)

    ckpt_dir = os.path.join(arg.ckpt_dir, arg.model_name)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Load padded dataset and lookup tables
    with open(arg.padded_dataset_path, 'rb') as fin:
        dataset = pickle.load(fin)

    with open(arg.lookup_path, 'rb') as fin:
        lookup = pickle.load(fin)

    train_sens = dataset['sens_train']
    train_ents = dataset['ents_train']
    val_sens = dataset['sens_val']
    val_ents = dataset['ents_val']

    word2idx = lookup['word2idx']
    entity2idx = lookup['entity2idx']

    train_data = DataLoader(train_sens, train_ents)
    val_data = DataLoader(val_sens, val_ents)

    # Set number of vocabularies and entities
    arg.num_vocabs = len(word2idx)
    arg.num_entities = len(entity2idx)

    writer = SummaryWriter(arg.event_dir)
    earlystop = EarlyStopping(monitor='acc', min_delta=arg.min_delta, patience=arg.patience)

    model = build_model(arg.model_name, arg).to(arg.device)
    optimizer = optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
    lr_decay = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=arg.lr_decay_factor, verbose=True,
                                                    patience=0, min_lr=arg.min_lr)

    # Load pretrained word embeddings and fixed it
    if arg.use_pretrained_embeddings and all(map(os.path.exists, arg.embedding_paths)):
        pretrained_embeddings = load_embeddings(*arg.embedding_paths)
        model.init_embeddings(pretrained_embeddings, freeze=True)

    # Show model parameters
    # for name, tensor in model.named_parameters():
    #     print('{:<40}{:<15}{}'.format(name, str(list(tensor.shape)), tensor.requires_grad))

    # Train
    print('Model: {}\nStart training'.format(arg.model_name))
    finished_batch = 0

    for epoch in range(1, arg.num_epochs + 1):
        model.train()
        for i, (sens, ents) in enumerate(train_data.gen_batch(arg.batch_size)):
            sens = torch.from_numpy(sens).long().to(arg.device)
            ents = torch.from_numpy(ents).long().to(arg.device)
            loss = model.loss(sens, ents)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            finished_batch += sens.size(0)

            if i % 10 == 0:
                print('[TRAIN] Epoch: {:>3d} Batch: {:4d} Loss: {:.4f}'.format(
                    epoch, (i + 1) * arg.batch_size, loss.item()))

                writer.add_scalar('train_loss', loss.cpu().item(), finished_batch)

        # Save model
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'ckpt_epoch_{:02d}.pt'.format(epoch)))
        if os.path.exists(os.path.join(ckpt_dir, 'ckpt_epoch_{:02d}.pt'.format(epoch - arg.patience - 1))):
            os.remove(os.path.join(ckpt_dir, 'ckpt_epoch_{:02d}.pt'.format(epoch - arg.patience - 1)))

        # Validate per epoch
        model.eval()

        y_true, y_pred = [], []  # true entities, predicted entities
        val_acml_loss = 0  # accumulated loss in this epoch

        for sens, ents in val_data.gen_batch(arg.batch_size * 4, shuffle=False):
            val_size = sens.shape[0]
            sens = torch.from_numpy(sens).long().to(arg.device)
            ents = torch.from_numpy(ents).long().to(arg.device)

            loss = model.loss(sens, ents)

            _, preds = model(sens, ents)
            targets = ents.cpu().detach().numpy()

            y_true.extend([ent for sen in targets for ent in sen if ent != arg.entity_pad[1]])
            y_pred.extend([ent for sen in preds for ent in sen])

            val_acml_loss += loss.item() * val_size

        val_loss = val_acml_loss / len(val_data)
        val_f1 = cal_f1score(y_true, y_pred)

        print('[VAL]   Epoch: {:>3d} Loss: {:.4f} F1-Score: {:.4f}'.format(epoch, val_loss, val_f1))
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_f1', val_f1, epoch)

        lr_decay.step(val_loss)  # learning rate decay

        # Early Stopping
        if earlystop.judge(epoch, val_f1):
            print('Early stop at epoch {}, with val F1-Score {}'.format(epoch, val_f1))
            print('Best perform epoch: {}, with best F1-Score {}'.format(earlystop.best_epoch, earlystop.best_val))
            break

    print('Done')
