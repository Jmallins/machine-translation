import argparse
import math
import logging
import os
import random
import torch
import torch.nn as nn

from tqdm import tqdm
from seq2seq import models, utils
from seq2seq.data.dictionary import Dictionary
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler
from seq2seq.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY


def get_args():
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--cuda', default=False, help='Use a GPU')

    # Add data arguments
    parser.add_argument('--data', default='data-bin', help='path to data directory')
    parser.add_argument('--source-lang', default=None, help='source language')
    parser.add_argument('--target-lang', default=None, help='target language')
    parser.add_argument('--max-tokens', default=16000, type=int, help='maximum number of tokens in a batch')
    parser.add_argument('--batch-size', default=None, type=int, help='maximum number of sentences in a batch')

    # Add model arguments
    parser.add_argument('--arch', default='lstm', choices=ARCH_MODEL_REGISTRY.keys(), help='model architecture')

    # Add optimization arguments
    parser.add_argument('--max-epoch', default=100, type=int, help='force stop training at specified epoch')
    parser.add_argument('--clip-norm', default=0.1, type=float, help='clip threshold of gradients')
    parser.add_argument('--lr', default=0.025, type=float, help='learning rate')

    # Add checkpoint arguments
    parser.add_argument('--log-file', default=None, help='path to save logs')
    parser.add_argument('--save-dir', default='checkpoints', help='path to save checkpoints')
    parser.add_argument('--restore-file', default='checkpoint_last.pt', help='filename to load checkpoint')
    parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')
    parser.add_argument('--no-save', action='store_true', help='don\'t save models or checkpoints')
    parser.add_argument('--epoch-checkpoints', action='store_true', help='store all epoch checkpoints')

    # Parse twice as model arguments are not known the first time
    args, _ = parser.parse_known_args()
    model_parser = parser.add_argument_group(argument_default=argparse.SUPPRESS)
    ARCH_MODEL_REGISTRY[args.arch].add_args(model_parser)
    args = parser.parse_args()
    ARCH_CONFIG_REGISTRY[args.arch](args)
    return args


def main(args):
    print ("Training")
    torch.manual_seed(42)

    utils.init_logging(args)
    



    # Load dictionaries
    src_dict = Dictionary.load(os.path.join(args.data, 'dict.{}'.format(args.source_lang)))
    logging.info('Loaded a source dictionary ({}) with {} words'.format(args.source_lang, len(src_dict)))
    tgt_dict = Dictionary.load(os.path.join(args.data, 'dict.{}'.format(args.target_lang)))
    logging.info('Loaded a target dictionary ({}) with {} words'.format(args.target_lang, len(tgt_dict)))

    # Load datasets
    def load_data(split):
        return Seq2SeqDataset(
            src_file=os.path.join(args.data, '{}.{}'.format(split, args.source_lang)),
            tgt_file=os.path.join(args.data, '{}.{}'.format(split, args.target_lang)),
            src_dict=src_dict, tgt_dict=tgt_dict)
    train_dataset = load_data(split='train')
    valid_dataset = load_data(split='valid')

    # Build model and criterion
    model = models.build_model(args, src_dict, tgt_dict)
    logging.info('Built a model with {} parameters'.format(sum(p.numel() for p in model.parameters())))
    criterion = nn.CrossEntropyLoss(ignore_index=src_dict.pad_idx, reduction='sum')
    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # Build an optimizer and a learning rate schedule
    optimizer = torch.optim.Adam(model.parameters(), args.lr)


    # Load last checkpoint if one exists
    state_dict = utils.load_checkpoint(args, model, optimizer)#lr_scheduler
    last_epoch = state_dict['last_epoch'] if state_dict is not None else -1

    for epoch in range(last_epoch + 1, args.max_epoch):
        train_loader = torch.utils.data.DataLoader(
            train_dataset, num_workers=1, collate_fn=train_dataset.collater,
            batch_sampler=BatchSampler(
                train_dataset, args.max_tokens, args.batch_size, 1,
                0, shuffle=True, seed=42))

        model.train()
        stats = {'loss': 0., 'lr': 0., 'num_tokens': 0., 'batch_size': 0., 'grad_norm': 0., 'clip': 0.}
        ##display 
        progress_bar = tqdm(train_loader, desc='| Epoch {:03d}'.format(epoch), leave=False, disable=False)

        for i, sample in enumerate(progress_bar):
            if args.cuda:
                sample = utils.move_to_cuda(sample)
            if len(sample) == 0:
                continue

            # Forward and backward pass
            output, _ = model(sample['src_tokens'], sample['src_lengths'], sample['tgt_inputs'])
            loss = criterion(output.view(-1, output.size(-1)), sample['tgt_tokens'].view(-1))
            optimizer.zero_grad()
            loss.backward()

       
            total_loss, num_tokens, batch_size = loss.item(), sample['num_tokens'], len(sample['src_tokens'])
            ## Clip gradients 
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            # Update statistics for progress bar
            stats['loss'] += total_loss / num_tokens / math.log(2)
            stats['lr'] += optimizer.param_groups[0]['lr']
            stats['num_tokens'] += num_tokens / len(sample['src_tokens'])
            stats['batch_size'] += batch_size
            stats['grad_norm'] += grad_norm
            stats['clip'] += 1 if grad_norm > args.clip_norm else 0
            progress_bar.set_postfix({key: '{:.4g}'.format(value / (i + 1)) for key, value in stats.items()}, refresh=True)

        logging.info('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.4g}'.format(
            value / len(progress_bar)) for key, value in stats.items())))

     
        valid_loss = validate(args, model, criterion, valid_dataset, epoch)

        # Save checkpoints
        if epoch % args.save_interval == 0:
            utils.save_checkpoint(args, model, optimizer, epoch, valid_loss)#lr_scheduler



def validate(args, model, criterion, valid_dataset, epoch):
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, num_workers=1, collate_fn=valid_dataset.collater,
        batch_sampler=BatchSampler(
            valid_dataset, args.max_tokens, args.batch_size, 1,
            0, shuffle=True, seed=42))

    model.eval()
    stats = {'valid_loss': 0, 'num_tokens': 0, 'batch_size': 0}
    progress_bar = tqdm(valid_loader, desc='| Epoch {:03d}'.format(epoch), leave=False)

    for i, sample in enumerate(progress_bar):
        if args.cuda:
            sample = utils.move_to_cuda(sample)
        if len(sample) == 0:
            continue
        with torch.no_grad():
            output, attn_scores = model(sample['src_tokens'], sample['src_lengths'], sample['tgt_inputs'])
            loss = criterion(output.view(-1, output.size(-1)), sample['tgt_tokens'].view(-1))
        total_loss, num_tokens, batch_size = loss.item(), sample['num_tokens'], len(sample['src_tokens'])
        stats['valid_loss'] += loss.item()  / num_tokens / math.log(2)
        stats['num_tokens'] += sample['num_tokens'] / len(sample['src_tokens'])
        stats['batch_size'] += len(sample['src_tokens'])
        progress_bar.set_postfix({key: '{:.3g}'.format(value / (i + 1)) for key, value in stats.items()}, refresh=True)

    logging.info('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.3g}'.format(
        value / len(progress_bar)) for key, value in stats.items())))
    return stats['valid_loss'] / len(progress_bar)


if __name__ == '__main__':
    args = get_args()
    print ("Main method")
    args.device_id = 0
    main(args)

