import warnings
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import build_transformer
from torch.utils.tensorboard import SummaryWriter
from config import get_weights_file_path, get_config
from dataset import BilingualDataset, causal_mask


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
        ], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device,
                   print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = 80
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            model_out = greedy_decode(model, encoder_input, encoder_mask,
                                      tokenizer_src, tokenizer_tgt, max_len, device)

            source_txt = batch['src_text'][0]
            target_txt = batch['tgt_text'][0]
            model_out_txt = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_txt)
            expected.append(target_txt)
            predicted.append(model_out_txt)

            print_msg('-' * console_width)
            print_msg(f'SOURCE:     {source_txt}')
            print_msg(f'TARGET:     {target_txt}')
            print_msg(f'PREDICTION: {model_out_txt}')

            if count == num_examples:
                break


def load_dataset(src_file: str, tgt_file: str):
    with open(src_file, 'r', encoding='utf-8') as src_f, open(tgt_file, 'r', encoding='utf-8') as tgt_f:
        src_lines = src_f.readlines()
        tgt_lines = tgt_f.readlines()

    assert len(src_lines) == len(tgt_lines), f"Mismatched line counts: {len(src_lines)} vs {len(tgt_lines)}"
    return [(src.strip(), tgt.strip()) for src, tgt in zip(src_lines, tgt_lines)]


def get_or_build_tokenizer(config, dataset, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not tokenizer_path.exists():
        print(f"Building tokenizer for {lang}...")
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )

        lines = (pair[0] if lang == config['lang_src'] else pair[1] for pair in dataset)
        tokenizer.train_from_iterator(lines, trainer=trainer)
        tokenizer.save(str(tokenizer_path))
        print(f"Tokenizer saved to {tokenizer_path}")
    else:
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):
    train = load_dataset('../dataset/parallel-n/IITB.en-hi.en', '../dataset/parallel-n/IITB.en-hi.hi')
    test = load_dataset('../dev_test/dev_test/test.en', '../dev_test/dev_test/test.hi')

    print(f"Loaded {len(train)} training pairs and {len(test)} test pairs")

    tokenizer_src = get_or_build_tokenizer(config, train, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, train, config['lang_tgt'])

    print(f"Source vocab size: {tokenizer_src.get_vocab_size()}")
    print(f"Target vocab size: {tokenizer_tgt.get_vocab_size()}")

    train_formatted = []
    for src, tgt in train:
        train_formatted.append({
            'translation': {
                config['lang_src']: src,
                config['lang_tgt']: tgt
            }
        })

    test_formatted = []
    for src, tgt in test:
        test_formatted.append({
            'translation': {
                config['lang_src']: src,
                config['lang_tgt']: tgt
            }
        })

    train_ds = BilingualDataset(
        train_formatted, tokenizer_src, tokenizer_tgt,
        config['lang_src'], config['lang_tgt'], config['seq_len']
    )
    test_ds = BilingualDataset(
        test_formatted, tokenizer_src, tokenizer_tgt,
        config['lang_src'], config['lang_tgt'], config['seq_len']
    )

    max_len_src = 0
    max_len_tgt = 0
    for src, tgt in train:
        src_ids = tokenizer_src.encode(src).ids
        tgt_ids = tokenizer_tgt.encode(tgt).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True)

    return train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len, vocab_tgt_len,
        config['seq_len'], config['seq_len'],
        config['d_model']
    )
    return model


def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    num_params = count_parameters(model)
    print(f"Total Trainable Parameters: {num_params:,}")

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        print(f"Resumed from epoch {initial_epoch}")

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id('[PAD]'),
        label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        print(f"\nEpoch {epoch:02d}/{config['num_epochs'] - 1}")
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Training epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1)
            )

            batch_iterator.set_postfix({"Loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        run_validation(
            model, test_dataloader, tokenizer_src, tokenizer_tgt,
            config['seq_len'], device, print, global_step, writer
        )

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        print(f"Model saved to {model_filename}")

    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    config = get_config()
    train_model(config)