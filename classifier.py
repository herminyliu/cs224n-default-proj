import random, numpy as np, argparse
from types import SimpleNamespace
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

from tokenizer import BertTokenizer
from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class BertSentimentClassifier(torch.nn.Module):
    '''
    This module performs sentiment classification using BERT embeddings on the SST dataset.

    In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
    Thus, your forward() should return one logit for each of the 5 classes.
    '''

    def __init__(self, config):
        super(BertSentimentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained(r".\bert-base-uncased-manual-download", local_files_only=True)
        # BertModel inherit from BertPreTrainedModel, the method .from_pretrained is defined in BertPreTrainedModel.

        # Pretrain mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True

        # Create any instance variables you need to classify the sentiment of BERT embeddings.
        ### TODO
        # Convert the final BERT contextualized embedding, the hidden state of [CLS] token to the prediction
        # probabilities vector whose length is self.num_labels classes.
        self.cls_hs_to_logits_proj = torch.nn.Linear(in_features=config.hidden_size, out_features=self.num_labels)
        self.cls_hs_dropout = torch.nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask, config):
        '''Takes a batch of sentences and returns logits for sentiment classes'''
        # The final BERT contextualized embedding is the hidden state of [CLS] token (the first token).
        # HINT: You should consider what is an appropriate return value given that
        # the training loop currently uses F.cross_entropy as the loss function.
        ### TODO
        # The 'config' variable is added in order to retrieve config.max_position_embeddings in bert.Bertmodel.forward
        output = self.bert(input_ids, attention_mask, config)
        # The example of how forward function is used:
        # logits = model(b_ids, b_mask)
        # loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
        # From that we can tell this function should return a single variable
        # The shape of pooler_output and last_hidden_state is
        # [num_batches, batch_size, hidden_size=768(for base bert)]
        # There need a linear projection to proj hidden_size -> self.num_labels

        if output['oversize_flag']:
            return 0, output['oversize_flag']

        output['pooler_output'] = self.cls_hs_dropout(output['pooler_output'])
        logits = self.cls_hs_to_logits_proj(output['pooler_output'])

        # output['oversize_flag'] is a newly added return
        return logits, output['oversize_flag']


class SentimentDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained(r".\bert-base-uncased-manual-download", local_files_only=True)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sents': sents,
            'sent_ids': sent_ids
        }

        return batched_data


class SentimentTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained(r".\bert-base-uncased-manual-download", local_files_only=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'sents': sents,
            'sent_ids': sent_ids
        }

        return batched_data


# Load the data: a list of (sentence, label).
def load_data(filename, flag='train'):
    num_labels = {}
    data = []
    if flag == 'test':
        with open(filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                data.append((sent, sent_id))
    else:
        with open(filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                data.append((sent, label, sent_id))
        print(f"load {len(data)} data from {filename}")

    if flag == 'train':
        return data, len(num_labels)
    else:
        return data


# Evaluate the model on train/dev examples. Used in following train and test function
def model_eval(dataloader, model, device, config):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_true = []
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
            batch['labels'], batch['sents'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits, oversize_flag = model(b_ids, b_mask, config)
        if oversize_flag:
            # disregard this batch
            continue
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sents, sent_ids


# Evaluate the model on test examples. Used in the following test function.
def model_test_eval(dataloader, model, device, config):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'], \
            batch['sents'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits, oversize_flag = model(b_ids, b_mask, config)
        if oversize_flag:
            # disregard this batch
            continue
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    return y_pred, sents, sent_ids


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    train_data, num_labels = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')

    train_dataset = SentimentDataset(train_data, args)
    dev_dataset = SentimentDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    # Init model.
    # max_position_embeddings is newly added, the value 512 is retirved from config.py line 196
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode,
              'max_position_embeddings': 512}

    config = SimpleNamespace(**config)

    model = BertSentimentClassifier(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        # debug_slice = 82
        # debug_count = 0
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # if debug_count < debug_slice:
            #     debug_count = debug_count + 1
            #     continue
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()

            logits, oversize_flag = model(b_ids, b_mask, config)
            if oversize_flag:
                # disregard this batch
                continue
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        # NOTE: The dev dataset and validation set are the same thing
        # Here not only do the evaluation on training set but also on the dev set.
        train_acc, train_f1, *_ = model_eval(train_dataloader, model, device, config)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device, config)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(
            f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


def test(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']
        model = BertSentimentClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")

        dev_data = load_data(args.dev, 'valid')
        dev_dataset = SentimentDataset(dev_data, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=dev_dataset.collate_fn)

        test_data = load_data(args.test, 'test')
        test_dataset = SentimentTestDataset(test_data, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=test_dataset.collate_fn)

        dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(dev_dataloader, model, device, config)
        print('DONE DEV')
        test_pred, test_sents, test_sent_ids = model_test_eval(test_dataloader, model, device, config)
        print('DONE Test')
        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sent_ids, dev_pred):
                f.write(f"{p} , {s} \n")

        with open(args.test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sent_ids, test_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-3)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)

    print('Training Sentiment Classifier on SST...')
    config = SimpleNamespace(
        filepath='sst-classifier.pt',
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train='data/ids-sst-train.csv',
        dev='data/ids-sst-dev.csv',
        test='data/ids-sst-test-student.csv',
        fine_tune_mode=args.fine_tune_mode,
        dev_out = 'predictions/' + args.fine_tune_mode + '-sst-dev-out.csv',
        test_out = 'predictions/' + args.fine_tune_mode + '-sst-test-out.csv'
    )

    train(config)

    print('Evaluating on SST...')
    test(config)

    print('Training Sentiment Classifier on cfimdb...')
    config = SimpleNamespace(
        filepath='cfimdb-classifier.pt',
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=8,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train='data/ids-cfimdb-train.csv',
        dev='data/ids-cfimdb-dev.csv',
        test='data/ids-cfimdb-test-student.csv',
        fine_tune_mode=args.fine_tune_mode,
        dev_out = 'predictions/' + args.fine_tune_mode + '-cfimdb-dev-out.csv',
        test_out = 'predictions/' + args.fine_tune_mode + '-cfimdb-test-out.csv'
    )

    train(config)

    print('Evaluating on cfimdb...')
    test(config)
