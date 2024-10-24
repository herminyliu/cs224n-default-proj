# CS 224N Default Final Project - Multitask BERT - From Coursework Team

This is the default final project for the Stanford CS 224N class. Please refer to the project handout on the course website for detailed instructions and an overview of the codebase.

This project comprises two parts. In the first part, you will implement some important components of the BERT model to better understand its architecture. 
In the second part, you will use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection, and semantic similarity. You will implement extensions to improve your model's performance on the three downstream tasks.

In broad strokes, Part 1 of this project targets:
* bert.py: Missing code blocks.
* classifier.py: Missing code blocks.
* optimizer.py: Missing code blocks.

And Part 2 targets:
* multitask_classifier.py: Missing code blocks.
* datasets.py: Possibly useful functions/classes for extensions.
* evaluation.py: Possibly useful functions/classes for extensions.

## Setup instructions

Follow `setup.sh` to properly setup a conda environment and install dependencies.

## Acknowledgement

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).




# CS 224N Default Final Project - My Implementation

## Part 1: Implementation of the Multi-head Self-attention and Transformer Layers of the Original BERT Model

*base_bert.py* should not be modified, since it provides the class *BertPreTrainedModel* which the BertModel class in *bert.py* inherited this.

In *bert.py*, there are three classes together formed the miniBERT structure.

- BertSelfAttention: Construct the attention layer inside a BertLayer.

- BertLayer: Construct a single bert layer.

- BertModel: The whole miniBERT model which is made up of many BertLayers.

## Part 2: Self-defined Adam Optimizer Implementation

In *optimizer.py*, a self-defined AdamW class is defined and used as the optimizer in *classifier.py*.

## Part 3: Perform Sentiment Classification Based on Finished Works

In *classifier.py*, Sentiment Classification is performed using the model and optimizer implemented before.

The accuracy my model obtained using the developing set for two fine tune modes {full-model, last-linear-layer} on the dataset {SST, CFIMDB}

> full-model mode: require gradient computation and adjustment in bert model
> 
> last-linear-layer mode: don’t require gradient computation in bert model

- Fine-tuning the last linear layer for SST: Dev Accuracy : **0.396**
- Fine-tuning the last linear layer for CFIMDB: Dev Accuracy : **0.804**
- Fine-tuning the full model for SST: Dev Accuracy : **0.526**
- Fine-tuning the full model for CFIMDB: Dev Accuracy : **0.955**

Note: 

1. To avoid file encoding problem, in classifier.load_data we set the file opening using 'utf-8'.

2. In mainland China hugging face can not be visited without VPN. Even when I use a VPN and switched it to the global mode, there is still something wrong with the Internet connection on my machine, so I manually download pytorch_model.bin, vocab.txt and config.json from https://huggingface.co/google-bert/bert-base-uncased/tree/main and place it in the folder ./bert-base-uncased-manual-download, and set local_files_only to be True. Files under the folder ./bert-base-uncased-manual-download is not tracked by git.

>If you prefer to manually download the files like me, remember to download the files and change the file path before you run this code. 

```
self.bert = BertModel.from_pretrained(r".\path\to\your\downloaded\files", local_files_only=True)
self.tokenizer = BertTokenizer.from_pretrained(r".\path\to\your\downloaded\files", local_files_only=True)
```

>If you prefer to download automatically, remember to set local_files_only back to the default value False, and change the local file path to the model name 'bert-base-uncased'.

```
self.bert = BertModel.from_pretrained('bert-base-uncased')
self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```

3. When running the following command on the CFIMDB dev dataset(./data/ids-cfimdb-dev.csv), an unexpected shape mismatch error occurred. 

```
python3 classifier.py --fine-tune-mode {full-model,last-linear-layer} --lr {1e-3,1e-5}
```
One of the batch whose seq length is 514 which is larger than 512, the max_position_embeddings. Thus, the position embedding length for that batch is still 512 not 514. Besides, the pretrained BERT model only can handle the seq no longer than 512. However, the input embedding and token embedding length for that batch is still 514. In bert.BertModel, at the start of the each sublayer, position embedding, input embedding and token embedding will be added together. As a result, a shape mismatch error occurred.

The max_position_embeddings is set in `config.BertConfig.__init__`. Making the value larger will lead to mismatch between the retrained model params.

I added a few lines to disregard the batches whose seq length is larger than 512 in `classifier.train`, `classifier.model_eval`, `classifier.model_test_eval`, `bert.BertModel.forward`, `classifier.BertSentimentClassifier.forward`.

## Part 4: Free Extension on Multitasks

This part has not implemented for now.
