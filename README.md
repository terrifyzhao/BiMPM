# BIMPM-pytorch
Re-implementation of [BIMPM](https://arxiv.org/abs/1702.03814)(Bilateral Multi-Perspective Matching for Natural Language Sentences, Zhiguo Wang et al., IJCAI 2017) on Pytorch

## Results

Dataset: [SNLI](https://nlp.stanford.edu/projects/snli/)

| Model        |  ACC(%)   | 
|--------------|:----------:|
| **Re-implementation** |			 **86.5** |  
| Baseline from the paper (Single BiMPM)	|  86.9    |    

Dataset: [Quora](https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view) (provided by the authors)

| Model        |  ACC(%)   | 
|--------------|:----------:|
| **Re-implementation** 			| **87.3** |  
| Baseline from the paper (Single BiMPM)     	|  88.17   |

Note: I could not observe much gain from character embeddings. The implementation for the character embedding might be too naive to improve the performance.

## Development Environment
- OS: Ubuntu 16.04 LTS (64bit)
- Language: Python 3.6.2.
- Pytorch: 0.3.0

## Requirements

Please install the following library requirements specified in the **requirements.txt** first.

    nltk==3.2.4
    torchtext==0.2.0
    torch==0.3.0
    tensorboardX==0.8

## Training

> python train.py --help

	usage: train.py [-h] [--batch-size BATCH_SIZE] [--char-dim CHAR_DIM]
                [--char-hidden-size CHAR_HIDDEN_SIZE] [--data-type DATA_TYPE]
                [--dropout DROPOUT] [--epoch EPOCH] [--gpu GPU]
                [--hidden-size HIDDEN_SIZE] [--learning-rate LEARNING_RATE]
                [--max-sent-len MAX_SENT_LEN]
                [--num-perspective NUM_PERSPECTIVE] [--print-freq PRINT_FREQ]
                [--use-char-emb] [--word-dim WORD_DIM]

    optional arguments:
      -h, --help                    show this help message and exit
      --batch-size BATCH_SIZE
      --char-dim CHAR_DIM
      --char-hidden-size CHAR_HIDDEN_SIZE
      --data-type DATA_TYPE         available: SNLI or Quora
      --dropout DROPOUT
      --epoch EPOCH
      --gpu GPU
      --hidden-size HIDDEN_SIZE
      --learning-rate LEARNING_RATE
      --max-sent-len MAX_SENT_LEN   max length of input sentences model can accept, if -1,
                                    it accepts any length
      --num-perspective NUM_PERSPECTIVE
      --print-freq PRINT_FREQ
      --use-char-emb
      --word-dim WORD_DIM


## Test

> python test.py --help

	usage: test.py [-h] [--batch-size BATCH_SIZE] [--char-dim CHAR_DIM]
               [--char-hidden-size CHAR_HIDDEN_SIZE] [--dropout DROPOUT]
               [--data-type DATA_TYPE] [--epoch EPOCH] [--gpu GPU]
               [--hidden-size HIDDEN_SIZE] [--learning-rate LEARNING_RATE]
               [--num-perspective NUM_PERSPECTIVE] [--use-char-emb]
               [--word-dim WORD_DIM] --model-path MODEL_PATH

    optional arguments:
      -h, --help                    show this help message and exit
      --batch-size BATCH_SIZE
      --char-dim CHAR_DIM
      --char-hidden-size CHAR_HIDDEN_SIZE
      --dropout DROPOUT
      --data-type DATA_TYPE         available: SNLI or Quora
      --epoch EPOCH
      --gpu GPU
      --hidden-size HIDDEN_SIZE
      --learning-rate LEARNING_RATE
      --num-perspective NUM_PERSPECTIVE
      --use-char-emb
      --word-dim WORD_DIM
      --model-path MODEL_PATH

	
Note: You should execute **test.py** with the same hyperparameters, which are used for training the model you want to run.    
