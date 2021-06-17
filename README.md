# About
Project from Udacity. <br>
Final project from 3rd Section 
Original [GitHub project](https://github.com/udacity/CVND---Image-Captioning-Project)


# Aproach
* Started with a simple DecoderRNN of: 1 Embedding Layer -> LSTM -> Linear
* Tried to add an attention layer after LSTM: 1 Embedding Layer -> LSTM -> MultiHeadAttention -> Linear
* Tried to add an attention layer before LSTM: 1 Embedding Layer -> MultiHeadAttention -> LSTM -> Linear
* Found in the internet literature on best case performances and how they did it. Particularly found [GitHub Example](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) as well as [this medium tutorial](https://medium.com/analytics-vidhya/image-captioning-with-attention-part-1-e8a5f783f6d3)
* Compared performance of all models with an excelent example from: [here](https://medium.com/analytics-vidhya/image-captioning-with-attention-part-1-e8a5f783f6d3)

# Instructions

## Submission Files
* :white_check_mark:`models.py`
* :white_check_mark:`1_Preliminaries.ipynb`
* :white_check_mark:`2_Training.ipynb`
* :white_check_mark:`3_Inference.ipynb`

## Project Rubric [link_original](https://review.udacity.com/#!/rubrics/1427/view)

### `models.py`

#### Specify the CNNEncoder and RNNDecoder
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------:| 
|  :white_check_mark: `CNNEncoder`. |  The chosen CNN architecture in the `CNNEncoder` class in **model.py** makes sense as an encoder for the image captioning task.|
| :white_check_mark: `RNNDecoder`. |  The chosen RNN architecture in the `RNNDecoder` class in **model.py** makes sense as a decoder for the image captioning task.|


### 2_Training.ipynb

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------:| 
| :white_check_mark:Using the Data Loader |  When using the `get_loader` function in **data_loader.py** to train the model, most arguments are left at their default values, as outlined in **Step 1** of **1_Preliminaries.ipynb**. In particular, the submission only (optionally) changes the values of the following arguments: `transform`, `mode`, `batch_size`, `vocab_threshold`, `vocab_from_file`. |
| :white_check_mark:Step 1, Question 1 |  The submission describes the chosen CNN-RNN architecture and details how the hyperparameters were selected. |
| :white_check_mark:Step 1, Question 2 |  The transform is congruent with the choice of CNN architecture. If the transform has been modified, the submission describes how the transform used to pre-process the training images was selected.|
| :white_check_mark:Step 1, Question 3 |  The submission describes how the trainable parameters were selected and has made a well-informed choice when deciding which parameters in the model should be trainable.|
| :white_check_mark:Step 1, Question 4 |  The submission describes how the optimizer was selected.|
| :white_check_mark:Step 2 |  The code cell in **Step 2** details all code used to train the model from scratch. The output of the code cell shows exactly what is printed when running the code cell. If the submission has amended the code used for training the model, it is well-organized and includes comments.|

### 3_Inference.ipynb

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------:| 
| :white_check_mark:`transform_test` |  The transform used to pre-process the test images is congruent with the choice of CNN architecture. It is also consistent with the transform specified in `transform_train` in **2_Training.ipynb**.| 
| :white_check_mark:Step 3 | The implementation of the `sample` method in the `RNNDecoder` class correctly leverages the RNN to generate predicted token indices.| 
| :white_check_mark:Step 4 | The `clean_sentence` function passes the test in **Step 4**. The sentence is reasonably clean, where any `<start>` and `<end>` tokens have been removed.| 
| :white_check_mark: Step 5 | The submission shows two image-caption pairs where the model performed well, and two image-caption pairs where the model did not perform well.| 

## Bonus :boom::boom::boom:
* :white_check_mark: Use the validation set to guide your search for appropriate hyperparameters.
* :white_check_mark: Tinker with your model - and train it for long enough - to obtain results that are comparable to (or surpass!) recent research articles
* ❗ Implement beam search to generate captions on new images.

# Results

## Approach
From the varios rounds performed, I evaluated 3 different runs:
`v102_paramsDecoder_withoutFlipTrans_batch_size10_vocabThr3_embedSize512_hiddenSize1024_totEpochs10
v121_paramsDecoder_withoutFlipTrans_batch_size10_vocabThr3_embedSize512_hiddenSize1024_totEpochs3
v120_paramsDecoder_withoutFlipTrans_batch_size10_vocabThr3_embedSize512_hiddenSize1024_totEpochs3`

All using the benchmark of the model found [here](https://medium.com/analytics-vidhya/image-captioning-with-attention-part-1-e8a5f783f6d3)
`v120_paramsDecoder_withoutFlipTrans_batch_size10_vocabThr3_embedSize512_hiddenSize1024_totEpochs3`

## Final Results
**Benchmark** (`v200_paramsDecoder_withoutFlipTrans_batch_size10_vocabThr4_embedSize512_hiddenSize1024_totEpochs10`)
* Attention: 2 epochs: 
> ratio: 0.9943724833842694; Bleu_1: 0.679; Bleu_2: 0.496; Bleu_3: 0.344; Bleu_4: 0.232; METEOR: 0.211; ROUGE_L: 0.491; CIDEr: 0.639
* Attention: 3 epochs: 
> ratio: 1.0094033452608777; Bleu_1: 0.659; Bleu_2: 0.484; Bleu_3: 0.339; Bleu_4: 0.232; ; METEOR: 0.214; ROUGE_L: 0.488; CIDEr: 0.640

**Best and most Simple** (`v102_paramsDecoder_withoutFlipTrans_batch_size10_vocabThr3_embedSize512_hiddenSize1024_totEpochs10`)
* my simple model - 1 epochs:
> ratio: 1.009220942322141; Bleu_1: 0.587; Bleu_2: 0.383; Bleu_3: 0.244; Bleu_4: 0.157; METEOR: 0.182; ROUGE_L: 0.424; CIDEr: 0.507
* my simple model - 2 epochs:
> ratio: 1.0005286538264253; Bleu_1: 0.616; Bleu_2: 0.416; Bleu_3: 0.275; Bleu_4: 0.183; METEOR: 0.192; ROUGE_L: 0.442; CIDEr: 0.574
* my simple model - 3 epochs:
> ratio: 1.014850954868193; Bleu_1: 0.637; Bleu_2: 0.449; Bleu_3: 0.306; Bleu_4: 0.209; METEOR: 0.211; ROUGE_L: 0.466; CIDEr: 0.667
* my simple model - 4 epochs:
>
* my simple model - 5 epochs:
>

**Attention Before LSTM** (`v121_paramsDecoder_withoutFlipTrans_batch_size10_vocabThr3_embedSize512_hiddenSize1024_totEpochs3`)
* my att first model - 1 epochs:
> ratio: 1.102866731062698; Bleu_1: 0.357; Bleu_2: 0.179; Bleu_3: 0.065; Bleu_4: 0.029; METEOR: 0.091; ROUGE_L: 0.285; CIDEr: 0.037
* my att first model - 2 epochs:
> ratio: 0.706783945336399; Bleu_1: 0.205; Bleu_2: 0.091; Bleu_3: 0.032; Bleu_4: 0.014; METEOR: 0.050; ROUGE_L: 0.174; CIDEr: 0.015

**Attention After LSTM** (`v120_paramsDecoder_withoutFlipTrans_batch_size10_vocabThr3_embedSize512_hiddenSize1024_totEpochs3`)
* my att aft model - 1 epochs:
> ratio: 0.33351825744295616; Bleu_1: 0.052; Bleu_2: 0.034; Bleu_3: 0.021; Bleu_4: 0.014; METEOR: 0.037; ROUGE_L: 0.101; CIDEr: 0.071
* my att aft model - 2 epochs:
> ratio: 0.21256804304037424; Bleu_1: 0.005; Bleu_2: 0.005; Bleu_3: 0.004; Bleu_4: 0.002; METEOR: 0.014; ROUGE_L: 0.034; CIDEr: 0.029

## Thoughts
My best model is clearly 1.0.2 (although simple), proved to be better than adding a multihead attention.
And performance wise is close to the most complicated one used to benchmark.
