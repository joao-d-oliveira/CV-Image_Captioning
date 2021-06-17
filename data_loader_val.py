import os
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
import numpy as np
import json


def get_val_loader(transform,
               mode='val',
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               cocoapi_loc='/opt'):
    """Returns the data loader.
    Args:
      transform: Image transform.
      mode: One of 'train' or 'test'.
      batch_size: Batch size (if in testing mode, must have batch_size=1).
      vocab_threshold: Minimum word count threshold.
      vocab_file: File containing the vocabulary. 
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
      vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading 
      cocoapi_loc: The location of the folder containing the COCO API: https://github.com/cocodataset/cocoapi
    """
    
    assert mode in ['val'], "mode must be 'val' if you want to 'train' or 'test', please use get_val_loader from data_loader.py"
    assert vocab_from_file==True, "To generate vocab from captions file, must be in training mode (mode='train')."
    # Based on mode (train, val, test), obtain img_folder and annotations_file.
    assert batch_size==1, "Please change batch_size to 1 if validating your model."
    assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
    assert vocab_from_file==True, "Change vocab_from_file to True."
    annotationsFileLOC = 'cocoapi/annotations/captions_val2014.json'
    img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/val2014/')
    annotations_file = os.path.join(cocoapi_loc, annotationsFileLOC)

    # COCO caption dataset.
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=dataset.batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    return data_loader


class CoCoDataset(data.Dataset):
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word,
                 end_word, unk_word, annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
                                end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder

        val_data = json.loads(open(annotations_file).read())
        img_data = val_data['images']
        ann_data = val_data['annotations']
        self.dict_images = {f['id']:f['file_name'] for f in img_data}
        self.ids = [f['id'] for f in img_data]
        self.dict_captions = {f['image_id']:f['caption'] for f in ann_data}

    def __getitem__(self, index):
        # obtain image and caption if in training mode
        idImage = self.ids[index]
        path = self.dict_images[idImage]
        caption = self.dict_captions[idImage]

        # Convert image to tensor and pre-process using transform
        PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        orig_image = np.array(PIL_image)
        image = self.transform(PIL_image)

        # return original image and pre-processed image tensor
        return idImage, orig_image, image, caption

    def __len__(self):
        return len(self.ids)
