import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNNv101(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        

        # Embedding Layer: transform captions into embeded_size 
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # LSTM Layer: Do the magic of finding the next word
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        
        # convert output from LSTM into predictions for each word in vocab
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        embed = self.embedding(captions)
        
        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)
        
        # Initialize the hidden state
        batch_size = features.shape[0] # features is of shape (batch_size, embed_size)
        lstm_outputs, _ = self.lstm(embed, self.init_hidden(batch_size))
        out = self.linear(lstm_outputs)
        
        return out
    
    
    def sample(self, features, states=None, end_word = 1, max_len=20):
        output_ids = []
        inputs = features.unsqueeze(1)

        for i in range(max_len):
            # pass data through recurrent network
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))

            # find maximal predictions
            predicted = outputs.max(1)[1]

            # append results from given step to global results
            output_ids.append(predicted.cpu().numpy()[0].item())

            # prepare chosen words for next decoding step
            inputs = self.embedding(predicted)
            inputs = inputs.unsqueeze(1)
            if predicted == end_word : break #arrived to the end of the word

        return output_ids
    
    def init_hidden(self, batch_size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (torch.zeros((1, batch_size, self.hidden_size), device=device), \
                torch.zeros((1, batch_size, self.hidden_size), device=device))


class DecoderRNNv102(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # Embedding Layer: transform captions into embeded_size 
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # LSTM Layer: Do the magic of finding the next word
        self.lstm_hc = (nn.Linear(self.embed_size, self.hidden_size), nn.Linear(self.embed_size, self.hidden_size))
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        
        # convert output from LSTM into predictions for each word in vocab
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        embed = self.embedding(captions)
        
        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)
        
        # Initialize the hidden state
        batch_size = features.shape[0] # features is of shape (batch_size, embed_size)
        lstm_outputs, _ = self.lstm(embed, self.init_hd_hidden(self.lstm_hc[0], self.lstm_hc[1], features))
        out = self.linear(lstm_outputs)
        
        return out
    
    
    def sample(self, features, states=None, end_word = 1, max_len=20):
        output_ids = []
        inputs = features.unsqueeze(1)

        for i in range(max_len):
            # pass data through recurrent network
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))

            # find maximal predictions
            predicted = outputs.max(1)[1]

            # append results from given step to global results
            output_ids.append(predicted.cpu().numpy()[0].item())

            # prepare chosen words for next decoding step
            inputs = self.embedding(predicted)
            inputs = inputs.unsqueeze(1)
            if predicted == end_word : break #arrived to the end of the word

        return output_ids
    
    def init_hd_hidden(self, h, c, features):

        """Initializes hidden state and cell memory using average feature vector.
        Arguments:
        ----------
        - features - features returned from Encoder
        Retruns:
        ----------
        - h0 - initial hidden state (short-term memory)
        - c0 - initial cell state (long-term memory)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mean_annotations = torch.mean(features, dim = 1).to(device)
#        print('gothere 1.1', mean_annotations.is_cuda)
#        print(features.shape)
#        print(mean_annotations.shape)
        if torch.cuda.is_available():
            h = h.cuda()
            c = c.cuda()
        
#        print(h)
        h0 = h(features).unsqueeze(0)
#        print('gothere 1.2')        
        c0 = c(features).unsqueeze(0)
        return h0, c0
    

class DecoderRNNv110(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, num_heads=8):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Embedding Layer: transform captions into embeded_size 
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Get the focus from features where it should
        self.attention = nn.MultiheadAttention(embed_size, num_heads)
        
        # LSTM Layer: Do the magic of finding the next word
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        
        # convert output from LSTM into predictions for each word in vocab
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):

        captions_ = captions[:, :-1] # cut last

        embed = self.embedding(captions_)
        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)

        att_out, _ = self.attention(embed,embed,embed)
        
        # Initialize the hidden state
        batch_size = features.shape[0] 
        lstm_outputs, _ = self.lstm(att_out, self.init_hidden(batch_size, self.hidden_size))
        out = self.linear(lstm_outputs)
        
        return out
    
    
    def sample(self, features, states=None, end_word = 1, max_len=20):
        output_ids = []
        inputs = features.unsqueeze(1)

        for i in range(max_len):
            # pass data through recurrent network
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))

            # find maximal predictions
            predicted = outputs.max(1)[1]

            # append results from given step to global results
            output_ids.append(predicted.cpu().numpy()[0].item())

            # prepare chosen words for next decoding step
            inputs = self.embedding(predicted)
            inputs = inputs.unsqueeze(1)
            if predicted == end_word : break #arrived to the end of the word

        return output_ids
    
    def init_hidden(self, batch_size, size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (torch.zeros((1, batch_size, size), device=device), \
                torch.zeros((1, batch_size, size), device=device))


class DecoderRNNv120(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, num_heads=8):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Embedding Layer: transform captions into embeded_size 
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Get the focus from features where it should
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        
        # LSTM Layer: Do the magic of finding the next word
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        
        # convert output from LSTM into predictions for each word in vocab
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):

        captions_ = captions[:, :-1] # cut last

        embed = self.embedding(captions_)
        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)

        # Initialize the hidden state
        batch_size = features.shape[0] 
        lstm_outputs, _ = self.lstm(embed, self.init_hidden(batch_size, self.hidden_size))

        att_out, _ = self.attention(lstm_outputs, lstm_outputs, lstm_outputs)

        out = self.linear(att_out)
        
        return out
    
    
    def sample(self, features, states=None, end_word = 1, max_len=20):
        output_ids = []
        inputs = features.unsqueeze(1)

        for i in range(max_len):
            # pass data through recurrent network
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))

            # find maximal predictions
            predicted = outputs.max(1)[1]

            # append results from given step to global results
            output_ids.append(predicted.cpu().numpy()[0].item())

            # prepare chosen words for next decoding step
            inputs = self.embedding(predicted)
            inputs = inputs.unsqueeze(1)
            if predicted == end_word : break #arrived to the end of the word

        return output_ids
    
    def init_hidden(self, batch_size, size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (torch.zeros((1, batch_size, size), device=device), \
                torch.zeros((1, batch_size, size), device=device))

    
class DecoderRNNv121(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, num_heads=8):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Embedding Layer: transform captions into embeded_size 
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Get the focus from features where it should
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        
        # LSTM Layer: Do the magic of finding the next word
        self.lstm_hc = (nn.Linear(self.embed_size, self.hidden_size), nn.Linear(self.embed_size, self.hidden_size))
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        
        # convert output from LSTM into predictions for each word in vocab
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):

        captions_ = captions[:, :-1] # cut last

        embed = self.embedding(captions_)
        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)

        # Initialize the hidden state
        batch_size = features.shape[0] 
        lstm_outputs, _ = self.lstm(embed, self.init_hd_hidden(self.lstm_hc[0], self.lstm_hc[1], features))
        att_out, _ = self.attention(lstm_outputs, lstm_outputs, lstm_outputs)

        out = self.linear(att_out)
        
        return out
    
    
    def sample(self, features, states=None, end_word = 1, max_len=20):
        output_ids = []
        inputs = features.unsqueeze(1)

        for i in range(max_len):
            # pass data through recurrent network
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))

            # find maximal predictions
            predicted = outputs.max(1)[1]

            # append results from given step to global results
            output_ids.append(predicted.cpu().numpy()[0].item())

            # prepare chosen words for next decoding step
            inputs = self.embedding(predicted)
            inputs = inputs.unsqueeze(1)
            if predicted == end_word : break #arrived to the end of the word

        return output_ids
    
    def init_hidden(self, batch_size, size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (torch.zeros((1, batch_size, size), device=device), \
                torch.zeros((1, batch_size, size), device=device))
    
    def init_hd_hidden(self, h, c, features):

        """Initializes hidden state and cell memory using average feature vector.
        Arguments:
        ----------
        - features - features returned from Encoder
        Retruns:
        ----------
        - h0 - initial hidden state (short-term memory)
        - c0 - initial cell state (long-term memory)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mean_annotations = torch.mean(features, dim = 1).to(device)
#        print('gothere 1.1', mean_annotations.is_cuda)
#        print(features.shape)
#        print(mean_annotations.shape)
        if torch.cuda.is_available():
            h = h.cuda()
            c = c.cuda()

#        print(h)
        h0 = h(features).unsqueeze(0)
#        print('gothere 1.2')        
        c0 = c(features).unsqueeze(0)
        return h0, c0
