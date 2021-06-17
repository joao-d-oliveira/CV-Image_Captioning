import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


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


###
# Added for comparasitation of results.
# Taken from the tutorial at:
# https://medium.com/analytics-vidhya/image-captioning-with-attention-part-1-e8a5f783f6d3
###
class EncoderCNNv1(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNNv1, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        # first, we need to resize the tensor to be 
        # (batch, size*size, feature_maps)
        batch, feature_maps, size_1, size_2 = features.size()
        features = features.permute(0, 2, 3, 1)
        features = features.view(batch, size_1*size_2, feature_maps)

        return features


# Simplest Decoder
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
        embed = self.embedding(captions)

        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)

        # Initialize the hidden state
        batch_size = features.shape[0]
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
            # arrived to the end of the sentence
            if predicted == end_word : break

        return output_ids

    # taken from the previous lesson
    def init_hidden(self, batch_size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (torch.zeros((1, batch_size, self.hidden_size), device=device), \
                torch.zeros((1, batch_size, self.hidden_size), device=device))


# Slightly more complex than v101 with a Linear layer as states from LSTM
# proved that performs better
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
        embed = self.embedding(captions)

        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)

        # Initialize the hidden state
        self.lstm_hc = self.init_hd_hidden(self.lstm_hc[0], self.lstm_hc[1], features)
        lstm_outputs, self.lstm_hc = self.lstm(embed, self.lstm_hc)
        out = self.linear(lstm_outputs)

        return out

    def sample(self, inputs, hidden=None, end_word = 1, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        output = []
        # Either get hidden states already from pretrained or initialize
        if hidden is None:
            batch_size = inputs.shape[0]
            hidden = self.init_hidden(batch_size)

        while True:
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.linear(lstm_out)
            outputs = outputs.squeeze(1)

            # get the word with the best ranking
            _, found_word = torch.max(outputs, dim=1)

            # save new word
            output.append(found_word.cpu().numpy()[0].item()) # storing the word predicted
            # In case new word is the end of the sentence... end the sampling
            if found_word == end_word or len(output) > max_len: break

            # embed the last predicted word to predict next
            inputs = self.embedding(found_word)
            inputs = inputs.unsqueeze(1)

        return output

    def init_hd_hidden(self, h, c, features):
        if torch.cuda.is_available(): h = h.cuda(); c = c.cuda()
        h = h(features).unsqueeze(0)
        c = c(features).unsqueeze(0)

        return h, c

    def init_hidden(self, batch_size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (torch.zeros((1, batch_size, self.hidden_size), device=device), \
                torch.zeros((1, batch_size, self.hidden_size), device=device))


# Added a MultiHeadAttention Layer after the LSTM, trying to focus the attention after LSTM
# Then performing the operations with EMB -> LSTM -> Attention -> Linear
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
        self.lstm_hc = (nn.Linear(self.embed_size, self.hidden_size), nn.Linear(self.embed_size, self.hidden_size))
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)

        # convert output from LSTM into predictions for each word in vocab
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embed = self.embedding(captions)
        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)

        # Initialize the hidden state
        self.lstm_hc = self.init_hd_hidden(self.lstm_hc[0], self.lstm_hc[1], features)
        lstm_outputs, self.lstm_hc = self.lstm(embed, self.lstm_hc)
        att_out, _ = self.attention(lstm_outputs, lstm_outputs, lstm_outputs)

        out = self.linear(att_out)

        return out

    def sample(self, inputs, hidden=None, end_word = 1, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        output = []
        if hidden is None:
            batch_size = inputs.shape[0]
            hidden = self.init_hidden(batch_size)

        while True:
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs, _ = self.attention(lstm_out, lstm_out, lstm_out)
            outputs = self.linear(outputs)
            outputs = outputs.squeeze(1)

            # get the word with the best ranking
            _, max_indice = torch.max(outputs, dim=1)

            # save new word
            output.append(max_indice.cpu().numpy()[0].item())  # storing the word predicted
            # In case new word is the end of the sentence... end the sampling
            if max_indice == end_word or len(output) > max_len: break

            # embed the last predicted word to predict next
            inputs = self.embedding(max_indice)
            inputs = inputs.unsqueeze(1)

        return output

    def init_hd_hidden(self, h, c, features):
        if torch.cuda.is_available(): h = h.cuda(); c = c.cuda()
        h = h(features).unsqueeze(0)
        c = c(features).unsqueeze(0)

        return h, c

    def init_hidden(self, batch_size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (torch.zeros((1, batch_size, self.hidden_size), device=device), \
                torch.zeros((1, batch_size, self.hidden_size), device=device))


# Added a MultiHeadAttention Layer before, trying to focus the attention first at features
# It actually performs better than the v120
# Then performing the operations with EMB -> Attention -> LSTM ->  Linear
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
        self.attention = nn.MultiheadAttention(embed_size, num_heads)

        # LSTM Layer: Do the magic of finding the next word
        self.lstm_hc = (nn.Linear(self.embed_size, self.hidden_size), nn.Linear(self.embed_size, self.hidden_size))
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)

        # convert output from LSTM into predictions for each word in vocab
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embed = self.embedding(captions)
        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)

        # Initialize the hidden state
        self.lstm_hc = self.init_hd_hidden(self.lstm_hc[0], self.lstm_hc[1], features)
        lstm_outputs, self.lstm_hc = self.lstm(embed, self.lstm_hc)
        att_out, _ = self.attention(lstm_outputs, lstm_outputs, lstm_outputs)

        out = self.linear(att_out)

        return out

    def sample(self, inputs, hidden=None, end_word = 1, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output = []
        if hidden is None:
            batch_size = inputs.shape[0]
            hidden = self.init_hidden(batch_size)

        while True:
            outputs, _ = self.attention(inputs, inputs, inputs)
            outputs, hidden = self.lstm(outputs, hidden)
            outputs = self.linear(outputs)
            outputs = outputs.squeeze(1)

            _, found_word = torch.max(outputs, dim=1) # predict the most likely next word, found_word shape : (1)
            # save new word
            output.append(found_word.cpu().numpy()[0].item())
            # In case new word is the end of the sentence... end the sampling
            if found_word == end_word or len(output) > max_len: break

            # embed the last predicted word to predict next
            inputs = self.embedding(found_word)
            inputs = inputs.unsqueeze(1)

        return output

    def init_hd_hidden(self, h, c, features):
        if torch.cuda.is_available(): h = h.cuda(); c = c.cuda()
        h = h(features).unsqueeze(0)
        c = c(features).unsqueeze(0)

        return h, c

    def init_hidden(self, batch_size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (torch.zeros((1, batch_size, self.hidden_size), device=device), \
                torch.zeros((1, batch_size, self.hidden_size), device=device))


###
# Added for comparasitation of results.
# Taken from the tutorial at:
# https://medium.com/analytics-vidhya/image-captioning-with-attention-part-1-e8a5f783f6d3
###
class BahdanauAttention(nn.Module):
    """ Class performs Additive Bahdanau Attention.
        Source: https://arxiv.org/pdf/1409.0473.pdf
     
    """
    def __init__(self, num_features, hidden_dim, output_dim = 1):
        super(BahdanauAttention, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # fully-connected layer to learn first weight matrix Wa
        self.W_a = nn.Linear(self.num_features, self.hidden_dim)
        # fully-connected layer to learn the second weight matrix Ua
        self.U_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        # fully-connected layer to produce score (output), learning weight matrix va
        self.v_a = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, features, decoder_hidden):
        """
        Arguments:
        ----------
        - features - features returned from Encoder
        - decoder_hidden - hidden state output from Decoder
                
        Returns:
        ---------
        - context - context vector with a size of (1,2048)
        - atten_weight - probabilities, express the feature relevance
        """
        # add additional dimension to a hidden (required for summation)
        decoder_hidden = decoder_hidden.unsqueeze(1)
        atten_1 = self.W_a(features)
        atten_2 = self.U_a(decoder_hidden)
        # apply tangent to combine result from 2 fc layers
        atten_tan = torch.tanh(atten_1+atten_2)
        atten_score = self.v_a(atten_tan)
        atten_weight = F.softmax(atten_score, dim = 1)
        # first, we will multiply each vector by its softmax score
        # next, we will sum up this vectors, producing the attention context vector
        # the size of context equals to a number of feature maps
        context = torch.sum(atten_weight * features,  dim = 1)
        atten_weight = atten_weight.squeeze(dim=2)

        return context, atten_weight


###
# Added for comparasitation of results.
# Taken from the tutorial at:
# https://medium.com/analytics-vidhya/image-captioning-with-attention-part-1-e8a5f783f6d3
###
class DecoderRNNv200(nn.Module):
    """Attributes:
         - embedding_dim - specified size of embeddings;
         - hidden_dim - the size of RNN layer (number of hidden states)
         - vocab_size - size of vocabulary 
         - p - dropout probability
    """
    def __init__(self, num_features, embedding_dim, hidden_dim, vocab_size, p =0.5):

        super(DecoderRNNv200, self).__init__()

        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        # scale the inputs to softmax
        self.sample_temp = 0.5

        # embedding layer that turns words into a vector of a specified size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # LSTM will have a single layer of size 512 (512 hidden units)
        # it will input concatinated context vector (produced by attention) 
        # and corresponding hidden state of Decoder
        self.lstm = nn.LSTMCell(embedding_dim + num_features, hidden_dim)
        # produce the final output
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # add attention layer 
        self.attention = BahdanauAttention(num_features, hidden_dim)
        # dropout layer
        self.drop = nn.Dropout(p=p)
        # add initialization fully-connected layers
        # initialize hidden state and cell memory using average feature vector 
        # Source: https://arxiv.org/pdf/1502.03044.pdf
        self.init_h = nn.Linear(num_features, hidden_dim)
        self.init_c = nn.Linear(num_features, hidden_dim)

    def forward(self, captions, features, sample_prob = 0.0):
        import numpy as np

        embed = self.embeddings(captions)
        h, c = self.init_hidden(features)
        seq_len = captions.size(1)
        feature_size = features.size(1)
        batch_size = features.size(0)
        # these tensors will store the outputs from lstm cell and attention weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(device)
        atten_weights = torch.zeros(batch_size, seq_len, feature_size).to(device)
        # scheduled sampling for training
        # we do not use it at the first timestep (<start> word)
        # but later we check if the probability is bigger than random
        for t in range(seq_len):
            sample_prob = 0.0 if t == 0 else 0.5
            use_sampling = np.random.random() < sample_prob
            if use_sampling == False:
                word_embed = embed[:,t,:]
            context, atten_weight = self.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + hidden_dim)
            input_concat = torch.cat([word_embed, context], 1)
            h, c = self.lstm(input_concat, (h,c))
            h = self.drop(h)
            output = self.fc(h)
            if use_sampling == True:
                # use sampling temperature to amplify the values before applying softmax
                scaled_output = output / self.sample_temp
                scoring = F.log_softmax(scaled_output, dim=1)
                top_idx = scoring.topk(1)[1]
                word_embed = self.embeddings(top_idx).squeeze(1)
            outputs[:, t, :] = output
            #atten_weights[:, t, :] = atten_weights
        return outputs, atten_weights

    def init_hidden(self, features):

        """Initializes hidden state and cell memory using average feature vector.
        Arguments:
        ----------
        - features - features returned from Encoder
    
        Retruns:
        ----------
        - h0 - initial hidden state (short-term memory)
        - c0 - initial cell state (long-term memory)
        """
        mean_annotations = torch.mean(features, dim = 1)
        h0 = self.init_h(mean_annotations)
        c0 = self.init_c(mean_annotations)
        return h0, c0

    def sample(self, features, max_sentence = 20):

        """Greedy search to sample top candidate from distribution.
        Arguments
        ----------
        - features - features returned from Encoder
        - max_sentence - max number of token per caption (default=20)
        Returns:
        ----------
        - sentence - list of tokens
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sentence = []
        weights = []
        input_word = torch.tensor(0).unsqueeze(0).to(device)
        h, c = self.init_hidden(features)
        while True:
            embedded_word = self.embeddings(input_word)
            context, atten_weight = self.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + context size)
            input_concat = torch.cat([embedded_word, context],  dim = 1)
            h, c = self.lstm(input_concat, (h,c))
            h = self.drop(h)
            output = self.fc(h)
            scoring = F.log_softmax(output, dim=1)
            top_idx = scoring[0].topk(1)[1]
            sentence.append(top_idx.item())
            weights.append(atten_weight)
            input_word = top_idx
            if (len(sentence) >= max_sentence or top_idx == 1):
                break
        return sentence, weights
