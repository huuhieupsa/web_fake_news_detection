import torch
import torch.nn as nn
from transformers import AutoTokenizer
from langdetect import detect

class LSTMNet(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout):
        super(LSTMNet,self).__init__()
        # Embedding layer converts integer sequences to vector sequences
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        # LSTM layer process the vector sequences
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers = n_layers,
                            bidirectional = bidirectional,
                            dropout = dropout,
                            batch_first = True
                           )
        # Dense layer to predict
        self.fc = nn.Linear(hidden_dim * 2,output_dim)
        # Prediction activation function
        self.sigmoid = nn.Sigmoid()
    def forward(self,text,text_lengths):
        embedded = self.embedding(text).to(device)
        # Thanks to packing, LSTM don't see padding tokens
        # and this makes our model better
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths,batch_first=True)
        packed_output,(hidden_state,cell_state) = self.lstm(packed_embedded)
        # Concatenating the final forward and backward hidden states
        hidden = torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1)
        dense_outputs=self.fc(hidden)
        #Final activation function
        outputs=self.sigmoid(dense_outputs)

        return outputs

# Load tokenizer
tokenizer_EN = AutoTokenizer.from_pretrained("./static/tokenizer_EN")
tokenizer_VN = AutoTokenizer.from_pretrained("./static/tokenizer_VN")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = tokenizer_VN.vocab_size - 1
embedding_dim = 100
hidden_dim = 64
output_dim = 1
n_layers = 2
bidirectional = True
dropout = 0.2
model = LSTMNet(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
model.load_state_dict(torch.load('./static/model.pth', map_location=device, weights_only=True))

# # Predict
def model_predict(text):   
    tokenizer = None
    if (detect(text) == 'vi'):
        tokenizer = tokenizer_VN
    else:
        tokenizer = tokenizer_EN

    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(device)
    # Get the length of the input sequence
    length = torch.tensor(ids.shape[1], dtype=torch.long).unsqueeze(0)
        # Evaluate the model on the input text
    with torch.no_grad():
        model.eval()
        predictions = model(ids, length)

    binary_predictions = torch.round(predictions).cpu().numpy()

    if (int(binary_predictions[0][0]) == 0):
        # print("Đây là tin giả")
        return 0
    else:
        # print("Đây là tin thật")
        return 1