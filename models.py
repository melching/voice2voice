from turtle import forward
import torch
import torch.nn as nn
import torch.functional as F

'''
Define model to extract information about the speaker whose voice-features should be used to transform a voice.

Concept: CNN over Spectogram Data, followed by sequence model (prob lstm or transformer) to generate feature vector

'''
class VoiceEmbedder001(nn.Module):

    def __init__(self, freq_dim, out_dim, enc_n_head, enc_num_layers) -> None:
        super().__init__()
        # parameter
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        self.enc_n_head = enc_n_head
        self.enc_num_layers = enc_num_layers

        # layer
        self.cnn1 = nn.Conv2d(1, 64, (3,1), 1)
        self.cnn2 = nn.Conv2d(64, 32, (3,1), 1)
        self.cnn3 = nn.Conv2d(32, 1, (1,1), 1)
        
        self.transf_enc_layer = nn.TransformerEncoderLayer(d_model=self.freq_dim, n_head=self.enc_n_head, batch_first=True)
        self.transf_enc = nn.TransformerEncoder(encoder_layer=self.transf_enc_layer, num_layers=self.enc_num_layers)
        
        self.adapt_max_pool = nn.AdaptiveMaxPool2d((64, None))

        self.flat = nn.Flatten(1, -1)

        self.lin_out = nn.Linear(64*freq_dim, out_dim)


    # takes data as (batch, sequence, feature)
    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.transf_enc(x)
        x = self.adapt_max_pool(x)
        x = self.flat(x)
        x = self.lin_out(x)
        return x

'''
Define model to create (seemingly spoken) sound matching the voice of another person.
'''
class VoiceCreator001(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, voice_embedding):
        return x, voice_embedding