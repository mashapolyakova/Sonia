import mido
import sys
sys.path.append('../')
#from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import utils
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from mido import Message, MidiFile, MidiTrack

from midi2audio import FluidSynth

import os

from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

def indexToNote(index):
    decoder = {}
    for i in range(128):
        for j in range(5):
            decoder[i + j*128] = [i, j]
    return decoder[index]

from mido import Message, MidiFile, MidiTrack
def toMidi(Indexes, ticks):
    #print(Indexes)
    moments = [[] for i in range(len(Indexes) + 4)]
    current = 0
    for note in Indexes:
        decoded = indexToNote(note)
        moments[current].append(decoded[0])
        #print(current, decoded[1], len(Indexes))
        moments[current + decoded[1]].append(-decoded[0])
        current += 1
    newMid = MidiFile()
    track = MidiTrack()
    newMid.tracks.append(track)
    tick_time = round(ticks* 0.1) * 2
    for moment in moments:
        for ivent in moment:
            if (ivent < 0):
                track.append(Message('note_off', note=-ivent, velocity=0, time=tick_time))
            else:
                track.append(Message('note_on', note=ivent, velocity=64, time=tick_time))
    return newMid


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


def evaluate(input_tensor, decoder, prob, max_length=100):
    with torch.no_grad():
        
        input_tensor = torch.FloatTensor(input_tensor)
        decoder_input = torch.tensor([[SOS_token]])  # SOS

        decoder_hidden = input_tensor

        decoded_words = []

        
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(prob)
            item = random.randint(0, prob - 1)
            
            topv = topv[0][item]
            topi = topi[0][item]
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(topi.item())

            decoder_input = topi.squeeze().detach()

        return decoded_words

def generate_sample(number, decoder):
    begin = []
    note = number % 100
    for i in range(256):
        begin.append(note)
#     for i in range(256):
#         begin.append(random.randint(200, 240))
    begin = np.reshape(begin, (1,1, 256))
    #print("begin_predict")
    generated = evaluate(begin, decoder, 20)
    mid = toMidi(generated, 480)
    return mid
    #mid.save('generated1.mid')
#generate_sample()


def createDec(path):
    decoder =  torch.load(path)
    return decoder
def run(number, path):
    hidden_size = 256
    output_size = 1000
    decoder = DecoderRNN(hidden_size, output_size)
    middle_tick = 480
    SOS_token = 0
    EOS_token = 1
    decoder = createDec(path)
    input_mid = generate_sample(number, decoder)
    return input_mid


number = int(input())
random.seed(number)
SOS_token = 0
EOS_token = 1
file_path = os.getcwd()
path = file_path + '/decoder_tchaik'
mid = run(number, path)
mid.save("input.mid")
path = file_path + '/FluidR3_GM.sf3'
fs = FluidSynth(path)
path = input()
fs.midi_to_audio('input.mid', path + '/output.wav')

