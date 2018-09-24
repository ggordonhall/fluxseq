#!/usr/local/bin/julia
using Flux
using Flux: flip, argmax, throttle, crossentropy
using StatsBase: wsample

cd(@__DIR__)
include("preprocess.jl")
include("embedding.jl")

seqlen = 5
bsize = 3
hiddensize = 3
vsize = 5

# Read translation files
open("data/short_english_bible.txt") do file
    global insents, invocab
    insents, invocab = preprocess(readlines(file), vsize)
end

open("data/short_latin_bible.txt") do file
    global outsents, outvocab
    outsents, outvocab = preprocess(readlines(file), vsize)
end

# Build word, index dictionaries
indict = builddict(invocab)
outdict = builddict(outvocab)
# Vocab + (pad, unk, sos, eos)
insize = size(invocab, 1)
outsize = size(outvocab, 1)

# Batch sentences
Xs = batchpipe(insents, indict, seqlen, bsize)
Ys = batchpipe(outsents, outdict, seqlen, bsize)

# Embedding Matrices
inembed = Embedding(insize, hiddensize)
outembed = Embedding(outsize, hiddensize)
embed(batch, embeding) = embedding(batch)

# Bidirectional LSTM Encoder
forward  = LSTM(insize, hiddensize÷2)
backward = LSTM(insize, hiddensize÷2)
encode(tokens) = vcat.(forward.(tokens), flip(backward, tokens))

# Alignment Layer
alignnet = Dense(2hiddensize, 1)
align(s, t) = alignnet(vcat(t, s .* trues(1, size(t, 2))))

recur   = LSTM(insize, hiddensize)
toalpha = Dense(hiddensize, outsize)

# Why not use the standard softmax?
function asoftmax(xs)
    xs = [exp.(x) for x in xs]
    s = sum(xs)
    return [x ./ s for x in xs]
end

function decode1(tokens, phone)
    weights = asoftmax([align(recur.state[2], t) for t in tokens])
    context = sum(map((a, b) -> a .* b, weights, tokens))
    y = recur(vcat(float(phone), context))
    return softmax(toalpha(y))
end

# What is phone? Why is it a sequence? Does this represent a batch?
decode(tokens, phones) = [decode1(tokens, phone) for phone in phones]

# The full model
# Embedding could be added to this -> and the `param` setting removed from
# embedding.jl 
state = (forward, backward, alignnet, recur, toalpha)

function model(x, y)
  ŷ = decode(encode(x), y)
  reset!(state)
  return ŷ
end

# Need extra data to feed to the model, i.e. the optimal
# prediction at stage i of the decoding process.
loss(x, yo, y) = sum(crossentropy.(model(x, yo), y))

evalcb = () -> @show loss(data[500]...)
opt = ADAM(params(state))

Flux.train!(loss, data, opt, cb = throttle(evalcb, 10))

# Purpose of the Embedding:
# The vector embedding representation of each input is passed to the encoder,
# i.e. token at index 4 -> [0.3, 0.8]. These vectors are passed through the model.
# The embedding at the previous time-step, Y_o is passed to the decoder and used to produce an output, Y_hat.
# Y_hat should be a probability distribution over tokens in the vocabulary.



