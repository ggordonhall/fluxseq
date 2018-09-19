#!/usr/local/bin/julia
using Flux
using Flux: argmax, throttle, crossentropy
using StatsBase: wsample

cd(@__DIR__)
include("preprocess.jl")
include("embedding.jl")

seqlen = 30
batchsize = 16
hiddensize = 64
vsize = 5000

# Read translation files
open("data/english_bible.txt") do file
    global insents, invocab
    insents, invocab = preprocess(readlines(file), vsize)
end

open("data/latin_bible.txt") do file
    global outsents, outvocab
    outsents, outvocab = preprocess(readlines(file), vsize)
end

# Build word, index dictionaries
indict = builddict(invocab)
outdict = builddict(outvocab)
# Vocab + (pad, unk, sos, eos)
vsize = vsize + 4

# Batch sentences
inbatch = batchpipe(insents, indict, seqlen, batchsize)
outbatch = batchpipe(outsents, outdict, seqlen, batchsize)

# Embed
inembed = embed(vsize, hiddensize)
outembed = embed(vsize, hiddensize)

# Get feature vectors
Xs = lookup(inembed, inbatch)
Ys = lookup(outembed, outbatch)

encoder = Chain(
    Dense(hiddensize, hiddensize),
    LSTM(hiddensize, hiddensize)
)

decoder = Chain(
    LSTM(hiddensize, hiddensize),
    Dense(hiddensize, vsize),
    softmax
)

m = gpu(Chain(encoder, decoder))

opt = ADAM(params(m), 0.01)
tx, ty = (gpu.(Xs), gpu.(Ys))
evalcb = () -> @show loss(tx, ty)

function loss(xs, ys)
    l = sum(crossentropy.(m.(gpu.(xs)), gpu.(ys)))
    Flux.truncate!(m)
    return l
end  

Flux.train!(loss, zip(Xs, Ys), opt,
            cb = throttle(evalcb, 30))

