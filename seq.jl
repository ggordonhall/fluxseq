#!/usr/local/bin/julia
using Flux
using Flux: argmax, throttle, crossentropy
using StatsBase: wsample

cd(@__DIR__)
include("preprocess.jl")
include("embedding.jl")

seqlen = 30
batchsize = 1
hiddensize = 2
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
insize = size(invocab, 1) + 4
outsize = size(outvocab, 1) + 4

# Batch sentences
Xs = batchpipe(insents, indict, seqlen, batchsize)
Ys = batchpipe(outsents, outdict, seqlen, batchsize)

encoder = Chain(
    Embedding(insize, hiddensize),
    Dense(hiddensize, hiddensize),
    LSTM(hiddensize, hiddensize),
    Dense(hiddensize, outsize),
    softmax,
    func,
    println
)

func(x) = map(y -> size(y), x)


m = gpu(encoder)

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


