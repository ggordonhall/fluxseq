using Flux
using Flux: chunk, batchseq, frequencies
using Base.Iterators: flatten

padidx = 1
sosidx = 2
eosidx = 3
unkidx = 4

"""
Strip punctuation from strings.
"""
nopunc(str) = replace(lowercase(str), r"[^A-Za-z0-9\s]" => s"")
cleanline(str) = split(nopunc(str))

"""
Add start and end token indices to vectorised line.
"""
delimit(line) = [sosidx, line..., eosidx]

"""
Return `vsize` tokens in vocabulary.
"""
getvocab(tokens, vsize) = ["<p>", "<s>", "<\\s>", "<unk>", truncvocab(tokens, vsize)...]

"""
Limit vocab to the top `vsize` occuring tokens.
"""
function truncvocab(tokens, vsize)
    count = sort(collect(frequencies(tokens)), by=x->x[2])
    if length(count) > vsize
        count = count[end-(vsize-1):end]
    end
    println("Vocabulary consists of $(length(count)) tokens")
    return first.(count)
end

"""
Get cleaned strings and vocabulary from a set of strings.
    -> split lines: [["the", "lord"...], ["was", "in"...]]
"""
function preprocess(lines, vsize)
    println("Preprocessing $(length(lines)) lines")
    tokens = cleanline.(lines) 
    return tokens, getvocab(collect(flatten(tokens)), vsize)
end

"""
Get a dictionary mapping words to vector indices.
    -> First three idxs are reserved!
"""
function builddict(vocab)
    word2idx = Dict{eltype(vocab),Int}()
    for (idx, word) in enumerate(vocab)
        word2idx[word] = idx
    end
    return word2idx
end

"""
Turn tokens to indices.
    -> to idx vector: [[1, 78...], [1, 32...]]
"""
function vectorise(lines, dict, maxlen)
    indexer = line -> delimit([get(dict, word, unkidx) for word in line])
    return pad(indexer.(lines), maxlen)
end

"""
Make all lines `maxlen` and pad.
"""
function pad(xs, maxlen)
    padfunc = x -> length(x) > maxlen ?
        x[1:maxlen] : rpad(x, maxlen, padidx)
    return padfunc.(xs)
end

"""
Batching pipeline.
    -> Index lines and split into evenly-sized batches
"""
function batchpipe(sents, dict, seqlen, batchsize)
    stopvect = ones(seqlen)
    # index and pad vectors
    vects = vectorise(sents, dict, seqlen)
    # batch and pad with `stopvect`
    return batchseq(chunk(vects, batchsize), stopvect)
end