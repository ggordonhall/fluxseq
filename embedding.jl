using Flux

mutable struct Embedding
    table::TrackedArray
end

"""
Construct an embedding table of dim:
    `vocab size` * `hidden size`
"""
Embedding(ntokens::Int, hiddensize::Int) = Embedding(param(randn(ntokens, hiddensize)))

"""
Given a 2-dim matrix of indices, get the corresponding embedding vectors:
    `batch size` * `sequence length` * `embedding size`
"""
(m::Embedding)(batch) = vcat(map(vect -> m.table[vect, :], batch))


"""
Get the max value and its index for each example in batch.
   batch: `vocab size` * `batch size` :: TrackedArray

   returns: `batch size` :: Array{Int}
"""
function batchmax(batch::TrackedArray) 
    last(findmax(batch.data, dims=1))
end

"""
Convert array of CartesianIndex to array of integer indices:
    cartarray :: Array{CartesianIndex{N}, N}

    returns: `batch size` :: Array{Int}
"""
convertcartesian(cartarr) = last.(map(x -> x.I, cartarr))