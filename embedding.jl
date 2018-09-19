mutable struct Embedding
    table::Array{Float64, 2}
end

"""
Converts batch vectors from Float64 to Int32
"""
toint32(arr) = convert(Array{Array{Int32}}, arr)

"""
Construct an embedding table of dim:
    `vocab size` * `embedding size`
"""
embed(ntokens::Int, emsize::Int) = Embedding(randn(ntokens, emsize))

"""
Given a 2-dim matrix of indices, get the corresponding embedding vectors:
    `batch size` * `sequence length` * `embedding size`
"""
lookup(Embedding, batch) = vcat(map(vect -> Embedding.table[vect, :], toint32(batch)))

"""
Replace updated weights in the embedding table.
"""
function updatetable(Embedding, out::Array{Float64}, vect::Array{Int})
    return Embedding.table[vect, :] = out
end