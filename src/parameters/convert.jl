using PythonCall
using Lux
include("./src/models/Models.jl")
using .Models
using Random
using JLD2

function pytorch_parameters_sparse(
    path,
    model,
    parameters,
    states
)
    pydict = torch.load(path, map_location=torch.device("cpu"))
    function f(layer, parameters, states, keypath)
        #@show keypath
        return layer, (;), (;)
    end
    function f(layer::Lux.Conv, parameters, states, keypath)
        #@show keypath
        pykey = keypathtopykey(keypath)
        weight = reversedims(pygetindex(pydict, pykey, "weight"))
        bias = pygetindex(pydict, pykey, "bias")
        return layer, (; weight, bias), (;)
    end
    function f(layer::Models.Layers.RotaryPositionalEmbedding, parameters, states, keypath)
        @show keypath
        pykey = keypathtopykey(keypath)
        temp = pygetindex(pydict, pykey, "periods")
        periods = reshape(temp, 1, size(temp)...)# insertdims(pygetindex(pydict, pykey, "periods"), dims = 1)
        return layer, (;), (; periods)
    end
    # function f(layer::Models.Layers.MaskToken, parameters, states, keypath)
    # 	pykey = keypathtopykey(keypath)
    # 	token =
    # 	return layer, (;token), (;)
    # end
    function f(layer::Models.Layers.PrefixToken, parameters, states, keypath)
        #@show keypath
        token_class = reversedims(pygetindex(pydict, "cls_token"))
        token_register = reversedims(pygetindex(pydict, "storage_tokens"))
        token = hcat(token_class, token_register)
        return layer, (; token), (;)
    end
    function f(layer::Lux.LayerNorm, parameters, states, keypath)
        #@show keypath
        pykey = keypathtopykey(keypath)
        temp_scale = pygetindex(pydict, pykey, "weight")
        scale = reshape(temp_scale, size(temp_scale, 1), 1, 1)#insertdims(pygetindex(pydict, pykey, "weight"), dims = (2, 3))
        temp_bias = pygetindex(pydict, pykey, "bias")
        bias = reshape(temp_bias, size(temp_bias, 1), 1, 1)#insertdims(pygetindex(pydict, pykey, "bias"), dims = (2, 3))
        return layer, (; scale, bias), (;)
    end
    function f(layer::Lux.Dense, parameters, states, keypath)
        ##@show keypath
        pykey = keypathtopykey(keypath)
        weight = pygetindex(pydict, pykey, "weight")
        bias = pygetindex(pydict, pykey, "bias")
        if last(keypath) in (:dense_q, :dense_k, :dense_v)
            index = findfirst(==(last(keypath)), (:dense_q, :dense_k, :dense_v))
            dim_out = Int(size(weight, 1) / 3)
            index_begin = ((index - 1) * dim_out) + 1
            index_end = index * dim_out
            weight = weight[index_begin:index_end, :]
            bias = bias[index_begin:index_end]
        end
        return layer, (; weight, bias), (;)
    end
    function f(layer::Models.Layers.LayerScale, parameters, states, keypath)
        @show keypath
        pykey = keypathtopykey(keypath)
        @show pykey
        scale = pygetindex(pydict, pykey, "gamma")
        return layer, (; scale), (;)
    end
    function f(layer::Lux.Scale, parameters, states, keypath)
        @show keypath
        pykey = keypathtopykey(keypath)
        @show pykey
        weigth = pygetindex(pydict, pykey, "gamma")
        return layer, (; weigth), (;)
    end
    model, parameters, states = Lux.Experimental.layer_map(f, model, parameters, states)
    return parameters, states
end

function setupload(rng, model, path)
    parameters, states = LuxCore.setup(rng, model)
    parameters = merge_recursive(parameters, load(path, "parameters"))
    states = merge_recursive(states, load(path, "states"))
    return parameters, states
end

const PARAMETER_MAPPING = [
    :embedding_patch => "patch_embed",
    :embedding_rotarypositional => "rope_embed",
    :conv => "proj",
    :token_prefix => missing,
    :layers => missing,
]

const PARAMETER_MAPPING_BLOCKS = r"block_(?<index>\d+)" => s"\g<index -1>"

const PARAMETER_MAPPING_ATTENTION = [
    :skipconnection_attention => missing,
    :norm => "norm1",
    :attention => "attn",
    :scale => "ls1",
    :dense_q => "qkv",
    :dense_k => "qkv",
    :dense_v => "qkv",
    :dense_out => "proj",
    :block => missing,
]

const PARAMETER_MAPPING_FEEDFORWARD = [
    :skipconnection_feedforward => missing,
    :norm => "norm2",
    :feedforward => "mlp",
    :scale => "ls2",
    :dense_in => "fc1",
    :dense_out => "fc2",
    :block => missing,
]

function pygetindex(pydict, pykeys, pykey)
    return pygetindex(pydict, pykeys * "." * pykey)
end

function pygetindex(pydict, pykey)
    return pyconvert(Array{Float32}, getindex(pydict, pykey).to(dtype=torch.float32))
end

function keypathtopykey(keypath)
    keys = collect(keypath)
    if :skipconnection_attention in keys
        replace!(keys, PARAMETER_MAPPING_ATTENTION...)
    elseif :skipconnection_feedforward in keys
        replace!(keys, PARAMETER_MAPPING_FEEDFORWARD...)
    end
    replace!(keys, PARAMETER_MAPPING...)
    pykey = join(skipmissing(keys), ".")
    m = match(r"block_(?<index>\d+)", pykey)
    if m !== nothing
        index = parse(Int, m.captures[1]) - 1
        pykey = replace(pykey, r"block_\d+" => string(index))
    end
    return pykey
end

function reversedims(x)
    return permutedims(x, reverse(ntuple(identity, ndims(x))))
end

model = vit_large()
rng = Random.Xoshiro(42);
parameters, states = LuxCore.setup(rng, model);

parameters, states = pytorch_parameters_sparse(
    "./data/parameters/pytorch/dinov3_vitl16_pretrain_lvd1689m.pth",
    model,
    parameters,
    states
)

jldsave(
    "./data/parameters/lux/dinov3_vitl16_pretrain_lvd168m.jld2";
    parameters,
    states
)