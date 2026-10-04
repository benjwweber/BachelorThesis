#= MacroTools =#
using MacroTools
using ArgCheck

function argstotuple(args)
    if length(args) == 1
        return only(args)
    end
    names = []
    types = []
    for arg in args
        (name, type, _, _) = MacroTools.splitarg(arg)
        name = isnothing(name) ? :_ : name
        push!(names, name)
        push!(types, type)
    end
    return MacroTools.combinearg(Expr(:tuple, names...), Expr(:curly, :Tuple, types...), false, nothing)
end

function sublayers(type)
    stype = supertype(getfield(@__MODULE__, type))
    if stype <: Lux.AbstractLuxContainerLayer
        return Tuple(stype.parameters[1])
    elseif stype <: Lux.AbstractLuxContainerLayer
        return Tuple(stype.parameters)
    end
    return ()
end

function states_prefix(layers, states)
    exprs = []
    for layer in layers
        push!(exprs, :($(Symbol(:states_, layer)) = states.$layer))
    end
    if MacroTools.isexpr(states, :tuple)
        for expr in states.args
            (name, type, _, _) = MacroTools.splitarg(expr)
            if type == :AbstractRNG
                push!(exprs, :($name = LuxCore.replicate(states.$name)))
            else
                push!(exprs, :($name = states.$name))
            end
        end
    else
        (name, type, _, _) = MacroTools.splitarg(states)
        if type == :AbstractRNG
            push!(exprs, :($name = LuxCore.replicate(states.$name)))
        else
            push!(exprs, :($name = states.$name))
        end
    end
    return Expr(:block, exprs...)
end

function states_suffix(layers, states)
    exprs = []
    for layer in layers
        push!(exprs, Expr(:kw, layer, Symbol(:states_, layer)))
    end
    if MacroTools.isexpr(states, :tuple)
        for expr in states.args
            (name, type, _, _) = MacroTools.splitarg(expr)
            push!(exprs, name)
        end
    else
        (name, type, _, _) = MacroTools.splitarg(states)
        push!(exprs, name)
    end
    return Expr(:tuple, Expr(:parameters, exprs...))
end

function returnstates(expr_in, layers, states)
    expr_states = states_suffix(layers, states)
    expr_return = MacroTools.postwalk(expr_in) do expr
        if isexpr(expr, :return)
            return Expr(:return, Expr(:tuple, expr.args[1], expr_states))
        end

        return expr
    end
    expr_end = last(expr_return.args)
    if !isexpr(expr_end, :return)
        expr_return.args[end] = Expr(
            :return,
            Expr(:tuple,
                Expr(:block, expr_end),
                states_suffix(layers, states)
            )
        )
    end
    return expr_return
end

function states_args(states)
    exprs = []
    if MacroTools.isexpr(states, :tuple)
        for expr in states.args
            (name, type, _, _) = MacroTools.splitarg(expr)
            if isexpr(type, :curly)
                push!(exprs, expr)
            end
        end
    else
        (name, type, _, _) = MacroTools.splitarg(states)
        if isexpr(type, :curly)
            push!(exprs, states)
        end
    end
    return exprs
end

function states_args_call(states)
    exprs = []
    if MacroTools.isexpr(states, :tuple)
        for expr in states.args
            (name, type, _, _) = MacroTools.splitarg(expr)
            if isexpr(type, :curly)
                push!(exprs, :(states.$name))
            end
        end
    else
        (name, type, _, _) = MacroTools.splitarg(states)
        if isexpr(type, :curly)
            push!(exprs, :(states.$name))
        end
    end
    return exprs
end

function appendstates!(expr, layers, states)
    expr_last = last(expr.args)
    expr.args[end] = if isexpr(expr_last)
        if isexpr(expr_last, :return)
            #@show expr_last
            Expr(
                :return,
                Expr(:tuple,
                    expr_last.args[1],
                    states_suffix(layers, states)
                )
            )
        else
            Expr(
                :return,
                Expr(:tuple,
                    Expr(:block, expr_last),
                    states_suffix(layers, states)
                )
            )
        end
    else
        Expr(
            :return,
            Expr(:tuple,
                Expr(:block, expr_last),
                states_suffix(layers, states)
            )
        )
    end
end

replacelayercalls(layers) = Base.Fix1(replacelayercalls, layers)
function replacelayercalls(layer, layers, expr)
    @capture(expr, ($(layer).f_(args__))) || return expr
    if f in layers
        return Symbol(:outputs_, f)
    end
    return expr
end

function replace_pipe_expr(layer, sublayers, expr_in)
    expr_out = MacroTools.postwalk(expr_in) do expr
        @capture(expr, (arg_ |> $(layer).f_)) || return expr
        if f in sublayers
            return :($(layer).$(f)($arg))
        end
        return expr
    end
    return expr_out
end

function apply_(states, expr)
    @argcheck MacroTools.isdef(expr)
    dict_model = MacroTools.splitdef(expr)
    dict_apply = deepcopy(dict_model)

    layer, type, _, _ = MacroTools.splitarg(dict_model[:name])
    layer = isnothing(layer) ? gensym(:layer) : layer
    layers = sublayers(MacroTools.namify(type))
    dict_apply[:name] = Symbol(
        :apply_,
        lowercase(String(MacroTools.namify(type)))
    )
    dict_apply[:body] = returnstates(MacroTools.flatten(recursive_1(replace_pipe_expr(layer, layers, dict_model[:body]); layers, layer)), layers, states)
    pushfirst!(dict_apply[:body].args, states_prefix(layers, states))
    #appendstates!(dict_apply[:body], layers, states)
    dict_apply[:args] = [MacroTools.combinearg(layer, type, false, nothing), argstotuple(dict_model[:args]), :parameters, :states]
    append!(dict_apply[:args], states_args(states))
    expr_apply = MacroTools.combinedef(dict_apply)

    dict_model[:name] = MacroTools.combinearg(:layer, type, false, nothing)
    dict_model[:args] = [:inputs, :parameters, :states]
    dict_model[:kwargs] = []
    dict_model[:whereparams] = ()
    dict_model[:body] = quote
        return $(dict_apply[:name])(layer, inputs, parameters, states, $(states_args_call(states)...))
    end
    expr_model = MacroTools.combinedef(dict_model)

    return MacroTools.prettify(quote
        Core.@__doc__ $expr_model
        $expr_apply
    end)
end

function recursive_1(expr; layers, layer)
    if isexpr(expr)
        if isexpr(expr, :block)
            return Expr(expr.head, recursive_2.(expr.args; layers, layer)...)
        end
        return recursive_1(Expr(:block, expr); layers, layer)
    end
    return expr
end

function recursive_2(expr; layers, layer)
    if isexpr(expr)
        exprs = []
        if isexpr(expr, :call)
            MacroTools.@capture expr f_(args__)
            args = recursive_3.(args; layers, layer, exprs)
            if MacroTools.@capture(f, ($(layer).sublayer_))
                if sublayer in layers
                    arg = length(args) == 1 ? only(args) : Expr(:tuple, args...)
                    return quote
                        $(exprs...)
                        ($(Symbol(:outputs_, sublayer)), $(Symbol(:states_, sublayer))) = @inline LuxCore.apply($layer.$sublayer, $arg, parameters.$sublayer, $(Symbol(:states_, sublayer)))
                        $(Symbol(:outputs_, sublayer))
                    end
                end
            end
            return quote
                $(exprs...)
                $f($(args...))
            end
        end
        # return Expr(expr.head, recursive_2.(expr.args; layers, layer)...)
        args = recursive_3.(expr.args; layers, layer, exprs)
        return quote
            $(exprs...)
            $(Expr(expr.head, args...))
        end
    end
    return expr
end

function recursive_3(expr; layers, layer, exprs)
    if isexpr(expr)
        if isexpr(expr, :call)
            MacroTools.@capture expr f_(args__)
            args = recursive_3.(args; layers, layer, exprs)
            if MacroTools.@capture(f, ($(layer).sublayer_))
                if sublayer in layers
                    arg = length(args) == 1 ? only(args) : Expr(:tuple, args...)
                    push!(
                        exprs,
                        :(($(Symbol(:outputs_, sublayer)), $(Symbol(:states_, sublayer))) = @inline LuxCore.apply($layer.$sublayer, $arg, parameters.$sublayer, $(Symbol(:states_, sublayer))))
                    )
                    return Symbol(:outputs_, sublayer)
                end
            end
            return :($f($(args...)))
        end
        return Expr(expr.head, recursive_3.(expr.args; layers, layer, exprs)...)
    end
    return expr
end
macro apply(expr)
    return esc(apply_(:(()), expr))
end
macro apply(states, expr)
    return esc(apply_(states, expr))
end
