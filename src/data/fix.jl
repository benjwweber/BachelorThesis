# function _collect_worker_defs(roots)
#     structs = Expr[]
#     globals = Expr[]
#     funcs = Expr[]
#     mods = Set{Module}()
#     unresolved = Symbol[]
#     done = Set{Symbol}()                    # def-names already emitted
#     locs = Set{Tuple{String,Int}}()         # method source locations already shipped
#     seen = Base.IdSet{Any}()                # value-graph cycle guard
#     visit = Ref{Any}()

#     # Ship a method's source once per file location (a keyword method surfaces as several
#     # `Method` objects — body + sorter — all pointing at the same definition line).
#     pushmethod(m::Method, nm::Symbol) = begin
#         loc = (string(m.file), Int(m.line))
#         loc in locs && return
#         push!(locs, loc)
#         ex = _method_source(m)
#         ex === nothing ? push!(unresolved, nm) : push!(funcs, ex)
#         scanbody(m)
#     end

#     # Descend into a value's contents (struct fields / tuple / small array elements) to find
#     # nested `Main` functions and struct instances. Bits arrays (numbers, etc.) are skipped.
#     recurse(@nospecialize(v)) = begin
#         if v isa Tuple || v isa NamedTuple
#             for x in v
#                 visit[](x)
#             end
#         elseif v isa AbstractDict
#                 visit[](first(keys(v)))
#                 visit[](first(values(v)))
#         elseif v isa AbstractArray
#             (isbitstype(eltype(v)) || eltype(v) <: Union{AbstractString,Symbol}) || for x in v
#                 visit[](x)
#             end
#         elseif isstructtype(typeof(v))
#             for i in 1:nfields(v)
#                 isdefined(v, i) || continue
#                 f = try
#                     getfield(v, i)
#                 catch
#                     continue
#                 end
#                 visit[](f)
#             end
#         end
#     end

#     # Scan a method's lowered code for `GlobalRef`s: `Main` functions/structs → ship them,
#     # `Main` plain globals → ship by value, package refs → import that package on the worker.
#     scanbody(m::Method) = begin
#         ci = try
#             Base.uncompressed_ir(m)
#         catch
#             return
#         end
#         scan = Ref{Any}()
#         scan[] = (@nospecialize(x)) -> begin
#             if x isa GlobalRef
#                 if x.mod === Main && isdefined(Main, x.name)
#                     v = getglobal(Main, x.name)
#                     if v isa Function || v isa Type
#                         visit[](v)
#                     elseif !(v isa Module)
#                         _ismain(typeof(v)) && visit[](v)
#                         if !(x.name in done)
#                             push!(done, x.name)
#                             push!(globals, :(global $(x.name) = $(QuoteNode(v))))
#                         end
#                     end
#                 elseif x.mod !== Base && x.mod !== Core && x.mod !== Main
#                     push!(mods, Base.moduleroot(x.mod))
#                 end
#             elseif x isa Expr
#                 foreach(scan[], x.args)
#             elseif x isa Core.ReturnNode
#                 isdefined(x, :val) && scan[](x.val)
#             elseif x isa Core.GotoIfNot
#                 scan[](x.cond)
#             end
#         end
#         foreach(scan[], ci.code)
#     end

#     # A `Main`-bound named function's Symbol, or `nothing` for anonymous/closure/package fns.
#     mainfuncname(@nospecialize(f)) = begin
#         T = typeof(f)
#         (T isa DataType && T.name.module === Main && supertype(T) === Function) || return nothing
#         nm = nameof(f)
#         (isdefined(Main, nm) && getglobal(Main, nm) === f) ? nm : nothing
#     end

#     shipfunc(@nospecialize(f), nm::Symbol) = begin
#         nm in done && return
#         push!(done, nm)
#         for m in methods(f)
#             pushmethod(m, nm)
#         end
#     end

#     shipstruct(@nospecialize(T::DataType)) = begin
#         nm = nameof(T)
#         nm in done && return
#         push!(done, nm)
#         ex = _struct_source(T)
#         ex === nothing ? push!(unresolved, nm) : push!(structs, _guard_struct(nm, ex))
#         for m in methodswith(T; supertypes=false)
#             (m.name === nm || m.module !== Main) && continue   # skip ctor / package methods
#             pushmethod(m, m.name)
#         end
#     end

#     visit[] = (@nospecialize(v)) -> begin
#         (v isa Symbol || v isa Module) && return
#         v in seen && return
#         push!(seen, v)
#         if v isa Function
#             nm = mainfuncname(v)
#             if nm === nothing                       # anonymous/closure: ships by value…
#                 if typeof(v).name.module === Main    # …but scan its body + captured fields
#                     foreach(scanbody, methods(v))
#                     recurse(v)
#                 end
#             else
#                 shipfunc(v, nm)
#             end
#         elseif v isa Type
#             v isa DataType && _ismain(v) && shipstruct(v)
#         else
#             _ismain(typeof(v)) && shipstruct(typeof(v))
#             recurse(v)
#         end
#     end

#     foreach(visit[], roots)
#     return (; structs, globals, funcs, mods, unresolved)
# end

@eval MLUtils function _ship_worker_defs(data, collate, pids)
    Distributed.remotecall_eval(
        Main,
        pids,
        Expr(
            :toplevel,
            :(include("./src/data/Data.jl")),
            :(using .Data)
        )
    )
    return nothing
end
