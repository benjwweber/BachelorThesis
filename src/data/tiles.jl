RandomApply = Maybe
ColorToChannels = ImageToTensor

struct Tile
    path::String
    id::Int
    level::Int
    x::Int
    y::Int

    function Tile(path::AbstractString, id::Int, level::Int, x::Int, y::Int)
        return new(normpath(String(path)), id, level, x, y)
    end
end

function gettiles(path)
    slide = SQLite.DB("file:" * path * "?mode=ro")
    return map(DBInterface.execute(slide,
        "SELECT id, level, x, y FROM tiles WHERE level > (SELECT MAX(level) FROM tiles) - 1;"
    )) do result
        Tile(path, result.id, result.level, result.x, result.y)
    end
end

struct TileLoader
    slides#::Dict{String,SQLite.DB}
    tiles::Vector{Tile}
end

function TileLoader(tiles::AbstractVector{Tile})
    slides = Dict{String,SQLite.DB}()
    for tile in tiles
        path = tile.path
        if !haskey(slides, path)
            slides[path] = SQLite.DB("file:" * path * "?mode=ro")
        end
    end
    return TileLoader(slides, collect(tiles))
end

function TileLoader(paths::AbstractVector{<:AbstractString}, tiles::AbstractVector{Tile})
    slides = Dict{String,SQLite.DB}()
    for path in paths
        p = normpath(String(path))
        slides[p] = SQLite.DB("file:" * p * "?mode=ro")
    end
    return TileLoader(slides, collect(tiles))
end

Base.length(tileloader::TileLoader) = length(tileloader.tiles)
Base.lastindex(tileloader::TileLoader) = length(tileloader)
numobs(tileloader::TileLoader) = length(tileloader.tiles)

function Base.getindex(tileloader::TileLoader, i::Int)
    tile = tileloader.tiles[i]
    slide = tileloader.slides[tile.path]
    (; jpeg) = first(DBInterface.execute(slide, "SELECT jpeg FROM tiles WHERE id = $(tile.id)"))
    return jpeg_decode(jpeg)
end

function Serialization.serialize(s::AbstractSerializer, tileloader::TileLoader)
    Serialization.serialize_type(s, TileLoader)
    Serialization.serialize(s, tileloader.tiles)
    return nothing
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{TileLoader})
    tiles = Serialization.deserialize(s)
    return TileLoader(tiles)
end
