module Data
    using MLUtils
    using Serialization
    using SQLite
    using JpegTurbo

    using Reexport
    using SQLite, DBInterface#, Memoization
    using JpegTurbo, Colors
    using Transducers
    using FilePaths
    using FilePathsBase: /
    using Glob
    using BFloat16s
    @reexport using DataAugmentation
    @reexport using StructArrays
    # export LazyTile, Slide, LazyTileSet, DINOSet
    using Reexport
    using Random
    @reexport using DataAugmentation
    @reexport using MLUtils
    using Distributions
    using ImageFiltering
    using Colors

    export Tile,
        TileLoader,
        DINOSet

    export
        GaussianBlur,
        Grayscale,
        AdjustHue,
        AdjustSaturation,
        Solarize,
        RandomSequence,
        MaskGenerator,
        NoMaskGenerator,
        RandomSizedCrop

    include("./augmentations/blur.jl")
    include("./augmentations/grayscale.jl")
    include("./augmentations/hue.jl")
    include("./augmentations/sequence.jl")
    include("./augmentations/saturation.jl")
    include("./augmentations/solarize.jl")
    include("./augmentations/mask.jl")
    include("./augmentations/crop.jl")

    include("./dino.jl")
    include("./tiles.jl")
end
