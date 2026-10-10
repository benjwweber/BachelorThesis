struct Grayscale <: DataAugmentation.Transform end

function DataAugmentation.apply(::Grayscale, item::Image; randstate=nothing)
    return DataAugmentation.setdata(item, grayscale(DataAugmentation.itemdata(item)))
end

function DataAugmentation.apply!(buffer, ::Grayscale, item::Image; randstate=nothing)
    grayscale!(DataAugmentation.itemdata(buffer), DataAugmentation.itemdata(item))
    return buffer
end

function grayscale(image)
    return grayscale!(copy(image))
end

grayscale!(image) = grayscale!(image, image)

function grayscale!(dst::AbstractArray{DST}, src::AbstractArray{SRC}) where {DST,SRC}
    map!(dst, src) do rgb
        convert(DST, Colors.Gray(rgb))
    end
end
