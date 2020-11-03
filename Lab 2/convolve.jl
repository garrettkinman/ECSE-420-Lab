using CSV
using Images
using DataFrames
using CUDA

args = CSV.read("convolve_args.csv")

kernel = Matrix([1 2 -1
                2 0.25 -2
                1 -2 -1])

function ∑(window::AbstractArray, kernel::AbstractArray)::RGB
    # find the R, G, and B components of each convolution window
    R = 0.0
    G = 0.0
    B = 0.0
    for i ∈ 1:3
        for j ∈ 1:3
            @inbounds R += window[i,j].r * kernel[i,j]
            @inbounds G += window[i,j].g * kernel[i,j]
            @inbounds B += window[i,j].b * kernel[i,j]
        end
    end
    return thresh(RGB(R, G, B))
end

function ∑!(pixel::RGB, window::CuArray, kernel::CuArray)
    # find the R, G, and B components of each convolution window
    R = 0.0
    G = 0.0
    B = 0.0
    for i ∈ 1:3
        for j ∈ 1:3
            @inbounds R += window[i,j].r * kernel[i,j]
            @inbounds G += window[i,j].g * kernel[i,j]
            @inbounds B += window[i,j].b * kernel[i,j]
        end
    end
    pixel.r = R
    pixel.g = G
    pixel.b = B
    return nothing
end

function thresh(x::RGB)::RGB
    return RGB(thresh(x.r), thresh(x.g), thresh(x.b))
end

function thresh(x::Normed)::Normed
    if x > 1
        return 1.0
    elseif x < 0
        return 0.0
    end
    return x
end

function convolve_sequential(img::AbstractArray, kernel::AbstractArray)
    height, width = size(img)
    convolved = copy(img)
    for i ∈ 2:(height - 1)
        for j ∈ 2:(width - 1)
            @inbounds convolved[i,j] = ∑(img[i - 1:i + 1,j - 1:j + 1], kernel)
        end
    end
    return map(thresh, convolved)[2:height - 1,2:width - 1]
end

function convolve_parallel(img::AbstractArray, kernel::AbstractArray, num_threads::Int)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    height, width = size(img)
    convolved = copy(img)
    for i ∈ 2:(height - 1)
        for j ∈ 2:(width - 1)
            @inbounds convolved[i,j] = ∑(img[i - 1:i + 1,j - 1:j + 1], kernel)
        end
    end
    map(thresh, convolved)
    return convolved[2:height - 1,2:width - 1]
end

input_img = load("Test_1.png")
output_img = convolve_sequential(input_img, kernel)
output_img_cuda = convolve_parallel(cu(input_img), cu(kernel), 1024)

for row ∈ eachrow(args)
    input_filename, output_filename, num_threads = Tuple(row)
    # TODO:
    # 1) read in images
    # 2) convolve sequentially and in parallel
    # 3) clamp values in convolved images
    # 4) record benchmarks
    # 5) write convolved images

    input_img = load(input_filename)
    output_img = convolve_sequential(input_img, kernel)
    save(output_filename, output_img)
end
