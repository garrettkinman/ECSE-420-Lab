using CSV
using Images
using DataFrames

args = CSV.read("convolve_args.csv")

kernel = Matrix([1 2 -1
                2 0.25 -2
                1 -2 -1])

function ∑(window::AbstractMatrix{RGB}, kernel::AbstractMatrix)::RGB
    # find the R, G, and B components of each convolution window
    r_component = dot(red.(window), kernel)
    g_component = dot(green.(window), kernel)
    b_component = dot(green.(window), kernel)
    return RGB(r_component, g_component, b_component)
end

function thresh(x::RGB)::RGB
    return RGB(clamp(x.r), clamp(x.g), clamp(x.b))
end

function thresh(x::Integer)::Integer
    if x > 255
        return 255
    elseif x < 0
        return 0
    end
    return x
end

function convolve_sequential(img::AbstractMatrix{RGB}, kernel::AbstractMatrix)::AbstractMatrix{RGB}
    # TODO
    height, width = size(img)
    convolved = Matrix(RGB(0, 0, 0), (height - 2, width - 2))
    for i ∈ 2:(height - 1)
        for j ∈ 2:(width - 1)
            convolved[i,j] = round(∑(img[i-1:i+1,j-1:j+1], kernel))
        end
    end
    return map(clamp, convolved)
end

function convolve_parallel(img::AbstractMatrix{RGB}, kernel::AbstractMatrix, num_threads::Int)::AbstractMatrix{RGB}
    # TODO
end

input_img = load("Test_1.png")
output_img = convolve_sequential(input_img, kernel)

for row ∈ eachrow(args)
    input_filename, output_filename, num_threads = Tuple(row)
    # TODO:
    # 1) read in images
    # 2) convolve sequentially and in parallel
    # 3) clamp values in convolved images
    # 4) record benchmarks
    # 5) write convolved images

    input_img = load(input_filename)
    println(typeof(input_img))
    convolve_sequential(input_img, kernel)
end
