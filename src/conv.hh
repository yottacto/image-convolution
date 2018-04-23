#pragma once
#include <vector>
#include <array>
#include <omp.h>

namespace opt
{

template <
    class T,
    class Vec,
    std::size_t N
>
void convolve(int rows, int cols, Vec const& din, Vec& dout,
    std::array<std::array<T, N>, N> const& filer)
{
    using value_type = typename Vec::value_type;
    int filter_size = N;
    auto out_rows = rows - filter_size + 1;
    auto out_cols = cols - filter_size + 1;
    for (auto x = out_rows; x < rows; x++)
    for (auto y = out_cols; y < cols; y++)
        dout[x * cols + y] = 0;

    #pragma omp parallel for num_threads(8)
    for (auto x = 0; x < out_rows; x++)
    for (auto y = 0; y < out_cols; y++) {
        auto sum = value_type{0};
        for (auto fx = 0; fx < filter_size; fx++)
        for (auto fy = 0; fy < filter_size; fy++) {
            auto fi = filer[fx][fy];
            auto di = din[(x + fx) * cols + y + fy];
            sum += fi * di;
        }
        if (sum < 0) sum = 0;
        if (sum > 255) sum = 255;
        dout[x * cols + y] = sum;
    }
}

} // namespace opt

