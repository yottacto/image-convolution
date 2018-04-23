#pragma once
#include <vector>
#include <iterator>
#include <functional>
#include <algorithm>
#include <thread>
#include <array>
#include <omp.h>

namespace thread
{

template <
    class T,
    class Vec,
    std::size_t N
>
void convolve(int rows, int cols, Vec const& din, Vec& dout,
    std::array<std::array<T, N>, N> const& filer, int size = -1)
{
    using value_type = typename Vec::value_type;
    int filter_size = N;
    auto out_rows = rows - filter_size + 1;
    auto out_cols = cols - filter_size + 1;
    for (auto x = out_rows; x < rows; x++)
    for (auto y = out_cols; y < cols; y++)
        dout[x * cols + y] = 0;

    std::vector<std::thread> threads;
    if (size < 0)
        size = std::thread::hardware_concurrency();

    for (auto i = 0; i < size; i++)
        threads.emplace_back([&](int rank) {
            auto block = out_rows / size;
            auto extra = out_rows % size;
            auto start = rank       * block + std::min(rank,     extra);
            auto end   = (rank + 1) * block + std::min(rank + 1, extra);
            block = end - start + 1;

            for (auto x = start; x < end; x++)
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
        }, i);

    std::for_each(
        std::begin(threads),
        std::end(threads),
        std::mem_fn(&std::thread::join)
    );
}

} // namespace opt

namespace opt
{

template <
    class T,
    class Vec,
    std::size_t N
>
void convolve(int rows, int cols, Vec const& din, Vec& dout,
    std::array<std::array<T, N>, N> const& filer, int)
{
    using value_type = typename Vec::value_type;
    int filter_size = N;
    auto out_rows = rows - filter_size + 1;
    auto out_cols = cols - filter_size + 1;
    for (auto x = out_rows; x < rows; x++)
    for (auto y = out_cols; y < cols; y++)
        dout[x * cols + y] = 0;

    // #pragma omp parallel for num_threads(8)
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

