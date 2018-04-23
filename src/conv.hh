#pragma once
#include <vector>
#include <iterator>
#include <functional>
#include <algorithm>
#include <thread>
#include <array>
#include <omp.h>

namespace meta_tuning
{

namespace impl
{

template <int Size, class T>
struct multi_tmp
{
    using value_type = T;
    using sub_type = multi_tmp<Size - 1, T>;

    multi_tmp(value_type const& value) : value{value}, sub{value} {}

    value_type value;
    sub_type sub;
};

template <class T>
struct multi_tmp<0, T>
{
    using value_type = T;
    multi_tmp(value_type const&) {}
};

// Size is filter size
template <int I, int MAXI, int J, int MAXJ>
struct mult_block
{
    using next = mult_block<I, MAXI, J + 1, MAXJ>;

    template <class Tmp, class MatA, class T, std::size_t Size>
    void operator()(
        Tmp& tmp,
        MatA const& A,
        std::array<std::array<T, Size>, Size> const& B,
        int x, int y, int cols)
    {
        #pragma unroll
        for (auto i = 0u; i < Size; i++)
            #pragma unroll
            for (auto j = 0u; j < Size; j++)
                tmp.value += A[(x + I + i) * cols + y + J + j] * B[i][j];
        next()(tmp.sub, A, B, x, y, cols);
    }

    template <class Tmp, class Mat>
    void update(Tmp& tmp, Mat& C, int x, int y, int cols)
    {
        C[(x + I) * cols + y + J] = tmp.value;
        next().update(tmp.sub, C, x, y, cols);
    }
};

template <int I, int MAXI, int MAXJ>
struct mult_block<I, MAXI, MAXJ, MAXJ>
{
    using next = mult_block<I + 1, MAXI, 0, MAXJ>;

    template <class Tmp, class MatA, class T, std::size_t Size>
    void operator()(
        Tmp& tmp,
        MatA const& A,
        std::array<std::array<T, Size>, Size> const& B,
        int x, int y, int cols)
    {
        #pragma unroll
        for (auto i = 0u; i < Size; i++)
            #pragma unroll
            for (auto j = 0u; j < Size; j++)
                tmp.value += A[(x + I + i) * cols + y + MAXJ + j] * B[i][j];
        next()(tmp.sub, A, B, x, y, cols);
    }

    template <class Tmp, class Mat>
    void update(Tmp& tmp, Mat& C, int x, int y, int cols)
    {
        C[(x + I) * cols + y + MAXJ] = tmp.value;
        next().update(tmp.sub, C, x, y, cols);
    }

};

template <int MAXI, int MAXJ>
struct mult_block<MAXI, MAXI, MAXJ, MAXJ>
{
    template <class Tmp, class MatA, class T, std::size_t Size>
    void operator()(
        Tmp& tmp,
        MatA const& A,
        std::array<std::array<T, Size>, Size> const& B,
        int x, int y, int cols)
    {
        #pragma unroll
        for (auto i = 0u; i < Size; i++)
            #pragma unroll
            for (auto j = 0u; j < Size; j++)
                tmp.value += A[(x + MAXI + i) * cols + y + MAXJ + j] * B[i][j];
    }

    template <class Tmp, class Mat>
    void update(Tmp& tmp, Mat& C, int x, int y, int cols)
    {
        C[(x + MAXI) * cols + y + MAXJ] = tmp.value;
    }
};


} // namespace impl

template <
    int BI,
    int BJ,
    class T,
    class Vec,
    std::size_t N
>
void convolve(int rows, int cols, Vec const& din, Vec& dout,
    std::array<std::array<T, N>, N> const& filter, int size)
{
    using value_type = typename Vec::value_type;
    int constexpr filter_size = N;
    auto out_rows = rows - filter_size + 1;
    auto out_cols = cols - filter_size + 1;
    for (auto x = out_rows; x < rows; x++)
    for (auto y = out_cols; y < cols; y++)
        dout[x * cols + y] = 0;

    size = std::min(size, omp_get_max_threads());

    impl::mult_block<0, BI - 1, 0, BJ - 1> block;

    #pragma omp parallel for num_threads(size)
    for (auto x = 0; x < out_rows; x += BI)
    for (auto y = 0; y < out_cols; y += BJ) {
        impl::multi_tmp<BI * BJ, value_type> sum(value_type{0});
        block(sum, din, filter, x, y, cols);
        block.update(sum, dout, x, y, cols);
    }
}

} // namespace meta_tuning

namespace unroll
{

template <
    int BI,
    int BJ,
    class T,
    class Vec,
    std::size_t N
>
void convolve(int rows, int cols, Vec const& din, Vec& dout,
    std::array<std::array<T, N>, N> const& filter, int size)
{
    using value_type = typename Vec::value_type;
    int constexpr filter_size = N;
    auto out_rows = rows - filter_size + 1;
    auto out_cols = cols - filter_size + 1;
    for (auto x = out_rows; x < rows; x++)
    for (auto y = out_cols; y < cols; y++)
        dout[x * cols + y] = 0;

    size = std::min(size, omp_get_max_threads());

    #pragma omp parallel for num_threads(size)
    for (auto x = 0; x < out_rows; x += BI)
    for (auto y = 0; y < out_cols; y += BJ) {
        value_type sum[BI * BJ] = {value_type{0}};

        #pragma unroll
        for (auto fx = 0; fx < filter_size; fx++)
        #pragma unroll
        for (auto fy = 0; fy < filter_size; fy++) {
            auto fi = filter[fx][fy];

            for (auto i = 0; i < BI; i++)
                for (auto j = 0; j < BJ; j++) {
                    auto di = din[(x + i + fx) * cols + y + j + fy];
                    sum[i * BJ + j] += fi * di;
                }
        }

        for (auto i = 0; i < BI; i++)
            for (auto j = 0; j < BJ; j++) {
                auto s = sum[i * BJ + j];
                if (s < 0) s = 0;
                if (s > 255) s = 255;
                dout[(x + i) * cols + y + j] = s;
            }
    }
}

} // namespace unroll

namespace thread
{

template <
    class T,
    class Vec,
    std::size_t N
>
void convolve(int rows, int cols, Vec const& din, Vec& dout,
    std::array<std::array<T, N>, N> const& filter, int size = -1)
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
                    auto fi = filter[fx][fy];
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

} // namespace thread

namespace omp
{

template <
    int BI,
    int BJ,
    class T,
    class Vec,
    std::size_t N
>
void convolve(int rows, int cols, Vec const& din, Vec& dout,
    std::array<std::array<T, N>, N> const& filter, int size)
{
    using value_type = typename Vec::value_type;
    int filter_size = N;
    auto out_rows = rows - filter_size + 1;
    auto out_cols = cols - filter_size + 1;
    for (auto x = out_rows; x < rows; x++)
    for (auto y = out_cols; y < cols; y++)
        dout[x * cols + y] = 0;

    size = std::min(size, omp_get_max_threads());

    #pragma omp parallel for num_threads(size)
    for (auto x = 0; x < out_rows; x++)
    for (auto y = 0; y < out_cols; y++) {
        auto sum = value_type{0};
        for (auto fx = 0; fx < filter_size; fx++)
        for (auto fy = 0; fy < filter_size; fy++) {
            auto fi = filter[fx][fy];
            auto di = din[(x + fx) * cols + y + fy];
            sum += fi * di;
        }
        if (sum < 0) sum = 0;
        if (sum > 255) sum = 255;
        dout[x * cols + y] = sum;
    }
}

} // namespace omp

namespace opt
{

template <
    class T,
    class Vec,
    std::size_t N
>
void convolve(int rows, int cols, Vec const& din, Vec& dout,
    std::array<std::array<T, N>, N> const& filter, int)
{
    using value_type = typename Vec::value_type;
    int filter_size = N;
    auto out_rows = rows - filter_size + 1;
    auto out_cols = cols - filter_size + 1;
    for (auto x = out_rows; x < rows; x++)
    for (auto y = out_cols; y < cols; y++)
        dout[x * cols + y] = 0;

    for (auto x = 0; x < out_rows; x++)
    for (auto y = 0; y < out_cols; y++) {
        auto sum = value_type{0};
        for (auto fx = 0; fx < filter_size; fx++)
        for (auto fy = 0; fy < filter_size; fy++) {
            auto fi = filter[fx][fy];
            auto di = din[(x + fx) * cols + y + fy];
            sum += fi * di;
        }
        if (sum < 0) sum = 0;
        if (sum > 255) sum = 255;
        dout[x * cols + y] = sum;
    }
}

} // namespace opt

