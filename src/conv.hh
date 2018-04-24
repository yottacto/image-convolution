#pragma once
#include <vector>
#include <iterator>
#include <functional>
#include <algorithm>
#include <thread>
#include <array>
#include <immintrin.h>
#include <omp.h>

namespace vectorization
{

namespace detail
{

template <typename T>
constexpr T sqrt_helper(T x, T lo, T hi)
{
    if (lo == hi)
        return lo;

    const T mid = (lo + hi + 1) / 2;

    if (x / mid < mid)
        return sqrt_helper<T>(x, lo, mid - 1);
    else
        return sqrt_helper(x, mid, hi);
}

template <typename T>
constexpr T ct_sqrt(T x)
{
    return sqrt_helper<T>(x, 0, x / 2 + 1);
}

} // namespace detail

template <
    int BI,
    int BJ,
    class T,
    class Vec,
    std::size_t N
>
void convolve(int rows, int cols, Vec const& din, Vec& dout,
    std::array<T, N> const& filter, int size)
{
    int constexpr filter_size = detail::ct_sqrt(N);
    auto out_rows = rows - filter_size + 1;
    auto out_cols = cols - filter_size + 1;
    for (auto x = out_rows; x < rows; x++)
    for (auto y = out_cols; y < cols; y++)
        dout[x * cols + y] = 0;

    size = std::min(size, omp_get_max_threads());

    #pragma omp parallel for schedule(auto) num_threads(size)
    for (auto x = 0; x < out_rows; x += BI)
    for (auto y = 0; y < out_cols; y += BJ * 16) {
        __m512i sum[BI * BJ] = {_mm512_setzero_epi32()};

        #pragma unroll
        for (auto fx = 0; fx < filter_size; fx++)
        #pragma unroll
        for (auto fy = 0; fy < filter_size; fy++) {
            __m512i fi = _mm512_load_epi32(filter.data() + fx * filter_size + fy);


            for (auto i = 0; i < BI; i++)
            for (auto j = 0; j < BJ; j++) {
                __m512i di = _mm512_load_epi32(din.data()
                    + (x + fx + i) * cols + y + fy + j * 8);
                sum[i * BJ + j] = _mm512_add_epi32(
                    _mm512_mul_epi32(fi, di),
                    sum[i * BJ + j]
                );
            }
        }

        for (auto i = 0; i < BI; i++)
            for (auto j = 0; j < BJ; j++) {
                // FIXME
                // if (s < 0) s = 0;
                // if (s > 255) s = 255;

                _mm512_store_epi32(
                    dout.data() + (x + i) * cols + y + j * 8,
                    sum[i * BJ + j]
                );
            }
    }
}

} // namespace loop_tiling


namespace meta_tuning
{

namespace impl
{

template <int Size, class Value>
struct multi_tmp
{
    using sub_type = multi_tmp<Size - 1, Value>;

    multi_tmp(Value const& value) : value{value}, sub{value} {}

    Value value;
    sub_type sub;
};

template <class Value>
struct multi_tmp<0, Value>
{
    multi_tmp(Value const&) {}
};

// Size is filter size
template <int I, int MAXI, int J, int MAXJ>
struct mult_block
{
    using next = mult_block<I, MAXI, J + 1, MAXJ>;

    template <class Tmp, class MatA, class T>
    void operator()(
        Tmp& tmp,
        MatA const& A,
        T Fij,
        int x, int y, int i, int j, int cols)
    {
        tmp.value += A[(x + I + i) * cols + y + J + j] * Fij;
        next{}(tmp.sub, A, Fij, x, y, i, j, cols);
    }

    template <class Tmp, class Mat>
    void update(Tmp& tmp, Mat& C, int x, int y, int cols)
    {
        if (tmp.value < 0  ) tmp.value = 0;
        if (tmp.value > 255) tmp.value = 255;
        C[(x + I) * cols + y + J] = tmp.value;
        next{}.update(tmp.sub, C, x, y, cols);
    }
};

template <int I, int MAXI, int MAXJ>
struct mult_block<I, MAXI, MAXJ, MAXJ>
{
    using next = mult_block<I + 1, MAXI, 0, MAXJ>;

    template <class Tmp, class MatA, class T>
    void operator()(
        Tmp& tmp,
        MatA const& A,
        T Fij,
        int x, int y, int i, int j, int cols)
    {
        tmp.value += A[(x + I + i) * cols + y + MAXJ + j] * Fij;
        next{}(tmp.sub, A, Fij, x, y, i, j, cols);
    }

    template <class Tmp, class Mat>
    void update(Tmp& tmp, Mat& C, int x, int y, int cols)
    {
        if (tmp.value < 0  ) tmp.value = 0;
        if (tmp.value > 255) tmp.value = 255;
        C[(x + I) * cols + y + MAXJ] = tmp.value;
        next{}.update(tmp.sub, C, x, y, cols);
    }

};

template <int MAXI, int MAXJ>
struct mult_block<MAXI, MAXI, MAXJ, MAXJ>
{
    template <class Tmp, class MatA, class T>
    void operator()(
        Tmp& tmp,
        MatA const& A,
        T Fij,
        int x, int y, int i, int j, int cols)
    {
        tmp.value += A[(x + MAXI + i) * cols + y + MAXJ + j] * Fij;
    }

    template <class Tmp, class Mat>
    void update(Tmp& tmp, Mat& C, int x, int y, int cols)
    {
        if (tmp.value < 0  ) tmp.value = 0;
        if (tmp.value > 255) tmp.value = 255;
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
    using value_type = T;
    int constexpr filter_size = N;
    auto out_rows = rows - filter_size + 1;
    auto out_cols = cols - filter_size + 1;
    for (auto x = out_rows; x < rows; x++)
    for (auto y = out_cols; y < cols; y++)
        dout[x * cols + y] = 0;

    size = std::min(size, omp_get_max_threads());

    impl::mult_block<0, BI - 1, 0, BJ - 1> block;

    #pragma omp parallel for schedule(auto) num_threads(size) private(block)
    for (auto x = 0; x < out_rows; x += BI)
    for (auto y = 0; y < out_cols; y += BJ) {
        impl::multi_tmp<BI * BJ, value_type> sum(value_type{0});

        #pragma unroll
        for (auto i = 0; i < filter_size; i++)
            #pragma unroll
            for (auto j = 0; j < filter_size; j++)
                block(sum, din, filter[i][j], x, y, i, j, cols);

        block.update(sum, dout, x, y, cols);
    }
}

} // namespace meta_tuning

namespace loop_tiling
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
    using value_type = T;
    int constexpr filter_size = N;
    auto out_rows = rows - filter_size + 1;
    auto out_cols = cols - filter_size + 1;
    for (auto x = out_rows; x < rows; x++)
    for (auto y = out_cols; y < cols; y++)
        dout[x * cols + y] = 0;

    size = std::min(size, omp_get_max_threads());

    #pragma omp parallel for schedule(auto) num_threads(size)
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
                auto const tid = (x + i + fx) * cols + y + j + fy;

                // _MM_HINT_T0, _MM_HINT_T1, _MM_HINT_T2, _MM_HINT_NTA
                // _mm_prefetch((const char*)(din[tid + 5]), _MM_HINT_T1);
                // _mm_prefetch((const char*)(din[tid + 2]), _MM_HINT_T1);

                auto di = din[tid];
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

} // namespace loop_tiling

namespace thread
{

template <
    int BI,
    int BJ,
    class T,
    class Vec,
    std::size_t N
>
void convolve(int rows, int cols, Vec const& din, Vec& dout,
    std::array<std::array<T, N>, N> const& filter, int size = -1)
{
    using value_type = T;
    int filter_size = N;
    auto out_rows = rows - filter_size + 1;
    auto out_cols = cols - filter_size + 1;
    for (auto x = out_rows; x < rows; x++)
    for (auto y = out_cols; y < cols; y++)
        dout[x * cols + y] = 0;

    if (size < 0)
        size = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

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
    using value_type = T;
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

namespace basic
{

template <
    int BI,
    int BJ,
    class T,
    class Vec,
    std::size_t N
>
void convolve(int rows, int cols, Vec const& din, Vec& dout,
    std::array<std::array<T, N>, N> const& filter, int)
{
    using value_type = T;
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

} // namespace basic

