// ml:run = time -p $bin ../data/2.jpg
// ml:ldf += -lOpenCL -I/usr/include/opencv -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dpm -lopencv_face -lopencv_photo -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ml -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_flann -lopencv_xobjdetect -lopencv_imgcodecs -lopencv_objdetect -lopencv_xphoto -lopencv_imgproc -lopencv_core

#include <iostream>
#include <iterator>
#include <vector>
#include <array>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "timer.hh"
#include "conv.hh"

#define OPEN_IMAGE 0

auto constexpr COLOR_RST = "\e[0m";
auto constexpr COLOR_ATR = "\e[36m";
auto constexpr rep = 200;

using value_type = int;

std::array<std::array<value_type, 5>, 5> constexpr kernel{
    std::array<value_type, 5>{-3,  0, -1,  0,  2},
    std::array<value_type, 5>{ 0, -1,  0,  2,  0},
    std::array<value_type, 5>{-1,  0,  4,  0, -1},
    std::array<value_type, 5>{ 0,  2,  0, -1,  0},
    std::array<value_type, 5>{ 2,  0, -1,  0, -3},
};

// std::array<value_type, 5 * 5> constexpr kernel{
//     -3,  0, -1,  0,  2,
//      0, -1,  0,  2,  0,
//     -1,  0,  4,  0, -1,
//      0,  2,  0, -1,  0,
//      2,  0, -1,  0, -3
// };

int main(int argc, char* argv[])
{
    char const* filename = argc >= 2 ? argv[1] : "data/0.jpg";
    cv::Mat src;
    if (argc >= 3 && argv[2][0] == 'G')
        src = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    else
        src = cv::imread(filename, cv::IMREAD_COLOR);

    auto rows = src.rows;
    auto cols = src.cols;
    auto chan = src.channels();

    std::cout << "image size: " << rows << " * " << cols << "\n";

    #if OPEN_IMAGE
    // cv::namedWindow("Input", cv::WINDOW_AUTOSIZE);
    // cv::imshow("Input", src);
    // while (cv::waitKey(1000) != 27 /* esc */) {
    // }
    #endif

    std::vector<cv::Mat> channels(chan);
    cv::split(src, channels);

    std::vector<std::vector<value_type>> din;
    for (auto i = 0; i < chan; i++)
        din.emplace_back(channels[i].begin<uchar>(), channels[i].end<uchar>());
    auto dout = din;

    utils::timer t;
    t.start();
    // TODO
    for (auto i = 0; i < rep; i++) {
        for (auto i = 0u; i < din.size(); i++)
            opt::convolve(rows, cols, din[i], dout[i], kernel);
    }
    t.stop();

    std::cout << "Hand-written elapsed time: "
        << COLOR_ATR
        << t.elapsed_milliseconds()/rep << "ms\n"
        << COLOR_RST;

    std::vector<cv::Mat> transformed_channels(channels);
    for (auto i = 0; i < chan; i++) {
        auto const& d = dout[i];
        auto it = transformed_channels[i].begin<uchar>();

        for (auto i = 0u; i < d.size(); i++)
            it[i] = static_cast<uchar>(d[i]);
    }

    cv::Mat dst;
    cv::merge(transformed_channels, dst);



    #if OPEN_IMAGE

    cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
    cv::imshow("Output", dst);
    while (cv::waitKey(1000) != 27 /* esc */) {
    }
    #endif
}

