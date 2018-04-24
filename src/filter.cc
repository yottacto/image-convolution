// ml:run = time -p $bin ../data/0.jpg
// ml:ldf += -lOpenCL -I/usr/include/opencv -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dpm -lopencv_face -lopencv_photo -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ml -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_flann -lopencv_xobjdetect -lopencv_imgcodecs -lopencv_objdetect -lopencv_xphoto -lopencv_imgproc -lopencv_core

#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "timer.hh"

#define OPEN_IMAGE 1

auto constexpr COLOR_RST = "\e[0m";
auto constexpr COLOR_ATR = "\e[36m";
auto constexpr rep = 10;

int main(int argc, char* argv[])
{
    char const* filename = argc >= 2 ? argv[1] : "../data/0.jpg";
    cv::Mat src, dst;
    if (argc >= 3 && argv[2][0] == 'G')
        src = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    else
        src = cv::imread(filename, cv::IMREAD_COLOR);

    #if OPEN_IMAGE
    cv::namedWindow("Input", cv::WINDOW_NORMAL);
    cv::setWindowProperty("Input", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    cv::imshow("Input", src);
    while (cv::waitKey(1000) != 27 /* esc */) {
    }
    #endif

    cv::Mat kernel = (cv::Mat_<char>(5, 5) <<
        -3,  0, -1,  0,  2,
         0, -1,  0,  2,  0,
        -1,  0,  4,  0, -1,
         0,  2,  0, -1,  0,
         2,  0, -1,  0, -3
    );

    utils::timer t;
    t.start();

    for (auto i = 0; i < rep; i++)
        cv::filter2D(src, dst, src.depth(), kernel);

    t.stop();
    std::cout << "Built-in filter2D elapsed time: "
        << COLOR_ATR
        << t.elapsed_milliseconds()/rep << " ms\n"
        << COLOR_RST;

    #if OPEN_IMAGE
    // cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Output", cv::WINDOW_NORMAL);
    cv::setWindowProperty("Output", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    cv::imshow("Output", dst);
    while (cv::waitKey(1000) != 27 /* esc */) {
    }
    #endif
}

