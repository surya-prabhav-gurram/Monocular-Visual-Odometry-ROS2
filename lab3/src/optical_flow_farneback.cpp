#include <iostream>
#include <string>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

namespace fs = std::filesystem;
using namespace cv;
using std::string;

static Mat flowToColor(const Mat& flow) {
    // Convert 2-channel flow into a BGR visualization (HSV trick)
    Mat flow_parts[2]; split(flow, flow_parts);
    Mat magnitude, angle; cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);

    // Normalize magnitude to [0,1]
    Mat magn_norm; normalize(magnitude, magn_norm, 0.0, 1.0, NORM_MINMAX);

    // Hue: angle, Saturation: 1, Value: magnitude
    Mat _hsv[3];
    _hsv[0] = angle * (1.f/2.f);                 // angle in [0,360] -> [0,180] for OpenCV HSV
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;

    Mat hsv, hsv8, bgr;
    merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cvtColor(hsv8, bgr, COLOR_HSV2BGR);
    return bgr;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  optical_flow_farneback <input_video> [output_dir=deliverables9] [output_video.mp4]\n";
        return 1;
    }
    string inPath = argv[1];
    string outDir = (argc > 2) ? argv[2] : "deliverables9";
    string outVid = (argc > 3) ? argv[3] : "";  // optional

    fs::create_directories(outDir);

    VideoCapture cap(inPath);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: cannot open video: " << inPath << "\n";
        return 2;
    }

    // Read the first frame
    Mat framePrevBGR, prevGray;
    if (!cap.read(framePrevBGR)) {
        std::cerr << "ERROR: empty video\n";
        return 3;
    }
    cvtColor(framePrevBGR, prevGray, COLOR_BGR2GRAY);

    // Optional: prepare writer
    VideoWriter writer;
    if (!outVid.empty()) {
        double fps = cap.get(CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0;
        writer.open(outVid, VideoWriter::fourcc('m','p','4','v'), fps,
                    Size(framePrevBGR.cols, framePrevBGR.rows));
        if (!writer.isOpened()) {
            std::cerr << "WARNING: could not open video writer for " << outVid << "\n";
        }
    }

    int frameIdx = 0;
    while (true) {
        Mat frameBGR, nextGray;
        if (!cap.read(frameBGR)) break;
        cvtColor(frameBGR, nextGray, COLOR_BGR2GRAY);

        Mat flow; // CV_32FC2
        // Farneback default-ish parameters (robust and fast enough)
        calcOpticalFlowFarneback(prevGray, nextGray, flow,
                                 /*pyr_scale*/ 0.5,
                                 /*levels*/    3,
                                 /*winsize*/   15,
                                 /*iterations*/3,
                                 /*poly_n*/    5,
                                 /*poly_sigma*/1.2,
                                 /*flags*/     0);

        Mat flowBGR = flowToColor(flow);

        // Save every 10th frame as PNG
        if (frameIdx % 10 == 0) {
            string pngPath = outDir + "/flow_" + std::to_string(frameIdx) + ".png";
            imwrite(pngPath, flowBGR);
        }

        // Optional write to video
        if (writer.isOpened()) writer.write(flowBGR);

        // Show (press 'q' or ESC to quit)
        imshow("Dense Optical Flow (Farneback)", flowBGR);
        int key = waitKey(1);
        if (key == 'q' || key == 27) break;

        prevGray = nextGray.clone();
        frameIdx++;
    }

    return 0;
}
