#include <opencv2/opencv.hpp>
#include "lk_feature_tracker.h"
#include <iostream>
#include <filesystem>
#include <ctime>

int main(){
  // Images installed with the package
  const std::string base = std::string(std::getenv("HOME")) + "/vnav_ws/install/lab_3/share/lab_3/images/";
  cv::Mat im1 = cv::imread(base + "greece.jpg", cv::IMREAD_COLOR);
  cv::Mat im2 = cv::imread(base + "greece_in_scene.jpg", cv::IMREAD_COLOR);
  if (im1.empty() || im2.empty()){
    std::cerr<<"Missing demo images in install/share/lab_3/images\n";
    return 1;
  }
  // Ensure equal sizes for LK
  if (im1.size() != im2.size()) { cv::resize(im2, im2, im1.size(), 0, 0, cv::INTER_AREA); }

  // Output dir
  std::string out = std::string(std::getenv("HOME")) + "/vnav_ws/src/TEAM_7/lab3/output/lk_demo_" + std::to_string(std::time(nullptr));
  std::filesystem::create_directories(out);

  LKFeatureTracker lk;

  auto dump = [&](const cv::Mat& f, int k){
    cv::Mat vis = lk.render(f);  // annotated copy
    char p[128]; std::snprintf(p,sizeof(p),"%s/lk_tracks_%04d.png",out.c_str(),k);
    cv::imwrite(p, vis);
  };

  int k=0;
  for(int i=0;i<8;++i){ lk.trackFeatures(im1); dump(im1,k++); }
  for(int i=0;i<8;++i){ lk.trackFeatures(im2); dump(im2,k++); }

  std::cout<<"Saved to: "<<out<<"\n";
  return 0;
}
