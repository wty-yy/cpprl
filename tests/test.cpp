#include <vector>
#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

torch::Tensor mat2tensor(const cv::Mat& mat) {
  auto tensor = torch::from_blob(
      mat.data,
      {mat.rows, mat.cols, mat.channels()},
      torch::kUInt8
  );
  tensor = tensor.permute({2, 0, 1});  // HWC -> CHW
  return tensor.clone();
}

int main() {
  std::string path_img = "/home/yy/Pictures/vlcsnap-2024-09-22-12h19m29s509.png";
  auto img = cv::imread(path_img, cv::IMREAD_COLOR);
  // auto img = cv::imread(path_img, cv::IMREAD_GRAYSCALE);
  cv::imshow("img", img);
  cv::waitKey(0);
  cv::imwrite("output.png", img);
  auto tensor = mat2tensor(img);
  std::cout << tensor.sizes() << '\n';
  return 0;
}