#include <iostream>
#include <ale/ale_interface.hpp>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

void saveFrameAsImage(const ale::ALEInterface& ale, const std::string& filename) {
  // 获取屏幕 RGB 数据
  std::vector<unsigned char> screen;
  ale.getScreenRGB(screen);
  auto tmp = ale.getScreen();

  // ALE 图像分辨率
  const int width = tmp.width();
  const int height = tmp.height();

  // OpenCV Mat 构造图像 (高度，宽度，3 通道)
  cv::Mat frame(height, width, CV_8UC3, screen.data());
  cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

  // OpenCV 保存为图片
  if (cv::imwrite(filename, frame)) {
    std::cout << "Image saved to " << filename << std::endl;
  } else {
    std::cerr << "Failed to save image!" << std::endl;
  }
}

void saveFrameFromTensor(const ale::ALEInterface& ale, const std::string& filename) {
  // 获取屏幕 RGB 数据
  std::vector<unsigned char> screen;
  ale.getScreenRGB(screen);
  auto tmp = ale.getScreen();

  // ALE 图像分辨率
  const int width = tmp.width();
  const int height = tmp.height();

  // OpenCV Mat 构造图像 (高度，宽度，3 通道)
  torch::Tensor img = torch::from_blob(screen.data(), {1, height, width, 3}, torch::kUInt8);
  img = img.permute({0, 3, 1, 2});
  // img = torch::nn::functional::interpolate(img, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({84, 84})).mode(torch::kLinear));
  // img = torch::nn::functional::interpolate(img, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({84, 84})).mode(torch::kLinear).align_corners(true));
  img = torch::nn::functional::interpolate(img, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({84, 84})));
  img = img.permute({0, 2, 3, 1}).contiguous();
  cv::Mat frame(img.size(1), img.size(2), CV_8UC3, img.data_ptr<uint8_t>());
  // cv::Mat frame(height, width, CV_8UC3, screen.data());
  // cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

  // OpenCV 保存为图片
  if (cv::imwrite(filename, frame)) {
    std::cout << "Image saved to " << filename << std::endl;
  } else {
    std::cerr << "Failed to save image!" << std::endl;
  }
}

int main(int argc, char** argv) {

  ale::ALEInterface ale(true);
  ale.setInt("random_seed", 123);
  ale.loadROM(std::string("/home/yy/Coding/datasets/roms/breakout.bin"));
  // ale.loadROM("asterix.bin");

  ale::ActionVect legal_actions = ale.getLegalActionSet();

  std::cout << "legal_actions.size=" << legal_actions.size() << '\n';
  ale.reset_game();

  float totalReward = 0.0;
  int count = 0;
  while (!ale.game_over()) {
    // 保存图片 (每次执行前检查)
    // std::string filename = "frame_" + std::to_string(count) + ".jpg";
    std::string filename = "current.jpg";
    // saveFrameAsImage(ale, filename);
    saveFrameFromTensor(ale, filename);

    int act_idx, repeat;  // NOOP, FIRE, UP, RIGHT, LEFT
    std::cin >> act_idx >> repeat;
    // ale::Action a = legal_actions[std::rand() % legal_actions.size()];
    if (act_idx == -1) {  // reset
      ale.reset_game();
      continue;
    }
    for (int i = 0; i < repeat; i++) {
      ale::Action a = legal_actions[act_idx];
      float reward = ale.act(a);
      totalReward += reward;
    }
    count += 1;

    printf("count=%d, score=%f, lives=%d\n", count, totalReward, ale.lives());
  }

  return 0;
}