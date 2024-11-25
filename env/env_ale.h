#pragma once

#include <SDL2/SDL.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <thread>
#include <memory>
#include <cassert>
#include "env.h"
#include <cmath>
#include <map>
#include <random>
#include <opencv2/opencv.hpp>
#include <ale/ale_interface.hpp>
#include <torch/torch.h>
#include <filesystem>

namespace fs = std::filesystem;

torch::Tensor mat2tensor(const cv::Mat& mat) {
  auto tensor = torch::from_blob(
      mat.data,
      {mat.rows, mat.cols, mat.channels()},
      torch::kUInt8
  );
  if (mat.channels() == 3) {
    tensor = tensor.permute({2, 0, 1});  // HWC -> CHW
  } else {
    tensor = tensor.squeeze(-1);
  }
  return tensor.clone();
}

std::map<std::string, std::vector<int>> name2action_map = {
  {"breakout", std::vector<int>({0, 1, 3, 4})}
};

struct ALEGameOption {
  ALEGameOption(std::string path_rom) : _path_rom(path_rom) {
    action_map = name2action_map[fs::path(path_rom).stem().string()];
  }
  std::string _path_rom;
  double _seed = NAN;
  bool _use_gray = true;
  int _frame_stack = 4;
  int _frame_max_and_skip = 4;
  bool _display_screen = false;
  int _rescale_width = 84;
  int _rescale_height = 84;
  int _min_reward = -1, _max_reward = 1;
  std::vector<int> action_map;
  ALEGameOption &&path_rom(std::string& path_rom) {_path_rom = path_rom; return std::move(*this);}
  ALEGameOption &&seed(double seed) {_seed = seed; return std::move(*this);}
  ALEGameOption &&use_gray(bool use_gray) {_use_gray = use_gray; return std::move(*this);}
  ALEGameOption &&frame_stack(int frame_stack) {_frame_stack = frame_stack; return std::move(*this);}
  ALEGameOption &&frame_max_and_skip(int frame_max_and_skip) {_frame_max_and_skip = frame_max_and_skip; return std::move(*this);}
  ALEGameOption &&display_screen(bool display_screen) {_display_screen = display_screen; return std::move(*this);}
  ALEGameOption &&rescale_width(int rescale_width) {_rescale_width = rescale_width; return std::move(*this);}
  ALEGameOption &&rescale_height(int rescale_height) {_rescale_height = rescale_height; return std::move(*this);}
  ALEGameOption &&min_reward(int min_reward) {_min_reward = min_reward; return std::move(*this);}
  ALEGameOption &&max_reward(int max_reward) {_max_reward = max_reward; return std::move(*this);}
};

class ALEGame: public Env{
 public:
  ALEGame(ALEGameOption option) : ale(option._display_screen), option(option) {
    if (!std::isnan(option._seed)) {
      ale.setInt("random_seed", int(option._seed));
    }
    ale.loadROM(option._path_rom);
    legal_actions = ale.getLegalActionSet();
    width = ale.getScreen().width();
    height = ale.getScreen().height();
    obs_space = EnvSpace(std::vector<int>({option._frame_stack * (option._use_gray ? 1 : 3), option._rescale_height, option._rescale_width}));
    action_space = EnvSpace(std::vector<int>(), int(option.action_map.size()));

    std::vector<int64_t> shape({option._frame_stack, (option._use_gray ? 1 : 3), option._rescale_height, option._rescale_width});
    stack_frames = torch::zeros(shape);
    stack_idx = 0;
    shape[0] = 2;
    max_frames = torch::zeros(shape);
  }

  EnvInfo step(int action) {
    ale_action = legal_actions[option.action_map[action]];
    auto info = get_step_info();
    if (ale.lives() < last_lives) {
      ale.act(legal_actions[1]);  // auto fire after loss life
      ale.act(legal_actions[1]);  // auto fire after loss life
      ale.act(legal_actions[1]);  // auto fire after loss life
      ale.act(legal_actions[1]);  // auto fire after loss life
    }
    last_lives = ale.lives();
    return info;
  }

  EnvInfo reset() {
    ale.reset_game();
    last_lives = ale.lives();
    ale_action = legal_actions[1];  // auto fire after reset
    done = false;
    return get_step_info();
  }

 private:
  const ALEGameOption option;
  int width, height, reward, last_lives;
  bool done;
  int stack_idx;
  
  ale::ALEInterface ale;
  ale::Action ale_action;
  ale::ActionVect legal_actions;
  std::vector<unsigned char> screen;
  torch::Tensor stack_frames;  // use for frame stack
  torch::Tensor max_frames;  // use for frame max and skip

  torch::Tensor get_img() {
    if (option._use_gray) ale.getScreenGrayscale(screen);
    else ale.getScreenRGB(screen);
    // torch::Tensor img = torch::from_blob(
    //   screen.data(), {1, height, width, option._use_gray ? 1 : 3}, torch::kUInt8);
    // img = img.permute({0, 3, 1, 2}).contiguous();
    // img = torch::nn::functional::interpolate(img, torch::nn::functional::InterpolateFuncOptions()
    //   .size(std::vector<int64_t>({option._rescale_height, option._rescale_width})));
    // img = img[0]
    cv::Mat frame(height, width, CV_8UC3, screen.data());
    cv::resize(frame, frame, cv::Size(option._rescale_width, option._rescale_height), cv::INTER_LINEAR);
    if (!option._use_gray) cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    torch::Tensor img = torch::from_blob(
      frame.data,
      {option._rescale_height, option._rescale_width, option._use_gray ? 1 : 3},
      torch::kUInt8);
    img = img.permute({2, 0, 1}).contiguous();
    return img;  // (1, 84, 84)
  }

  torch::Tensor get_max_and_skip() {
    /* get maximum on the last two frame */
    for (int i = 0; i < option._frame_max_and_skip; ++i) {
      reward += std::max(std::min(ale.act(ale_action), option._max_reward), option._min_reward);
      if (i == option._frame_max_and_skip - 2) {
        max_frames[0] = get_img();
      } else if (i == option._frame_max_and_skip - 1) {
        max_frames[1] = get_img();
      }
      done = ale.game_over();
      if (done) break;
    }
    return std::get<0>(max_frames.max(0));
  }

  EnvInfo get_step_info() {
    /* if not is_reset, complete same action interact with environment,
       calculate stack obs and total reward.
    */
    reward = 0;
    stack_frames[stack_idx] = get_max_and_skip();
    (stack_idx += 1) %= option._frame_stack;
    return EnvInfo(
      stack_frames.view({-1, stack_frames.size(-2), stack_frames.size(-1)}),
      reward, done);
  }
};
