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
#include <random>
#include <opencv2/opencv.hpp>
#include <ale/ale_interface.hpp>
#include <torch/torch.h>

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

class ALEGame: public Env{
 public:
  ALEGame(std::string path_rom, double seed=NAN, bool use_gray=true, 
  int frame_stack=4, bool display_screen=false,
  int rescale_width=84, int rescale_height=84) : 
  ale(display_screen), use_gray(use_gray),
  frame_stack(frame_stack),
  rescale_width(rescale_width), rescale_height(rescale_height) {
    if (!std::isnan(seed)) {
      ale.setInt("random_seed", int(seed));
    }
    ale.loadROM(path_rom);
    legal_actions = ale.getLegalActionSet();
    width = ale.getScreen().width();
    height = ale.getScreen().height();
    obs_space = EnvSpace(std::vector<int>({height, width, 4}));
    action_space = EnvSpace(std::vector<int>(), legal_actions.size());
  }

  EnvInfo step(int action) {
    ale_action = legal_actions[action];
    reward = 0;
    auto info = get_step_info(false);
    if (ale.lives() < last_lives) ale.act(legal_actions[1]);  // auto fire after loss life
    last_lives = ale.lives();
    return info;
  }

  EnvInfo reset() {
    ale.reset_game();
    last_lives = ale.lives();
    ale.act(legal_actions[1]);  // auto fire after reset
    done = false;
    reward = 0;
    return get_step_info(true);
  }

 private:
  int seed, width, height, reward, frame_stack;
  int rescale_width, rescale_height, last_lives;
  bool done, use_gray;
  
  ale::ALEInterface ale;
  ale::Action ale_action;
  ale::ActionVect legal_actions;
  std::vector<unsigned char> screen;

  EnvInfo get_step_info(bool is_reset) {
    /* if not is_reset, complete same action interact with environment,
       calculate stack obs and total reward.
    */
    std::vector<torch::Tensor> obs_vector;
    torch::Tensor tmp;
    int i = 0;
    do {
      if (!done || i == 0) {
        if (use_gray) ale.getScreenGrayscale(screen);
        else ale.getScreenRGB(screen);
        cv::Mat frame(height, width, use_gray ? CV_8UC1 : CV_8UC3, screen.data());
        cv::resize(frame, frame, cv::Size(rescale_width, rescale_height));
        if (!use_gray) cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        tmp = mat2tensor(frame);
      }
      obs_vector.push_back(tmp);
      done = ale.game_over();
      if (done || is_reset) {  // Fill the remaining place with the same frame
        for (int j = i+1; j < frame_stack; ++j)
          obs_vector.push_back(tmp.clone());
        break;
      }
      reward += ale.act(ale_action);
      ++i;
    } while(i < frame_stack);

    torch::Tensor obs = torch::stack(obs_vector);
    return EnvInfo(obs, reward, done);
  }
};
