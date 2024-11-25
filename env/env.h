#pragma once

#include <vector>
#include <memory>
#include <torch/torch.h>

struct EnvSpace {
  EnvSpace(){}
  EnvSpace(std::vector<int> shape, double n=NAN) : shape(shape) {
    if (std::isnan(n)) {
      n = 1;
      for (auto& x : shape) n *= x;
      this->n = int(n);
    }
  }
  std::vector<int> shape;
  int n;
};

struct EnvInfo {
  EnvInfo(){}
  EnvInfo(torch::Tensor obs, double reward, bool done):obs(obs), reward(reward), done(done){}
  torch::Tensor obs;
  double reward;
  bool done;
};

struct Env {
 public:
  Env(){}
  Env(EnvSpace obs_space, EnvSpace action_space) : obs_space(obs_space), action_space(action_space) {}
  virtual EnvInfo step(int) = 0;  // 离散动作空间
  virtual EnvInfo reset() = 0;
  virtual ~Env() = default;
  std::pair<EnvSpace, EnvSpace> get_space() {
    return {obs_space, action_space};
  }
 protected:
  EnvSpace obs_space, action_space;
};
