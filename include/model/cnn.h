#pragma once

#include "torch/torch.h"
#include "env.h"
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>

/* Used for game size = (4, 84, 84)
*/

using namespace torch;

struct CNNImpl : public nn::Module {
  CNNImpl(std::vector<int> obs_shape, int action_nums) : 
  policy_output(nn::LinearOptions(512, action_nums).bias(false)),
  value_output(nn::LinearOptions(512, 1).bias(false)) {
    feature_cnn->push_back(nn::Conv2d(nn::Conv2dOptions(obs_shape[0], 32, 8).stride(4)));
    feature_cnn->push_back(nn::ReLU());
    feature_cnn->push_back(nn::Conv2d(nn::Conv2dOptions(32, 64, 4).stride(2)));
    feature_cnn->push_back(nn::ReLU());
    feature_cnn->push_back(nn::Conv2d(nn::Conv2dOptions(64, 64, 3).stride(1)));
    feature_cnn->push_back(nn::ReLU());
    feature_cnn->push_back(nn::Flatten());
    feature_cnn->push_back(nn::Linear(64 * 7 * 7, 512));
    feature_cnn->push_back(nn::ReLU());
    register_module("feature_cnn", feature_cnn);
    register_module("policy_output", policy_output);
    register_module("value_output", value_output);
  }

  std::pair<Tensor, Tensor> forward(Tensor x) {
    Tensor z = feature_cnn->forward(x);
    Tensor logist = policy_output->forward(z);
    Tensor value = value_output->forward(z);
    return {logist, value};
  }

  std::tuple<Tensor, Tensor, Tensor, Tensor> get_action_and_value(Tensor x, Tensor action={}) {
    auto [logits, value] = forward(x);
    auto probs = softmax(logits, -1);
    if (action.numel() == 0) {
      action = multinomial(probs, 1).squeeze(1);
    }
    auto logprob = log(probs.gather(-1, action.view({-1, 1}))).squeeze(1);
    auto entropy = -(probs * log(probs + 1e-18)).sum(-1);
    return std::make_tuple(action, logprob, entropy, value);
  }

  Tensor get_value(Tensor x) {
    Tensor z = feature_cnn->forward(x);
    Tensor value = value_output->forward(z);
    return value;
  }
    
  nn::Sequential feature_cnn;
  nn::Linear policy_output;
  nn::Linear value_output;
};

TORCH_MODULE(CNN);

void initialize_cnn_weights(CNN& module) {
  double weight_std = std::sqrt(2);
  double bias_value = 0.0;
  for (auto& child : module->children()) {
    if (auto conv = dynamic_cast<torch::nn::Conv2dImpl*>(child.get())) {
      torch::nn::init::orthogonal_(conv->weight, weight_std);
      if (conv->bias.defined()) {
        torch::nn::init::constant_(conv->bias, bias_value);
      }
    } else if (auto linear = dynamic_cast<torch::nn::LinearImpl*>(child.get())) {
      torch::nn::init::orthogonal_(linear->weight, weight_std);
      if (linear->bias.defined()) {
        torch::nn::init::constant_(linear->bias, bias_value);
      }
    }
  }
  torch::nn::init::orthogonal_(module->policy_output.get()->weight, 0.01);
  torch::nn::init::orthogonal_(module->value_output.get()->weight, 1);
}