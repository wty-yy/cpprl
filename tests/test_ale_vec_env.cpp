#include "env_ale.h"
#include "vec_env.h"
#include "opencv2/opencv.hpp"
#include <chrono>
#include <fmt/ranges.h>

std::string path_rom("/home/yy/Coding/datasets/roms/breakout.bin");
FILE* file = fopen("ale_env_speed_test.txt", "a");

/*
EnvInfo return cv::mat
process: 1 time used: 31.089000s
process: 2 time used: 16.830000s
process: 4 time used: 9.917000s
process: 8 time used: 7.964000s
process: 16 time used: 6.890000s
process: 32 time used: 6.749000s
process: 64 time used: 6.436000s
process: 128 time used: 6.092000s
process: 256 time used: 6.648000s
process: 512 time used: 5.993000s
process: 1024 time used: 5.477000s
process: 2048 time used: 5.523000s
process: 4096 time used: 5.620000s
*/

/*
EnvInfo return torch::Tensor, rescale (84, 84), clone tensor, stack 1
process:64, time used: 7.52s SPS=54497.07it/s
EnvInfo return torch::Tensor, no rescale, clone tensor, stack 1
process:64, time used: 13.86s SPS=29548.41it/s
EnvInfo return torch::Tensor, no rescale, no clone tensor, stack 1
process:64, time used: 10.83s SPS=37820.87it/s
EnvInfo return torch::Tensor, rescale (84, 84), no clone tensor, stack 1
process:64, time used: 7.36s SPS=55690.01it/s
EnvInfo return torch::Tensor, rescale (84, 84), clone tensor, stack 4
process:64, time used: 26.18s SPS=15646.73it/s
process: 8, time used: 41.31s, SPS=9914.79it/s
EnvInfo return torch::Tensor, rescale (84, 84), clone tensor, stack 4, VecEnvInfo
process: 8, time used: 43.54s, SPS=9407.44it/s
process: 64, time used: 29.32s, SPS=13968.56it/s
*/

double start_venv_game(int num_envs, int total_steps=100, bool verbose=false) {
  auto venv = VecEnv([](int i){return std::make_shared<ALEGame>(path_rom, i, true, 4);}, num_envs);
  auto infos = venv.reset();
  auto t1 = std::chrono::high_resolution_clock::now();
  auto [obs_space, action_space] = venv.get_space();
  if (verbose) fmt::print("obs_space={}, action_nums={}\n", obs_space.shape, action_space.n);
  for (int i = 0; i < total_steps; ++i) {
    if (verbose) printf("Step: %d\n", i);
    std::vector<int> actions(num_envs, 0);
    for (int j = 0; j < num_envs; ++j) actions[j] = rand() % 5;  // NOOP, FIRE, UP, RIGHT, LEFT
    auto infos = venv.step(actions);
    if (verbose)
      std::cout << "num=" << infos.num_envs << ", obs_shape=" << infos.obs.sizes()
        << ", reward_shape=" << infos.reward.sizes() << ", done_shape=" << infos.done.sizes() << '\n';
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  return 1.0*t3.count()/1e3;
}

int main() {
  int total_steps = 1024*4*100;
  int num_envs = 64;
  double time_used = start_venv_game(num_envs, total_steps/num_envs, true);
  printf("process: %d, time used: %.2lfs, SPS=%.2lfit/s\n", num_envs, time_used, 1.0*total_steps/time_used);
  return 0;
  for (int num_process = 1; num_process <= 4096; num_process *= 2) {
    double time_used = start_venv_game(num_process, 1024*4*100/num_process, false);
    fprintf(file, "process: %d time used: %lfs\n", num_process, time_used);
    fflush(file);
  }
  fclose(file);
  return 0;
}
