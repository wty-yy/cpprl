#include <iostream>
#include <vector>
#include <torch/torch.h>

int main() {
  std::vector<float> a;
  a.push_back(true);
  a.push_back(true);
  a.push_back(false);
  for (auto& x : a) std::cout << x << '\n';
  auto tmp = torch::from_blob(a.data(), a.size(), torch::kFloat32);
  std::cout << tmp << '\n';
  return 0;
}