#include <fmt/core.h>
#include <chrono>
#include <fmt/chrono.h>
#include <fmt/ranges.h>
#include <fmt/os.h>
#include <fmt/color.h>

int main() {
  auto now = std::chrono::system_clock::now();
  std::string s = fmt::format("I'd rather be {1} than {0}.", "right", "happy");
  fmt::print("s={}\n", s);
  fmt::print("Date and time: {}\n", now);
  fmt::print("Time: {:%H:%M}\n", now);
  std::vector<int> v = {1, 2, 3};
  fmt::print("{}\n", v);
  // std::string ss = fmt::format("{:d}", "not a number");
  auto out = fmt::output_file("guide.txt");
  out.print("Don't {}", "Be Sad");
  fmt::print(fg(fmt::color::crimson) | fmt::emphasis::bold, "hello, {}!\n", "world");
}