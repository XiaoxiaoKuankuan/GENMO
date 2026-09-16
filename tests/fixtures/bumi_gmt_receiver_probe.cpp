/*
 * GENMO 部署验收使用的独立 GMT Redis 接收探针。
 * 编译时包含用户指定 GMT 工作区的真实 MotionLoaderRedis.h，不复制或改写协议实现。
 * 只连接命令行指定的隔离 Redis 端口/key，构造真实 1092 维 command window 并回 ACK；
 * 不链接 ROS 控制器、不加载机器人策略、不连接硬件。输出 JSON 行用于验收脚本记录
 * 接收数量、centered-delay 绕过一致性和轨迹过期状态，SIGTERM 后退出便于自动清理。
 */
#include "rl_controllers/MotionLoaderRedis.h"
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {
volatile std::sig_atomic_t stopped = 0;
void stop(int) { stopped = 1; }
}

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  const int port = std::stoi(argv[1]);
  if (port <= 0 || port == 6379 || port > 65535) return 2;
  std::vector<std::string> names;
  std::ifstream input(argv[3]);
  std::string name;
  while (std::getline(input, name)) if (!name.empty()) names.push_back(name);
  if (names.size() != 21) return 2;
  std::signal(SIGTERM, stop);
  std::signal(SIGINT, stop);
  legged::MotionLoaderRedis loader("127.0.0.1", port, 0, argv[2], 21, 0.2f,
                                   names, std::string(argv[2]) + "_ack", 1000);
  uint64_t stream = 0, sequence = 0, accepted = 0;
  bool stale = false;
  while (!stopped) {
    loader.update(0.0f);
    if (loader.protocolKind() == legged::MotionRedisProtocolKind::TRAJECTORY_V1) {
      if (stream != loader.streamId() || sequence != loader.sequence()) {
        stream = loader.streamId(); sequence = loader.sequence(); ++accepted;
        const auto ordinary = loader.commandWindow(10, 52, false);
        const auto centered = loader.commandWindow(10, 52, true);
        if (ordinary.size() != 1092 || !ordinary.allFinite() ||
            !ordinary.isApprox(centered, 0.0f)) return 3;
        if (accepted == 1 || accepted % 50 == 0)
          std::cout << "{\"accepted\":" << accepted
                    << ",\"window_size\":1092,\"finite\":true,\"centered_equal\":true}"
                    << std::endl;
      }
      if (accepted > 0 && !loader.hasFreshData() && !stale) {
        stale = true;
        std::cout << "{\"stale_detected\":true}" << std::endl;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::cout << "{\"accepted_total\":" << accepted << ",\"stale_detected\":"
            << (stale ? "true" : "false") << "}" << std::endl;
  return accepted > 0 ? 0 : 4;
}
