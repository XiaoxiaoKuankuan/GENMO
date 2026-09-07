/*
 * BUMI 整段缓存协议的独立 C++ 验证驱动。
 * 直接编译实际部署 MotionLoaderRedis，连接测试专用 Redis。标准输入的 tick 代表
 * 一次策略更新，peek 只读取状态而不推进；标准输出 RESULT 返回真实消费端的帧号
 * 和 21×52 指令窗。驱动不加载 GMT 模型、不连接 ROS，也不驱动仿真或实机。
 */
#include "rl_controllers/MotionLoaderRedis.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc != 4 && argc != 5) return 2;
    std::vector<std::string> names;
    if (argc == 5) {
        std::ifstream input(argv[4]);
        std::string name;
        while (std::getline(input, name)) names.push_back(name);
    } else {
        for (int i = 0; i < 21; ++i) names.push_back("joint_" + std::to_string(i));
    }
    legged::MotionLoaderRedis loader("127.0.0.1", std::stoi(argv[1]), 0, argv[2],
                                    21, 0.2f, names, "", 1000, std::stoi(argv[3]) != 0);
    std::cout << "READY" << std::endl;
    std::string command;
    float time = 0;
    while (std::cin >> command) {
        if (command == "tick") { time += 0.02f; loader.update(time); }
        else if (command == "reset") loader.reset(Eigen::Vector3f::Zero(), 0);
        else if (command != "peek") return 3;
        std::cout << "RESULT " << loader.hasFreshData() << " " << loader.frameSequence();
        if (loader.hasFreshData()) {
            const auto window = loader.commandWindow(10, 52);
            for (int i = 0; i < window.size(); ++i)
                std::cout << " " << std::setprecision(9) << window[i];
        }
        std::cout << std::endl;
    }
    return 0;
}
