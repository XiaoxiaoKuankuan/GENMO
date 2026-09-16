/*
 * 本文件验证 GENMO + GMT trajectory_v1 接口及其对新部署代码 legacy Redis 路径
 * 的兼容性。纯协议测试覆盖二进制布局、哈希、CRC、维度、有限值、四元数和命令
 * 窗口；设置 GENMO_GMT_TEST_REDIS_PORT 后，还会连接专用测试 Redis，验证 stream/
 * sequence 接受规则、ACK 只响应新包，以及 legacy causal/centered 窗口没有被新
 * 协议改变。测试只使用独立 key，不启动 ROS 控制器，也不向机器人发送控制命令。
 */
#include "rl_controllers/GmtTrajectoryProtocol.h"
#include "rl_controllers/MotionLoaderNPZ.h"
#include "rl_controllers/MotionLoaderRedis.h"

#include <gtest/gtest.h>
#include <zlib.h>

#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{

const std::vector<std::string> kJointNames = {
    "l_leg_pitch_joint", "r_leg_pitch_joint", "waist_yaw_joint",
    "l_leg_roll_joint", "r_leg_roll_joint", "l_arm_pitch_joint",
    "r_arm_pitch_joint", "l_leg_yaw_joint", "r_leg_yaw_joint",
    "l_arm_roll_joint", "r_arm_roll_joint", "l_knee_pitch_joint",
    "r_knee_pitch_joint", "l_arm_yaw_joint", "r_arm_yaw_joint",
    "l_ankle_pitch_joint", "r_ankle_pitch_joint", "l_elbow_pitch_joint",
    "r_elbow_pitch_joint", "l_ankle_roll_joint", "r_ankle_roll_joint"};

void writeU16(std::vector<uint8_t>& value, size_t offset, uint16_t item)
{
  value[offset] = static_cast<uint8_t>(item & 0xffU);
  value[offset + 1U] = static_cast<uint8_t>((item >> 8U) & 0xffU);
}

void writeU32(std::vector<uint8_t>& value, size_t offset, uint32_t item)
{
  for (int i = 0; i < 4; ++i) value[offset + i] = (item >> (8U * i)) & 0xffU;
}

void writeU64(std::vector<uint8_t>& value, size_t offset, uint64_t item)
{
  for (int i = 0; i < 8; ++i) value[offset + i] = (item >> (8U * i)) & 0xffU;
}

void writeF32(std::vector<uint8_t>& value, size_t offset, float item)
{
  uint32_t raw;
  std::memcpy(&raw, &item, sizeof(raw));
  writeU32(value, offset, raw);
}

void refreshCrc(std::vector<uint8_t>& value)
{
  const size_t payloadBytes = value.size() - legged::GmtTrajectoryV1::kHeaderSize;
  uLong crc = ::crc32(0L, Z_NULL, 0);
  crc = ::crc32(crc, value.data() + legged::GmtTrajectoryV1::kHeaderSize,
                static_cast<uInt>(payloadBytes));
  writeU32(value, 100, static_cast<uint32_t>(crc));
}

std::vector<uint8_t> trajectoryPacket(uint64_t stream = 123U,
                                      uint64_t sequence = 456U,
                                      float valueOffset = 0.0f)
{
  constexpr size_t payloadBytes =
      legged::GmtTrajectoryV1::kFrameCount *
      legged::GmtTrajectoryV1::kFrameDim * sizeof(float);
  std::vector<uint8_t> value(
      legged::GmtTrajectoryV1::kHeaderSize + payloadBytes, 0U);
  const char magic[8] = {'O','M','G','B','T','0','0','1'};
  std::memcpy(value.data(), magic, sizeof(magic));
  writeU16(value, 8, 1);
  writeU16(value, 10, 104);
  writeU32(value, 12, 4);
  writeU64(value, 16, stream);
  writeU64(value, 24, sequence);
  writeU64(value, 32, 789000000000ULL);
  writeU64(value, 40, 7);
  writeU64(value, 48, 8);
  writeF32(value, 56, 50.0f);
  writeU16(value, 60, 110);
  writeU16(value, 62, 10);
  writeU16(value, 64, 21);
  writeU16(value, 66, 55);
  const auto hash = legged::gmtJointOrderSha256(kJointNames);
  std::memcpy(value.data() + 68, hash.data(), hash.size());

  for (int frame = 0; frame < 110; ++frame) {
    float row[55] = {};
    row[0] = valueOffset + static_cast<float>(frame);
    row[2] = valueOffset + 0.5f + static_cast<float>(frame) * 0.01f;
    row[3] = 1.0f;
    row[7] = valueOffset + static_cast<float>(frame);
    row[10] = valueOffset + static_cast<float>(frame) * 2.0f;
    for (int joint = 0; joint < 21; ++joint) {
      row[13 + joint] = valueOffset + static_cast<float>(frame * 100 + joint);
      row[34 + joint] = valueOffset + static_cast<float>(frame * 1000 + joint);
    }
    for (int element = 0; element < 55; ++element) {
      const size_t offset = 104U +
          (static_cast<size_t>(frame) * 55U + element) * sizeof(float);
      writeF32(value, offset, row[element]);
    }
  }
  refreshCrc(value);
  return value;
}

std::vector<uint8_t> legacyPacket(float timestamp, float jointOffset)
{
  constexpr int dof = 21;
  std::vector<float> values(1 + 3 + 4 + 3 + 3 + dof, 0.0f);
  size_t index = 0;
  values[index++] = timestamp;
  values[index++] = 0.0f; values[index++] = 0.0f; values[index++] = 0.5f;
  values[index++] = 1.0f; values[index++] = 0.0f;
  values[index++] = 0.0f; values[index++] = 0.0f;
  index += 6;
  for (int joint = 0; joint < dof; ++joint) {
    values[index++] = jointOffset + static_cast<float>(joint);
  }
  std::vector<uint8_t> blob(values.size() * sizeof(float));
  std::memcpy(blob.data(), values.data(), blob.size());
  return blob;
}

#ifdef RL_CONTROLLERS_HAS_HIREDIS
bool redisSet(redisContext* context, const std::string& key,
              const std::vector<uint8_t>& value)
{
  redisReply* reply = static_cast<redisReply*>(redisCommand(
      context, "SET %s %b", key.c_str(), value.data(), value.size()));
  if (!reply) return false;
  const bool ok = reply->type != REDIS_REPLY_ERROR;
  freeReplyObject(reply);
  return ok;
}

bool redisDelete(redisContext* context, const std::string& key)
{
  redisReply* reply = static_cast<redisReply*>(redisCommand(
      context, "DEL %s", key.c_str()));
  if (!reply) return false;
  const bool ok = reply->type != REDIS_REPLY_ERROR;
  freeReplyObject(reply);
  return ok;
}

std::vector<uint8_t> redisGet(redisContext* context, const std::string& key)
{
  redisReply* reply = static_cast<redisReply*>(redisCommand(
      context, "GET %s", key.c_str()));
  if (!reply) return {};
  std::vector<uint8_t> result;
  if (reply->type == REDIS_REPLY_STRING) {
    result.assign(reinterpret_cast<uint8_t*>(reply->str),
                  reinterpret_cast<uint8_t*>(reply->str) + reply->len);
  }
  freeReplyObject(reply);
  return result;
}
#endif

}  // namespace

TEST(GmtTrajectoryProtocol, HashMatchesGenmoPublisher)
{
  const auto hash = legged::gmtJointOrderSha256(kJointNames);
  std::ostringstream actual;
  for (uint8_t item : hash) {
    actual << std::hex << std::setw(2) << std::setfill('0')
           << static_cast<unsigned int>(item);
  }
  EXPECT_EQ(actual.str(),
            "c339e834953fe4e74daf76b19fdace06866ef49571bd2f849906f26a240d0522");
}

TEST(GmtTrajectoryProtocol, AckBinaryLayoutMatchesGenmoReader)
{
  const auto ack = legged::GmtTrajectoryAckV1::encode(12U, 34U, 5, 6, 7U);
  ASSERT_EQ(ack.size(), 52U);
  const char magic[8] = {'O','M','G','B','T','A','0','1'};
  EXPECT_EQ(std::memcmp(ack.data(), magic, sizeof(magic)), 0);
  EXPECT_EQ(legged::gmt_trajectory_detail::readU16(ack.data() + 8), 1U);
  EXPECT_EQ(legged::gmt_trajectory_detail::readU16(ack.data() + 10), 52U);
  EXPECT_EQ(legged::gmt_trajectory_detail::readU64(ack.data() + 12), 12U);
  EXPECT_EQ(legged::gmt_trajectory_detail::readU64(ack.data() + 20), 34U);
  EXPECT_EQ(legged::gmt_trajectory_detail::readI64(ack.data() + 28), 5);
  EXPECT_EQ(legged::gmt_trajectory_detail::readI64(ack.data() + 36), 6);
  EXPECT_EQ(legged::gmt_trajectory_detail::readU64(ack.data() + 44), 7U);
}

TEST(GmtTrajectoryProtocol, ParsesCurrentFrameAndTwentyOneRealFrames)
{
  legged::GmtTrajectoryV1 trajectory(kJointNames);
  std::string error;
  const auto packet = trajectoryPacket();
  ASSERT_EQ(packet.size(), 24304U);
  ASSERT_TRUE(trajectory.parse(packet, &error)) << error;
  EXPECT_EQ(trajectory.streamId(), 123U);
  EXPECT_EQ(trajectory.sequence(), 456U);
  EXPECT_FLOAT_EQ(trajectory.rootPosW().x(), 10.0f);
  const Eigen::VectorXf window = trajectory.commandWindow(10, 52);
  ASSERT_EQ(window.size(), 1092);
  for (int slot = 0; slot < 21; ++slot) {
    EXPECT_FLOAT_EQ(window(slot * 52), 0.5f + slot * 0.01f);
    EXPECT_FLOAT_EQ(window(slot * 52 + 4), static_cast<float>(slot));
    EXPECT_FLOAT_EQ(window(slot * 52 + 10), static_cast<float>(slot * 100));
    EXPECT_FLOAT_EQ(window(slot * 52 + 31), static_cast<float>(slot * 1000));
  }
}

TEST(GmtTrajectoryProtocol, RejectsEveryStrictContractViolation)
{
  legged::GmtTrajectoryV1 trajectory(kJointNames);
  std::string error;

  auto corruptCrc = trajectoryPacket();
  corruptCrc.back() ^= 0x1U;
  EXPECT_FALSE(trajectory.parse(corruptCrc, &error));
  EXPECT_NE(error.find("CRC32"), std::string::npos);

  auto wrongNames = kJointNames;
  std::swap(wrongNames[0], wrongNames[1]);
  legged::GmtTrajectoryV1 wrongOrder(wrongNames);
  EXPECT_FALSE(wrongOrder.parse(trajectoryPacket(), &error));
  EXPECT_NE(error.find("joint-order"), std::string::npos);

  auto wrongDimensions = trajectoryPacket();
  writeU16(wrongDimensions, 60, 109);
  EXPECT_FALSE(trajectory.parse(wrongDimensions, &error));
  EXPECT_NE(error.find("dimensions"), std::string::npos);

  auto wrongFps = trajectoryPacket();
  writeF32(wrongFps, 56, 30.0f);
  EXPECT_FALSE(trajectory.parse(wrongFps, &error));
  EXPECT_NE(error.find("50 Hz"), std::string::npos);

  auto nonFinite = trajectoryPacket();
  writeF32(nonFinite, 104, std::numeric_limits<float>::quiet_NaN());
  refreshCrc(nonFinite);
  EXPECT_FALSE(trajectory.parse(nonFinite, &error));
  EXPECT_NE(error.find("non-finite"), std::string::npos);

  auto badQuaternion = trajectoryPacket();
  for (int i = 0; i < 4; ++i) writeF32(badQuaternion, 104 + (3 + i) * 4, 0.0f);
  refreshCrc(badQuaternion);
  EXPECT_FALSE(trajectory.parse(badQuaternion, &error));
  EXPECT_NE(error.find("quaternion"), std::string::npos);

  auto shortPacket = trajectoryPacket();
  shortPacket.pop_back();
  EXPECT_FALSE(trajectory.parse(shortPacket, &error));
  EXPECT_NE(error.find("length"), std::string::npos);
}

TEST(MotionLoaderRedisContract, SequenceAckAndLegacyCompatibility)
{
#ifndef RL_CONTROLLERS_HAS_HIREDIS
  GTEST_SKIP() << "hiredis unavailable";
#else
  const char* portText = std::getenv("GENMO_GMT_TEST_REDIS_PORT");
  if (portText == nullptr) GTEST_SKIP() << "Set GENMO_GMT_TEST_REDIS_PORT";
  const int port = std::stoi(portText);
  const std::string motionKey = "gmt_online_frame_bumi_codex_test";
  const std::string ackKey = motionKey + "_ack";
  redisContext* context = redisConnect("127.0.0.1", port);
  ASSERT_NE(context, nullptr);
  ASSERT_EQ(context->err, 0) << context->errstr;
  ASSERT_TRUE(redisDelete(context, motionKey));
  ASSERT_TRUE(redisDelete(context, ackKey));

  legged::MotionLoaderRedis loader(
      "127.0.0.1", port, 0, motionKey, 21, 0.2f, kJointNames, ackKey, 1000);
  ASSERT_TRUE(redisSet(context, motionKey, trajectoryPacket(10U, 20U, 0.0f)));
  loader.update(0.0f);
  ASSERT_EQ(loader.protocolKind(), legged::MotionRedisProtocolKind::TRAJECTORY_V1);
  EXPECT_EQ(loader.streamId(), 10U);
  EXPECT_EQ(loader.sequence(), 20U);
  EXPECT_EQ(loader.commandHistorySize(), 110U);
  EXPECT_TRUE(loader.hasCenteredWindow(10));
  EXPECT_FLOAT_EQ(loader.commandWindow(10, 52, true)(10), 0.0f);

  auto ack = redisGet(context, ackKey);
  ASSERT_EQ(ack.size(), 52U);
  EXPECT_EQ(legged::gmt_trajectory_detail::readU64(ack.data() + 12), 10U);
  EXPECT_EQ(legged::gmt_trajectory_detail::readU64(ack.data() + 20), 20U);

  // 删除 ACK 后重放相同或倒序 sequence；loader 不应替换窗口，也不应重发 ACK。
  ASSERT_TRUE(redisDelete(context, ackKey));
  ASSERT_TRUE(redisSet(context, motionKey, trajectoryPacket(10U, 20U, 1000.0f)));
  loader.update(0.0f);
  EXPECT_FLOAT_EQ(loader.commandWindow(10, 52)(10), 0.0f);
  EXPECT_TRUE(redisGet(context, ackKey).empty());
  ASSERT_TRUE(redisSet(context, motionKey, trajectoryPacket(10U, 19U, 2000.0f)));
  loader.update(0.0f);
  EXPECT_EQ(loader.sequence(), 20U);
  EXPECT_FLOAT_EQ(loader.commandWindow(10, 52)(10), 0.0f);
  EXPECT_TRUE(redisGet(context, ackKey).empty());

  ASSERT_TRUE(redisSet(context, motionKey, trajectoryPacket(10U, 21U, 3000.0f)));
  loader.update(0.0f);
  EXPECT_EQ(loader.sequence(), 21U);
  EXPECT_FLOAT_EQ(loader.commandWindow(10, 52)(10), 3000.0f);
  ASSERT_EQ(redisGet(context, ackKey).size(), 52U);

  // CRC 错误的新 sequence 既不能替换最后合法窗口，也不能刷新 freshness 或 ACK。
  ASSERT_TRUE(redisDelete(context, ackKey));
  std::this_thread::sleep_for(std::chrono::milliseconds(220));
  auto invalidNewPacket = trajectoryPacket(10U, 22U, 3500.0f);
  invalidNewPacket.back() ^= 0x1U;
  ASSERT_TRUE(redisSet(context, motionKey, invalidNewPacket));
  loader.update(0.0f);
  EXPECT_EQ(loader.sequence(), 21U);
  EXPECT_FLOAT_EQ(loader.commandWindow(10, 52)(10), 3000.0f);
  EXPECT_FALSE(loader.hasFreshData());
  EXPECT_TRUE(redisGet(context, ackKey).empty());

  ASSERT_TRUE(redisSet(context, motionKey, trajectoryPacket(11U, 1U, 4000.0f)));
  loader.update(0.0f);
  EXPECT_EQ(loader.streamId(), 11U);
  EXPECT_EQ(loader.sequence(), 1U);
  EXPECT_FLOAT_EQ(loader.commandWindow(10, 52)(10), 4000.0f);

  // reset 后验证原有 legacy causal/centered 窗口仍按旧逻辑工作。
  loader.reset(Eigen::Vector3f::Zero());
  ASSERT_TRUE(redisSet(context, motionKey, legacyPacket(1.0f, 10.0f)));
  loader.update(0.0f);
  ASSERT_TRUE(redisSet(context, motionKey, legacyPacket(2.0f, 20.0f)));
  loader.update(0.0f);
  ASSERT_EQ(loader.protocolKind(), legged::MotionRedisProtocolKind::LEGACY_FRAME);
  const Eigen::VectorXf causal = loader.commandWindow(1, 52, false);
  const Eigen::VectorXf centered = loader.commandWindow(1, 52, true);
  ASSERT_EQ(causal.size(), 156);
  ASSERT_EQ(centered.size(), 156);
  EXPECT_FLOAT_EQ(causal(10), 10.0f);
  EXPECT_FLOAT_EQ(causal(52 + 10), 20.0f);
  EXPECT_FLOAT_EQ(causal(104 + 10), 20.0f);
  EXPECT_FLOAT_EQ(centered(10), 10.0f);
  EXPECT_FLOAT_EQ(centered(52 + 10), 10.0f);
  EXPECT_FLOAT_EQ(centered(104 + 10), 20.0f);

  redisDelete(context, motionKey);
  redisDelete(context, ackKey);
  redisFree(context);
#endif
}

TEST(MotionLoaderRedisContract, ReadsLiveGenmoBridgePacket)
{
#ifndef RL_CONTROLLERS_HAS_HIREDIS
  GTEST_SKIP() << "hiredis unavailable";
#else
  const char* portText = std::getenv("GENMO_GMT_TEST_REDIS_PORT");
  const char* liveGenmo = std::getenv("GENMO_GMT_TEST_LIVE_GENMO");
  if (portText == nullptr || liveGenmo == nullptr || std::string(liveGenmo) != "1") {
    GTEST_SKIP() << "Set GENMO_GMT_TEST_REDIS_PORT and GENMO_GMT_TEST_LIVE_GENMO=1";
  }
  const int port = std::stoi(portText);
  const std::string motionKey = "gmt_online_frame_bumi";
  const std::string ackKey = motionKey + "_ack";
  legged::MotionLoaderRedis loader(
      "127.0.0.1", port, 0, motionKey, 21, 0.2f, kJointNames, ackKey, 1000);
  loader.update(0.0f);
  ASSERT_EQ(loader.protocolKind(), legged::MotionRedisProtocolKind::TRAJECTORY_V1);
  EXPECT_TRUE(loader.hasFreshData());
  EXPECT_GT(loader.streamId(), 0U);
  const Eigen::VectorXf window = loader.commandWindow(10, 52, true);
  ASSERT_EQ(window.size(), 1092);
  EXPECT_TRUE(window.allFinite());

  redisContext* context = redisConnect("127.0.0.1", port);
  ASSERT_NE(context, nullptr);
  ASSERT_EQ(context->err, 0) << context->errstr;
  const auto ack = redisGet(context, ackKey);
  ASSERT_EQ(ack.size(), 52U);
  EXPECT_EQ(legged::gmt_trajectory_detail::readU64(ack.data() + 12),
            loader.streamId());
  EXPECT_EQ(legged::gmt_trajectory_detail::readU64(ack.data() + 20),
            loader.sequence());
  redisFree(context);
#endif
}

TEST(MotionLoaderNpzRegression, LoadsExistingOfflineMotionAndCommandWindow)
{
  const char* motionPath = std::getenv("GENMO_GMT_TEST_OFFLINE_NPZ");
  if (motionPath == nullptr) GTEST_SKIP() << "Set GENMO_GMT_TEST_OFFLINE_NPZ";
  legged::MotionLoaderNPZ loader(motionPath, 0, true);
  EXPECT_GT(loader.numFrames(), 0);
  EXPECT_GT(loader.duration(), 0.0f);
  EXPECT_GT(loader.dt(), 0.0f);
  loader.reset(Eigen::Vector3f::Zero(), 0.0f);
  loader.update(0.0f);
  const Eigen::VectorXf window = loader.commandWindow(0.0f, 10, 52);
  ASSERT_EQ(window.size(), 1092);
  EXPECT_TRUE(window.allFinite());
  EXPECT_EQ(loader.targetJointPos().size(), 21);
  EXPECT_EQ(loader.targetJointVel().size(), 21);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
