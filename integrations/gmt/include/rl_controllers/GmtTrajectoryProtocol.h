/*
 * 本文件定义 GENMO 与 GMT 在线控制器之间使用的 trajectory_v1 二进制协议。
 * 它只负责纯数据协议工作：小端字段读写、GMT 策略关节顺序 SHA256、轨迹包
 * 的完整性检查、当前参考提取、21 帧命令窗口生成以及 ACK 编码。协议解析采用
 * “候选数据全部校验通过后再提交”的方式，避免 CRC、维度、NaN、四元数或关节
 * 顺序错误的数据进入控制链路。本文件不包含 Redis 连接、机器人控制或 lowstate
 * 回传逻辑，便于独立单元测试，也避免影响新部署代码现有的 offline 和 legacy 路径。
 */
#pragma once

#include <Eigen/Dense>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace legged
{

enum class MotionRedisProtocolKind : uint8_t
{
  NONE = 0,
  LEGACY_FRAME = 1,
  TRAJECTORY_V1 = 2,
};

namespace gmt_trajectory_detail
{

inline uint32_t rotateRight(uint32_t value, uint32_t count)
{
  return (value >> count) | (value << (32U - count));
}

// 控制器仅为校验 ONNX joint_names 使用 SHA256，内置实现避免额外引入 OpenSSL。
inline std::array<uint8_t, 32> sha256(const std::string& text)
{
  static constexpr uint32_t k[64] = {
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
      0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
      0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
      0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
      0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
      0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
      0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
      0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
      0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
      0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
      0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
      0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
  std::vector<uint8_t> message(text.begin(), text.end());
  const uint64_t bitLength = static_cast<uint64_t>(message.size()) * 8U;
  message.push_back(0x80U);
  while ((message.size() % 64U) != 56U) message.push_back(0U);
  for (int shift = 56; shift >= 0; shift -= 8) {
    message.push_back(static_cast<uint8_t>((bitLength >> shift) & 0xffU));
  }

  uint32_t h[8] = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  for (size_t offset = 0; offset < message.size(); offset += 64U) {
    uint32_t w[64] = {};
    for (int i = 0; i < 16; ++i) {
      const size_t base = offset + static_cast<size_t>(i) * 4U;
      w[i] = (static_cast<uint32_t>(message[base]) << 24U) |
             (static_cast<uint32_t>(message[base + 1U]) << 16U) |
             (static_cast<uint32_t>(message[base + 2U]) << 8U) |
             static_cast<uint32_t>(message[base + 3U]);
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotateRight(w[i - 15], 7U) ^
                          rotateRight(w[i - 15], 18U) ^ (w[i - 15] >> 3U);
      const uint32_t s1 = rotateRight(w[i - 2], 17U) ^
                          rotateRight(w[i - 2], 19U) ^ (w[i - 2] >> 10U);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = rotateRight(e, 6U) ^ rotateRight(e, 11U) ^
                          rotateRight(e, 25U);
      const uint32_t choose = (e & f) ^ ((~e) & g);
      const uint32_t temp1 = hh + s1 + choose + k[i] + w[i];
      const uint32_t s0 = rotateRight(a, 2U) ^ rotateRight(a, 13U) ^
                          rotateRight(a, 22U);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = s0 + majority;
      hh = g; g = f; f = e; e = d + temp1;
      d = c; c = b; b = a; a = temp1 + temp2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }

  std::array<uint8_t, 32> result{};
  for (int i = 0; i < 8; ++i) {
    result[4 * i] = static_cast<uint8_t>((h[i] >> 24U) & 0xffU);
    result[4 * i + 1] = static_cast<uint8_t>((h[i] >> 16U) & 0xffU);
    result[4 * i + 2] = static_cast<uint8_t>((h[i] >> 8U) & 0xffU);
    result[4 * i + 3] = static_cast<uint8_t>(h[i] & 0xffU);
  }
  return result;
}

inline uint16_t readU16(const uint8_t* value)
{
  return static_cast<uint16_t>(value[0]) |
         (static_cast<uint16_t>(value[1]) << 8U);
}

inline uint32_t readU32(const uint8_t* value)
{
  return static_cast<uint32_t>(value[0]) |
         (static_cast<uint32_t>(value[1]) << 8U) |
         (static_cast<uint32_t>(value[2]) << 16U) |
         (static_cast<uint32_t>(value[3]) << 24U);
}

inline uint64_t readU64(const uint8_t* value)
{
  uint64_t result = 0U;
  for (int i = 0; i < 8; ++i) {
    result |= static_cast<uint64_t>(value[i]) << (8U * i);
  }
  return result;
}

inline int64_t readI64(const uint8_t* value)
{
  const uint64_t raw = readU64(value);
  int64_t result;
  std::memcpy(&result, &raw, sizeof(result));
  return result;
}

inline float readF32(const uint8_t* value)
{
  const uint32_t raw = readU32(value);
  float result;
  std::memcpy(&result, &raw, sizeof(result));
  return result;
}

inline void appendU16(std::vector<uint8_t>& output, uint16_t value)
{
  output.push_back(static_cast<uint8_t>(value & 0xffU));
  output.push_back(static_cast<uint8_t>((value >> 8U) & 0xffU));
}

inline void appendU64(std::vector<uint8_t>& output, uint64_t value)
{
  for (int i = 0; i < 8; ++i) {
    output.push_back(static_cast<uint8_t>((value >> (8U * i)) & 0xffU));
  }
}

inline void appendI64(std::vector<uint8_t>& output, int64_t value)
{
  uint64_t raw;
  std::memcpy(&raw, &value, sizeof(raw));
  appendU64(output, raw);
}

inline Eigen::Vector3f projectedGravity(const Eigen::Vector4f& quat)
{
  const float w = quat[0];
  const Eigen::Vector3f qv(quat[1], quat[2], quat[3]);
  const Eigen::Vector3f gravityW(0.0f, 0.0f, -1.0f);
  return gravityW * (2.0f * w * w - 1.0f)
       - qv.cross(gravityW) * (2.0f * w)
       + qv * (2.0f * qv.dot(gravityW));
}

inline Eigen::Vector3f quatToRpy(const Eigen::Vector4f& quat)
{
  const double w = quat[0], x = quat[1], y = quat[2], z = quat[3];
  const double roll = std::atan2(2.0 * (w * x + y * z),
                                 1.0 - 2.0 * (x * x + y * y));
  const double sinPitch = std::max(-1.0, std::min(1.0, 2.0 * (w * y - z * x)));
  const double pitch = std::asin(sinPitch);
  const double yaw = std::atan2(2.0 * (w * z + x * y),
                                1.0 - 2.0 * (y * y + z * z));
  return Eigen::Vector3f(static_cast<float>(roll), static_cast<float>(pitch),
                         static_cast<float>(yaw));
}

}  // namespace gmt_trajectory_detail

inline std::array<uint8_t, 32> gmtJointOrderSha256(
    const std::vector<std::string>& jointNames)
{
  std::string joined;
  for (size_t i = 0; i < jointNames.size(); ++i) {
    if (i != 0U) joined.push_back('\n');
    joined += jointNames[i];
  }
  return gmt_trajectory_detail::sha256(joined);
}

struct GmtTrajectoryAckV1
{
  static constexpr char kMagic[8] = {'O','M','G','B','T','A','0','1'};
  static constexpr uint16_t kVersion = 1;
  static constexpr uint16_t kSize = 52;

  static std::vector<uint8_t> encode(
      uint64_t streamId,
      uint64_t sequence,
      int64_t commandRevision,
      int64_t planId,
      uint64_t receivedUnixNs)
  {
    std::vector<uint8_t> result;
    result.reserve(kSize);
    result.insert(result.end(), kMagic, kMagic + 8);
    gmt_trajectory_detail::appendU16(result, kVersion);
    gmt_trajectory_detail::appendU16(result, kSize);
    gmt_trajectory_detail::appendU64(result, streamId);
    gmt_trajectory_detail::appendU64(result, sequence);
    gmt_trajectory_detail::appendI64(result, commandRevision);
    gmt_trajectory_detail::appendI64(result, planId);
    gmt_trajectory_detail::appendU64(result, receivedUnixNs);
    if (result.size() != kSize) {
      throw std::runtime_error("unexpected GMT trajectory ACK size");
    }
    return result;
  }
};

class GmtTrajectoryV1
{
public:
  static constexpr uint16_t kVersion = 1;
  static constexpr uint16_t kHeaderSize = 104;
  static constexpr uint16_t kJointCount = 21;
  static constexpr uint16_t kFrameDim = 55;
  static constexpr uint16_t kFrameCount = 110;
  static constexpr uint16_t kCurrentIndex = 10;
  static constexpr float kFps = 50.0f;

  explicit GmtTrajectoryV1(
      const std::vector<std::string>& expectedJointNames = {})
  : hasExpectedJointHash_(!expectedJointNames.empty())
  {
    if (hasExpectedJointHash_) {
      if (expectedJointNames.size() != kJointCount) {
        throw std::invalid_argument(
            "trajectory_v1 requires exactly 21 expected GMT joint names");
      }
      expectedJointHash_ = gmtJointOrderSha256(expectedJointNames);
    }
  }

  static bool hasMagic(const std::vector<uint8_t>& blob)
  {
    static constexpr char magic[8] = {'O','M','G','B','T','0','0','1'};
    return blob.size() >= sizeof(magic) &&
           (std::memcmp(blob.data(), magic, sizeof(magic)) == 0 ||
            std::memcmp(blob.data(), "OMGBF001", 8) == 0 ||
            std::memcmp(blob.data(), "OMGBS001", 8) == 0);
  }

  bool parse(const std::vector<uint8_t>& blob, std::string* error = nullptr)
  {
    auto reject = [error](const std::string& message) {
      if (error != nullptr) *error = message;
      return false;
    };
    if (!hasMagic(blob)) return reject("trajectory_v1 magic mismatch");
    if (blob.size() < kHeaderSize) return reject("trajectory_v1 header truncated");
    const bool bufferedClip = std::memcmp(blob.data(), "OMGBF001", 8) == 0;
    const bool bufferedControl = std::memcmp(blob.data(), "OMGBS001", 8) == 0;
    if (!hasExpectedJointHash_) {
      return reject(
          "trajectory_v1 requires GMT policy joint_names for SHA256 validation");
    }

    const uint8_t* header = blob.data();
    const uint16_t version = gmt_trajectory_detail::readU16(header + 8);
    const uint16_t headerSize = gmt_trajectory_detail::readU16(header + 10);
    const uint32_t flags = gmt_trajectory_detail::readU32(header + 12);
    const uint64_t streamId = gmt_trajectory_detail::readU64(header + 16);
    const uint64_t sequence = gmt_trajectory_detail::readU64(header + 24);
    const uint64_t publishedUnixNs = gmt_trajectory_detail::readU64(header + 32);
    const int64_t commandRevision = gmt_trajectory_detail::readI64(header + 40);
    const int64_t planId = gmt_trajectory_detail::readI64(header + 48);
    const float fps = gmt_trajectory_detail::readF32(header + 56);
    const uint16_t frameCount = gmt_trajectory_detail::readU16(header + 60);
    const uint16_t currentIndex = gmt_trajectory_detail::readU16(header + 62);
    const uint16_t jointCount = gmt_trajectory_detail::readU16(header + 64);
    const uint16_t frameDim = gmt_trajectory_detail::readU16(header + 66);
    const uint32_t expectedCrc = gmt_trajectory_detail::readU32(header + 100);

    if (version != kVersion || headerSize != kHeaderSize) {
      return reject("unsupported trajectory_v1 version/header size");
    }
    if ((bufferedClip ? (frameCount == 0 || currentIndex != 0)
                      : (frameCount != kFrameCount || currentIndex != kCurrentIndex)) ||
        jointCount != kJointCount || frameDim != kFrameDim) {
      return reject("trajectory_v1 dimensions/current index do not match GMT contract");
    }
    if (!std::isfinite(fps) || std::fabs(fps - kFps) > 1e-4f) {
      return reject("trajectory_v1 fps must be exactly 50 Hz");
    }
    const size_t payloadBytes = static_cast<size_t>(frameCount) * frameDim * sizeof(float);
    if (blob.size() != static_cast<size_t>(headerSize) + payloadBytes) {
      return reject("trajectory_v1 byte length mismatch");
    }
    if (std::memcmp(header + 68, expectedJointHash_.data(), expectedJointHash_.size()) != 0) {
      return reject("trajectory_v1 GMT joint-order SHA256 mismatch");
    }

    const uint8_t* payload = header + headerSize;
    uLong crc = ::crc32(0L, Z_NULL, 0);
    crc = ::crc32(crc, payload, static_cast<uInt>(payloadBytes));
    if (static_cast<uint32_t>(crc) != expectedCrc) {
      return reject("trajectory_v1 payload CRC32 mismatch");
    }

    std::vector<float> candidate(static_cast<size_t>(frameCount) * frameDim);
    for (size_t i = 0; i < candidate.size(); ++i) {
      candidate[i] = gmt_trajectory_detail::readF32(payload + i * sizeof(float));
      if (!std::isfinite(candidate[i])) {
        return reject("trajectory_v1 payload contains non-finite values");
      }
    }
    for (size_t frame = 0; frame < frameCount; ++frame) {
      const size_t q = frame * frameDim + 3U;
      const float norm = std::sqrt(candidate[q] * candidate[q] +
                                   candidate[q + 1U] * candidate[q + 1U] +
                                   candidate[q + 2U] * candidate[q + 2U] +
                                   candidate[q + 3U] * candidate[q + 3U]);
      if (!std::isfinite(norm) || norm < 0.5f || norm > 1.5f) {
        return reject("trajectory_v1 contains an invalid root quaternion");
      }
      for (size_t element = 0; element < 4U; ++element) {
        candidate[q + element] /= norm;
      }
    }

    frames_.swap(candidate);
    flags_ = flags;
    streamId_ = streamId;
    sequence_ = sequence;
    publishedUnixNs_ = publishedUnixNs;
    commandRevision_ = commandRevision;
    planId_ = planId;
    fps_ = fps;
    frameCount_ = frameCount;
    currentIndex_ = currentIndex;
    valid_ = true;
    bufferedClip_ = bufferedClip;
    bufferedControl_ = bufferedControl;
    if (error != nullptr) error->clear();
    return true;
  }

  void clear()
  {
    valid_ = false;
    streamId_ = 0;
    sequence_ = 0;
    publishedUnixNs_ = 0;
    flags_ = 0;
    commandRevision_ = 0;
    planId_ = -1;
    fps_ = 0.0f;
    frameCount_ = 0;
    currentIndex_ = 0;
    frames_.clear();
    bufferedClip_ = bufferedControl_ = false;
  }

  bool valid() const { return valid_; }
  uint64_t streamId() const { return streamId_; }
  uint64_t sequence() const { return sequence_; }
  uint64_t publishedUnixNs() const { return publishedUnixNs_; }
  uint32_t flags() const { return flags_; }
  int64_t commandRevision() const { return commandRevision_; }
  int64_t planId() const { return planId_; }
  float fps() const { return fps_; }
  bool bufferedClip() const { return bufferedClip_; }
  bool bufferedControl() const { return bufferedControl_; }
  uint16_t frameIndex() const { return currentIndex_; }
  uint16_t frameCount() const { return frameCount_; }

  // 完整缓存仅由 GMT 每次策略回调推进；网络重复包和网络频率不决定动作进度。
  void advanceBufferedFrame()
  {
    if (bufferedClip_ && currentIndex_ + 1U < frameCount_) ++currentIndex_;
  }

  void updateEnvelope(const GmtTrajectoryV1& control)
  {
    sequence_ = control.sequence_;
    publishedUnixNs_ = control.publishedUnixNs_;
    flags_ = control.flags_;
  }

  Eigen::Vector3f rootPosW() const
  {
    const float* row = currentRow();
    return Eigen::Vector3f(row[0], row[1], row[2]);
  }

  Eigen::Vector4f rootQuatWxyz() const
  {
    const float* row = currentRow();
    return Eigen::Vector4f(row[3], row[4], row[5], row[6]);
  }

  Eigen::Vector3f rootLinVelB() const
  {
    const float* row = currentRow();
    return Eigen::Vector3f(row[7], row[8], row[9]);
  }

  Eigen::Vector3f rootAngVelB() const
  {
    const float* row = currentRow();
    return Eigen::Vector3f(row[10], row[11], row[12]);
  }

  Eigen::VectorXf targetJointPos() const
  {
    const float* row = currentRow();
    return Eigen::Map<const Eigen::VectorXf>(row + 13, kJointCount);
  }

  Eigen::VectorXf targetJointVel() const
  {
    const float* row = currentRow();
    return Eigen::Map<const Eigen::VectorXf>(row + 34, kJointCount);
  }

  Eigen::VectorXf commandWindow(int halfWindow, int featureDim) const
  {
    if (!valid_) return Eigen::VectorXf();
    if (halfWindow < 0 || featureDim <= 0) return Eigen::VectorXf();
    const int windowSize = 2 * halfWindow + 1;
    Eigen::VectorXf output = Eigen::VectorXf::Zero(windowSize * featureDim);
    const bool sonicLayout = featureDim == 10 + 2 * static_cast<int>(kJointCount);
    const bool legacyLayout = featureDim == 12 + static_cast<int>(kJointCount);
    if (!sonicLayout && !legacyLayout) return output;

    for (int slot = 0; slot < windowSize; ++slot) {
      int frame = static_cast<int>(currentIndex_) - halfWindow + slot;
      frame = std::max(0, std::min(frame, static_cast<int>(frameCount_) - 1));
      const float* row = frames_.data() + static_cast<size_t>(frame) * kFrameDim;
      const Eigen::Vector4f quat(row[3], row[4], row[5], row[6]);
      int out = slot * featureDim;
      if (sonicLayout) {
        const Eigen::Vector3f gravity = gmt_trajectory_detail::projectedGravity(quat);
        output[out++] = row[2];
        output[out++] = gravity.x();
        output[out++] = gravity.y();
        output[out++] = gravity.z();
        for (int i = 7; i < 13; ++i) output[out++] = row[i];
        for (int i = 13; i < 55; ++i) output[out++] = row[i];
      } else {
        const Eigen::Vector3f rpy = gmt_trajectory_detail::quatToRpy(quat);
        output[out++] = row[0]; output[out++] = row[1]; output[out++] = row[2];
        output[out++] = rpy.x(); output[out++] = rpy.y(); output[out++] = rpy.z();
        for (int i = 7; i < 13; ++i) output[out++] = row[i];
        for (int i = 13; i < 34; ++i) output[out++] = row[i];
      }
    }
    return output;
  }

private:
  const float* currentRow() const
  {
    if (!valid_) throw std::runtime_error("trajectory_v1 has no valid packet");
    return frames_.data() + static_cast<size_t>(currentIndex_) * kFrameDim;
  }

  bool hasExpectedJointHash_ = false;
  std::array<uint8_t, 32> expectedJointHash_{};
  bool valid_ = false;
  bool bufferedClip_ = false;
  bool bufferedControl_ = false;
  uint64_t streamId_ = 0;
  uint64_t sequence_ = 0;
  uint64_t publishedUnixNs_ = 0;
  uint32_t flags_ = 0;
  int64_t commandRevision_ = 0;
  int64_t planId_ = -1;
  float fps_ = 0.0f;
  uint16_t frameCount_ = 0;
  uint16_t currentIndex_ = 0;
  std::vector<float> frames_;
};

}  // namespace legged
