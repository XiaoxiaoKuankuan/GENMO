/*
 * 同款 GMT 接入 GENMO 的 Redis 接收器移植副本。
 * 从已验证的 obs 工作区提取，配套 GmtTrajectoryProtocol.h 使用；保留原单帧输入兼容。
 * 负责接收和校验 trajectory_v1、保存真实过去/当前/未来轨迹、发布 ACK 与检测断流。
 * 此文件不加载 GENMO、不执行机器人 policy；AcController 中的接入步骤见同目录 README。
 * 来源指纹记录于 SOURCE_MANIFEST.json；本副本只增加本说明，原接收逻辑保持逐字一致。
 */
#pragma once

#include "rl_controllers/GmtTrajectoryProtocol.h"

#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#ifdef RL_CONTROLLERS_HAS_HIREDIS
#include <hiredis/hiredis.h>
#endif

namespace legged
{

// ============================================================
//  MotionLoaderRedis
//  从 Redis 实时读取 GMT motion 帧
//
//  自动识别两种协议：
//    - legacy float32 单帧：保留原有 causal/centered-delay 窗口；
//    - trajectory_v1：直接使用 GENMO 包内真实过去/当前/未来帧。
//
//  首帧自动对齐：
//    - xy 偏移到机器人当前位置（默认原点）
//    - yaw 旋转到 0°
//    - z 直接使用 mocap 值（不偏移）
// ============================================================

#ifdef RL_CONTROLLERS_HAS_HIREDIS
class MotionLoaderRedis
{
public:
    MotionLoaderRedis(const std::string& host,
                      int port,
                      int db,
                      const std::string& key,
                      int dof = 24,
                      float timeout_s = 0.2f,
                      const std::vector<std::string>& expected_joint_names = {},
                      const std::string& ack_key = "",
                      int ack_ttl_ms = 1000,
                      bool allow_buffered = false)
    : host_(host), port_(port), db_(db), dof_(dof), key_(key),
      ack_key_(ack_key.empty() ? key + "_ack" : ack_key),
      ack_ttl_ms_(std::max(1, ack_ttl_ms)), timeout_s_(timeout_s),
      trajectory_(expected_joint_names), expected_joint_names_(expected_joint_names),
      allow_buffered_(allow_buffered)
    {
        joint_pos_.resize(dof_);
        joint_pos_.setZero();
        joint_vel_.resize(dof_);
        joint_vel_.setZero();
        prev_joint_pos_.resize(dof_);
        prev_joint_pos_.setZero();
        connect_();
    }

    ~MotionLoaderRedis() { disconnect_(); }

    // 拉取最新帧（非阻塞）
    void update(float /*time_sec*/)
    {
        std::vector<uint8_t> blob;
        if (!getBlob_(blob)) return;
        parseFrame_(blob);
    }

    void reset(const Eigen::Vector3f& robot_root_pos_w, float /*t*/ = 0.0f)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        robot_origin_ = robot_root_pos_w;
        aligned_ = false;  // 下一帧重新对齐
        has_data_ = false;
        has_prev_frame_ = false;
        prev_timestamp_ = 0.0f;
        prev_joint_pos_.setZero();
        command_history_.clear();
        source_timestamp_ = -1.0f;
        frame_sequence_ = 0;
        protocol_kind_ = MotionRedisProtocolKind::NONE;
        trajectory_.clear();
        trajectory_stream_id_ = 0;
        trajectory_sequence_ = 0;
        has_trajectory_sequence_ = false;
        last_protocol_error_.clear();
    }

    // -------- getters --------

    Eigen::Vector3f rootPosW() const
    {
        std::lock_guard<std::mutex> lk(mtx_);
        return pos_w_;
    }

    Eigen::Vector4f rootQuatWxyz() const
    {
        std::lock_guard<std::mutex> lk(mtx_);
        return quat_wxyz_;
    }

    Eigen::Vector3f rootLinVelB() const
    {
        std::lock_guard<std::mutex> lk(mtx_);
        return lin_vel_b_;
    }

    Eigen::Vector3f rootAngVelB() const
    {
        std::lock_guard<std::mutex> lk(mtx_);
        return ang_vel_b_;
    }

    Eigen::VectorXf targetJointPos() const
    {
        std::lock_guard<std::mutex> lk(mtx_);
        return joint_pos_;
    }

    Eigen::VectorXf targetJointVel() const
    {
        std::lock_guard<std::mutex> lk(mtx_);
        return joint_vel_;
    }

    MotionRedisProtocolKind protocolKind() const
    {
        std::lock_guard<std::mutex> lk(mtx_);
        return protocol_kind_;
    }

    uint64_t streamId() const
    {
        std::lock_guard<std::mutex> lk(mtx_);
        return protocol_kind_ == MotionRedisProtocolKind::TRAJECTORY_V1
            ? trajectory_stream_id_ : 0U;
    }

    uint64_t sequence() const
    {
        std::lock_guard<std::mutex> lk(mtx_);
        return protocol_kind_ == MotionRedisProtocolKind::TRAJECTORY_V1
            ? trajectory_sequence_ : frame_sequence_;
    }

    // Online command window. In normal causal mode it contains real Pico
    // history + current + repeated current for the unavailable future.
    // In centered-delay mode the latest 2*half_window+1 real frames are used,
    // so the middle frame has true past and future context just like NPZ.
    // Centered-delay mode adds half_window Redis frames of reference latency.
    Eigen::VectorXf commandWindow(int half_window, int command_dim,
                                  bool centered_delay = false) const
    {
        const int window_size = 2 * half_window + 1;
        const int sonic_command_dim = 10 + 2 * dof_;
        if (half_window < 0 || command_dim <= 0) {
            return Eigen::VectorXf();
        }

        Eigen::VectorXf window(window_size * command_dim);
        window.setZero();

        std::lock_guard<std::mutex> lk(mtx_);
        if (protocol_kind_ == MotionRedisProtocolKind::TRAJECTORY_V1) {
            return trajectory_.commandWindow(half_window, command_dim);
        }
        if (command_dim != sonic_command_dim) return Eigen::VectorXf();
        if (command_history_.empty()) return window;

        if (centered_delay) {
            const int available = std::min(window_size,
                static_cast<int>(command_history_.size()));
            const int missing = window_size - available;
            const int history_begin = static_cast<int>(command_history_.size()) - available;
            const Eigen::VectorXf& oldest = command_history_[history_begin];

            // Match NPZ boundary clamping while the delayed history warms up.
            for (int slot = 0; slot < missing; ++slot) {
                window.segment(slot * command_dim, command_dim) = oldest;
            }
            for (int i = 0; i < available; ++i) {
                window.segment((missing + i) * command_dim, command_dim) =
                    command_history_[history_begin + i];
            }
            return window;
        }

        const int required_history = half_window + 1;  // past + current
        const int available = std::min(required_history,
            static_cast<int>(command_history_.size()));
        const int missing = required_history - available;
        const int history_begin = static_cast<int>(command_history_.size()) - available;
        const Eigen::VectorXf& oldest = command_history_[history_begin];
        const Eigen::VectorXf& current = command_history_.back();

        // Startup padding: repeat the oldest real frame on the missing past side.
        for (int slot = 0; slot < missing; ++slot) {
            window.segment(slot * command_dim, command_dim) = oldest;
        }
        for (int i = 0; i < available; ++i) {
            window.segment((missing + i) * command_dim, command_dim) =
                command_history_[history_begin + i];
        }
        // Pico has no future frames, so hold the current complete feature.
        for (int slot = required_history; slot < window_size; ++slot) {
            window.segment(slot * command_dim, command_dim) = current;
        }
        return window;
    }

    bool hasFreshData() const
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!has_data_) return false;
        auto elapsed = std::chrono::steady_clock::now() - last_recv_tp_;
        float secs = std::chrono::duration<float>(elapsed).count();
        return secs < timeout_s_;
    }

    bool hasCenteredWindow(int half_window) const
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (protocol_kind_ == MotionRedisProtocolKind::TRAJECTORY_V1) {
            return trajectory_.valid();
        }
        return half_window >= 0 &&
            command_history_.size() >= static_cast<size_t>(half_window + 1);
    }

    float dataAgeSec() const
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!has_data_) return -1.0f;
        return std::chrono::duration<float>(
            std::chrono::steady_clock::now() - last_recv_tp_).count();
    }

    double sourceTimestamp() const
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (protocol_kind_ == MotionRedisProtocolKind::TRAJECTORY_V1) {
            return static_cast<double>(trajectory_.publishedUnixNs()) * 1e-9;
        }
        return source_timestamp_;
    }

    uint64_t frameSequence() const
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (protocol_kind_ == MotionRedisProtocolKind::TRAJECTORY_V1) {
            if (trajectory_.bufferedClip()) return trajectory_.frameIndex();
            return trajectory_sequence_;
        }
        return frame_sequence_;
    }

    size_t commandHistorySize() const
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (protocol_kind_ == MotionRedisProtocolKind::TRAJECTORY_V1) {
            return GmtTrajectoryV1::kFrameCount;
        }
        return command_history_.size();
    }

private:
    std::string host_;
    int port_, db_, dof_;
    std::string key_;
    std::string ack_key_;
    int ack_ttl_ms_;
    float timeout_s_;

    redisContext* ctx_ = nullptr;
    mutable std::mutex mtx_;

    bool has_data_ = false;
    Eigen::Vector3f pos_w_{0,0,0};
    Eigen::Vector4f quat_wxyz_{1,0,0,0};
    Eigen::Vector3f lin_vel_b_{0,0,0};
    Eigen::Vector3f ang_vel_b_{0,0,0};
    Eigen::VectorXf joint_pos_;
    Eigen::VectorXf joint_vel_;
    Eigen::VectorXf prev_joint_pos_;
    std::deque<Eigen::VectorXf> command_history_;
    static constexpr size_t kMaxCommandHistory_ = 64;
    bool has_prev_frame_ = false;
    float prev_timestamp_ = 0.0f;
    double source_timestamp_ = -1.0;
    uint64_t frame_sequence_ = 0;
    std::chrono::steady_clock::time_point last_recv_tp_;
    MotionRedisProtocolKind protocol_kind_ = MotionRedisProtocolKind::NONE;
    GmtTrajectoryV1 trajectory_;
    std::vector<std::string> expected_joint_names_;
    bool allow_buffered_ = false;
    uint64_t trajectory_stream_id_ = 0;
    uint64_t trajectory_sequence_ = 0;
    bool has_trajectory_sequence_ = false;
    std::string last_protocol_error_;

    // ── Alignment state ──────────────────────────────────────────────────
    bool            aligned_ = false;
    Eigen::Vector3f robot_origin_{0, 0, 0};   // 机器人初始位置
    Eigen::Vector3f pos_offset_{0, 0, 0};     // xy 偏移
    float           yaw_offset_ = 0.0f;       // yaw 偏移
    Eigen::Vector4f yaw_fix_quat_{1, 0, 0, 0}; // yaw 修正四元数

    // ── Connection ───────────────────────────────────────────────────────
    bool connect_()
    {
        ctx_ = redisConnect(host_.c_str(), port_);
        if (!ctx_ || ctx_->err) {
            printf("[MotionLoaderRedis] connect failed: %s\n",
                   ctx_ ? ctx_->errstr : "null");
            return false;
        }
        if (db_ != 0) {
            redisReply* r = (redisReply*)redisCommand(ctx_, "SELECT %d", db_);
            if (r) freeReplyObject(r);
        }
        printf("[MotionLoaderRedis] connected to %s:%d db=%d key=%s ack_key=%s\n",
               host_.c_str(), port_, db_, key_.c_str(), ack_key_.c_str());
        return true;
    }

    void disconnect_()
    {
        if (ctx_) { redisFree(ctx_); ctx_ = nullptr; }
    }

    bool getBlob_(std::vector<uint8_t>& out)
    {
        return getBlobAt_(key_, out);
    }

    bool getBlobAt_(const std::string& key, std::vector<uint8_t>& out)
    {
        if (!ctx_) { connect_(); return false; }
        redisReply* r = (redisReply*)redisCommand(ctx_, "GET %s", key.c_str());
        if (!r) { disconnect_(); return false; }
        bool ok = false;
        if (r->type == REDIS_REPLY_STRING) {
            out.assign((uint8_t*)r->str, (uint8_t*)r->str + r->len);
            ok = true;
        }
        freeReplyObject(r);
        return ok;
    }

    bool publishAck_(uint64_t stream_id,
                     uint64_t sequence,
                     int64_t command_revision,
                     int64_t plan_id)
    {
        if (!ctx_) return false;
        const uint64_t received_unix_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        std::vector<uint8_t> payload = GmtTrajectoryAckV1::encode(
            stream_id, sequence, command_revision, plan_id, received_unix_ns);
        if (trajectory_.bufferedClip()) {
            // 扩展 ACK 回报本地实际帧号；旧 ACK 不能冒充缓存播放进度。
            std::memcpy(payload.data(), "OMGBFA01", 8);
            payload[10] = 60;
            payload[11] = 0;
            for (uint32_t value : {static_cast<uint32_t>(trajectory_.frameIndex()),
                                   static_cast<uint32_t>(trajectory_.frameCount())}) {
                for (int i = 0; i < 4; ++i) payload.push_back((value >> (8 * i)) & 0xffU);
            }
        }
        redisReply* reply = static_cast<redisReply*>(redisCommand(
            ctx_, "SET %s %b PX %d", ack_key_.c_str(), payload.data(),
            payload.size(), ack_ttl_ms_));
        if (!reply) {
            disconnect_();
            return false;
        }
        const bool ok = reply->type != REDIS_REPLY_ERROR;
        freeReplyObject(reply);
        return ok;
    }

    // ── Quaternion helpers ───────────────────────────────────────────────
    static float extractYaw_(const Eigen::Vector4f& q)
    {
        float w=q[0], x=q[1], y=q[2], z=q[3];
        return std::atan2(2.0f*(w*z + x*y), 1.0f - 2.0f*(y*y + z*z));
    }

    static Eigen::Vector4f yawQuat_(float angle)
    {
        float h = angle * 0.5f;
        return Eigen::Vector4f(std::cos(h), 0, 0, std::sin(h));
    }

    static Eigen::Vector4f quatMul_(const Eigen::Vector4f& q1, const Eigen::Vector4f& q2)
    {
        float w1=q1[0],x1=q1[1],y1=q1[2],z1=q1[3];
        float w2=q2[0],x2=q2[1],y2=q2[2],z2=q2[3];
        return Eigen::Vector4f(
            w1*w2-x1*x2-y1*y2-z1*z2,
            w1*x2+x1*w2+y1*z2-z1*y2,
            w1*y2-x1*z2+y1*w2+z1*x2,
            w1*z2+x1*y2-y1*x2+z1*w2);
    }

    static Eigen::Vector3f quatApply_(const Eigen::Vector4f& q, const Eigen::Vector3f& v)
    {
        float w=q[0];
        Eigen::Vector3f qv(q[1],q[2],q[3]);
        return v*(2*w*w-1) + qv.cross(v)*(2*w) + qv*(2*qv.dot(v));
    }

    static Eigen::Vector3f quatApplyInverse_(const Eigen::Vector4f& q,
                                              const Eigen::Vector3f& v)
    {
        float w = q[0];
        Eigen::Vector3f qv(q[1], q[2], q[3]);
        return v*(2*w*w-1) - qv.cross(v)*(2*w) + qv*(2*qv.dot(v));
    }

    // ── Frame parsing with alignment ─────────────────────────────────────
    bool parseFrame_(const std::vector<uint8_t>& blob)
    {
        if (GmtTrajectoryV1::hasMagic(blob)) {
            uint64_t ack_stream = 0;
            uint64_t ack_sequence = 0;
            int64_t ack_revision = 0;
            int64_t ack_plan = -1;
            {
                std::lock_guard<std::mutex> lk(mtx_);
                // 只解析小心跳包，不在每个策略周期复制整首缓存。
                GmtTrajectoryV1 candidate(expected_joint_names_);
                std::string error;
                if (!candidate.parse(blob, &error)) {
                    if (error != last_protocol_error_) {
                        std::printf("[MotionLoaderRedis] rejected trajectory_v1: %s\n",
                                    error.c_str());
                        last_protocol_error_ = error;
                    }
                    return false;
                }
                const bool new_stream = !has_trajectory_sequence_ ||
                    candidate.streamId() != trajectory_stream_id_;
                const bool new_sequence = new_stream ||
                    candidate.sequence() > trajectory_sequence_;
                if (candidate.bufferedClip()) return false; // 完整包只能从独立缓存键装载。
                if (candidate.bufferedControl()) {
                    if (!allow_buffered_) {
                        if (last_protocol_error_ != "buffered mode disabled") {
                            std::printf("[MotionLoaderRedis] buffered packet requires gmt_mode:=buffered\n");
                            last_protocol_error_ = "buffered mode disabled";
                        }
                        has_data_ = false;
                        return false;
                    }
                    if (new_stream) {
                        std::vector<uint8_t> clipBlob;
                        const std::string clipKey = key_ + ":clip:" + std::to_string(candidate.streamId());
                        GmtTrajectoryV1 clip(expected_joint_names_);
                        if (!getBlobAt_(clipKey, clipBlob) || !clip.parse(clipBlob, &error) ||
                            !clip.bufferedClip() || clip.streamId() != candidate.streamId() ||
                            clip.commandRevision() != candidate.commandRevision() ||
                            clip.planId() != candidate.planId()) {
                            std::printf("[MotionLoaderRedis] rejected buffered clip: %s\n", error.c_str());
                            has_data_ = false;
                            return false;
                        }
                        trajectory_ = std::move(clip);
                        std::printf("[MotionLoaderRedis] buffered clip cached: %u frames @ 50 simulation Hz\n",
                                    trajectory_.frameCount());
                    } else {
                        if (!trajectory_.bufferedClip() || candidate.sequence() < trajectory_sequence_ ||
                            candidate.commandRevision() != trajectory_.commandRevision() ||
                            candidate.planId() != trajectory_.planId()) return false;
                        trajectory_.advanceBufferedFrame();
                    }
                    trajectory_.updateEnvelope(candidate);
                } else {
                    if (allow_buffered_ && (candidate.flags() & 1U) == 0) {
                        has_data_ = false;
                        return false; // 缓存模式只允许普通固定 idle，禁止混入墙钟动作流。
                    }
                    if (!new_sequence) return true;
                    trajectory_ = std::move(candidate);
                }
                trajectory_stream_id_ = trajectory_.streamId();
                trajectory_sequence_ = trajectory_.sequence();
                has_trajectory_sequence_ = true;
                protocol_kind_ = MotionRedisProtocolKind::TRAJECTORY_V1;
                pos_w_ = trajectory_.rootPosW();
                quat_wxyz_ = trajectory_.rootQuatWxyz();
                lin_vel_b_ = trajectory_.rootLinVelB();
                ang_vel_b_ = trajectory_.rootAngVelB();
                joint_pos_ = trajectory_.targetJointPos();
                joint_vel_ = trajectory_.targetJointVel();
                command_history_.clear();
                has_prev_frame_ = false;
                has_data_ = true;
                if (new_sequence) last_recv_tp_ = std::chrono::steady_clock::now();
                last_protocol_error_.clear();
                ack_stream = trajectory_.streamId();
                ack_sequence = trajectory_.sequence();
                ack_revision = trajectory_.commandRevision();
                ack_plan = trajectory_.planId();
            }
            if (!publishAck_(ack_stream, ack_sequence, ack_revision, ack_plan)) {
                std::printf(
                    "[MotionLoaderRedis] failed to publish trajectory ACK "
                    "stream=%llu sequence=%llu key=%s\n",
                    static_cast<unsigned long long>(ack_stream),
                    static_cast<unsigned long long>(ack_sequence),
                    ack_key_.c_str());
            }
            return true;
        }

        const int expected = (1 + 3 + 4 + 3 + 3 + dof_) * sizeof(float);
        if ((int)blob.size() != expected) return false;

        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (protocol_kind_ != MotionRedisProtocolKind::LEGACY_FRAME) {
                command_history_.clear();
                has_prev_frame_ = false;
                prev_timestamp_ = 0.0f;
                prev_joint_pos_.setZero();
                aligned_ = false;
            }
        }

        const float* p = reinterpret_cast<const float*>(blob.data());
        int i = 0;
        float t = p[i++];  // Pico frame timestamp; used for dedup and joint velocity

        Eigen::Vector3f raw_pos(p[i], p[i+1], p[i+2]); i += 3;
        Eigen::Vector4f raw_quat(p[i], p[i+1], p[i+2], p[i+3]); i += 4;  // wxyz
        Eigen::Vector3f lv_w(p[i], p[i+1], p[i+2]); i += 3;
        Eigen::Vector3f av_w(p[i], p[i+1], p[i+2]); i += 3;

        Eigen::VectorXf jp(dof_);
        for (int j = 0; j < dof_; ++j) jp[j] = p[i++];

        const bool is_new_frame = !has_prev_frame_ || std::fabs(t - prev_timestamp_) > 1e-5f;
        if (!is_new_frame) return false;

        Eigen::VectorXf jv(dof_);
        jv.setZero();
        if (has_prev_frame_) {
            const float dt = t - prev_timestamp_;
            if (std::fabs(dt) > 1e-5f) {
                jv = (jp - prev_joint_pos_) / dt;
            }
        }
        prev_joint_pos_ = jp;
        prev_timestamp_ = t;
        has_prev_frame_ = true;

        // ── First-frame alignment ────────────────────────────────────────
        if (!aligned_) {
            // Compute yaw offset: rotate motion yaw to 0
            float raw_yaw = extractYaw_(raw_quat);
            yaw_offset_ = -raw_yaw;
            yaw_fix_quat_ = yawQuat_(yaw_offset_);

            // Compute xy offset: align motion start to robot origin
            Eigen::Vector3f rotated_pos = quatApply_(yaw_fix_quat_,
                Eigen::Vector3f(raw_pos[0], raw_pos[1], 0.0f));
            pos_offset_[0] = robot_origin_[0] - rotated_pos[0];
            pos_offset_[1] = robot_origin_[1] - rotated_pos[1];
            pos_offset_[2] = 0.0f;  // z 不偏移

            aligned_ = true;
            printf("[MotionLoaderRedis] aligned: yaw_offset=%.1f deg, "
                   "pos_offset=[%.3f, %.3f], raw_z=%.3f\n",
                   yaw_offset_ * 180.0f / M_PI,
                   pos_offset_[0], pos_offset_[1], raw_pos[2]);
        }

        // ── Apply alignment ──────────────────────────────────────────────
        // Rotate position and quaternion by yaw fix
        Eigen::Vector3f aligned_pos = quatApply_(yaw_fix_quat_, raw_pos) + pos_offset_;
        Eigen::Vector4f aligned_quat = quatMul_(yaw_fix_quat_, raw_quat);
        aligned_quat /= aligned_quat.norm();

        // Rotate world-frame velocities by yaw fix
        Eigen::Vector3f aligned_lv_w = quatApply_(yaw_fix_quat_, lv_w);
        Eigen::Vector3f aligned_av_w = quatApply_(yaw_fix_quat_, av_w);

        // 世界系速度 -> 体坐标系
        Eigen::Vector3f lv_b = quatApplyInverse_(aligned_quat, aligned_lv_w);
        Eigen::Vector3f av_b = quatApplyInverse_(aligned_quat, aligned_av_w);

        // Sonic command feature expected by the current BUMI policy:
        // [root_z, gravity_b(3), root_lin_vel_b(3), root_ang_vel_b(3),
        //  joint_pos(D), joint_vel(D)].
        const int command_dim = 10 + 2 * dof_;
        Eigen::VectorXf command_feature(command_dim);
        int out = 0;
        const Eigen::Vector3f gravity_b =
            quatApplyInverse_(aligned_quat, Eigen::Vector3f(0.0f, 0.0f, -1.0f));
        command_feature(out++) = aligned_pos.z();
        command_feature.segment<3>(out) = gravity_b; out += 3;
        command_feature.segment<3>(out) = lv_b; out += 3;
        command_feature.segment<3>(out) = av_b; out += 3;
        command_feature.segment(out, dof_) = jp; out += dof_;
        command_feature.segment(out, dof_) = jv;

        std::lock_guard<std::mutex> lk(mtx_);
        pos_w_      = aligned_pos;
        quat_wxyz_  = aligned_quat;
        lin_vel_b_  = lv_b;
        ang_vel_b_  = av_b;
        joint_pos_  = jp;
        joint_vel_  = jv;
        command_history_.push_back(command_feature);
        while (command_history_.size() > kMaxCommandHistory_) {
            command_history_.pop_front();
        }
        has_data_   = true;
        protocol_kind_ = MotionRedisProtocolKind::LEGACY_FRAME;
        source_timestamp_ = t;
        ++frame_sequence_;
        last_recv_tp_ = std::chrono::steady_clock::now();
        return true;
    }
};
#else
class MotionLoaderRedis
{
public:
    MotionLoaderRedis(const std::string& host,
                      int port,
                      int db,
                      const std::string& key,
                      int dof = 24,
                      float timeout_s = 0.2f,
                      const std::vector<std::string>& /*expected_joint_names*/ = {},
                      const std::string& /*ack_key*/ = "",
                      int /*ack_ttl_ms*/ = 1000)
    : joint_pos_(dof), joint_vel_(dof)
    {
        joint_pos_.setZero();
        joint_vel_.setZero();
        std::printf("[MotionLoaderRedis] hiredis unavailable; online mode disabled (requested %s:%d db=%d key=%s timeout=%.3f)\n",
                    host.c_str(), port, db, key.c_str(), timeout_s);
    }

    void update(float /*time_sec*/) {}
    void reset(const Eigen::Vector3f& robot_root_pos_w, float /*t*/ = 0.0f) { pos_w_ = robot_root_pos_w; }
    Eigen::Vector3f rootPosW() const { return pos_w_; }
    Eigen::Vector4f rootQuatWxyz() const { return quat_wxyz_; }
    Eigen::Vector3f rootLinVelB() const { return Eigen::Vector3f::Zero(); }
    Eigen::Vector3f rootAngVelB() const { return Eigen::Vector3f::Zero(); }
    Eigen::VectorXf targetJointPos() const { return joint_pos_; }
    Eigen::VectorXf targetJointVel() const { return joint_vel_; }
    MotionRedisProtocolKind protocolKind() const { return MotionRedisProtocolKind::NONE; }
    uint64_t streamId() const { return 0; }
    uint64_t sequence() const { return 0; }
    Eigen::VectorXf commandWindow(int half_window, int command_dim,
                                  bool /*centered_delay*/ = false) const
    {
        const int window_size = std::max(2 * half_window + 1, 0);
        return Eigen::VectorXf::Zero(window_size * std::max(command_dim, 0));
    }
    bool hasFreshData() const { return false; }
    bool hasCenteredWindow(int /*half_window*/) const { return false; }
    float dataAgeSec() const { return -1.0f; }
    double sourceTimestamp() const { return -1.0; }
    uint64_t frameSequence() const { return 0; }
    size_t commandHistorySize() const { return 0; }
private:
    Eigen::Vector3f pos_w_{0, 0, 0};
    Eigen::Vector4f quat_wxyz_{1, 0, 0, 0};
    Eigen::VectorXf joint_pos_;
    Eigen::VectorXf joint_vel_;
};
#endif

} // namespace legged
