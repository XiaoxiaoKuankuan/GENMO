"""MotionMillion 评测的长度协议身份。

fixed 保留历史固定120帧生成；gt 按官方资格记录的原始真实长度逐条生成，命名为
full_sequence_matched_length_v1。两者均不偷偷执行四帧随机裁剪，因此不声明论文
同协议复现。资格仍限原有60—200帧，201—300帧只能另建长动作诊断，不能混入这里。
"""


def length_protocol(mode="fixed", num_frames=120):
    if mode not in {"fixed", "gt"}:
        raise ValueError("length_mode 必须为 fixed/gt")
    if mode == "fixed" and not 60 <= num_frames <= 300:
        raise ValueError("fixed num_frames 必须在 [60,300]")
    return {
        "name": "fixed_length_v1" if mode == "fixed" else "full_sequence_matched_length_v1",
        "length_mode": mode,
        "fixed_num_frames": num_frames if mode == "fixed" else None,
        "fps": 30,
        "gt_crop": "none",
        "official_four_frame_random_crop": False,
    }


def prediction_length(source, protocol):
    frames = int(source["frames"])
    if not 60 <= frames <= 200:
        raise ValueError("官方资格集合仅支持60—200帧；长动作须使用单独诊断集合")
    return frames if protocol["length_mode"] == "gt" else protocol["fixed_num_frames"]
