#ifndef MEDIAPIPE_MOVE_POSE_MATH_H
#define MEDIAPIPE_MOVE_POSE_MATH_H

#include "mediapipe-load.h"
#include <cmath>

// MediaPipe facial transformation matrices are column-major 4x4 transforms.
// NVIDIA Move exposes head pose as quaternion X/Y/Z/W, not translation.
static inline bool mp_rotation_from_matrix(const float *m, mp_quaternion *q) {
  if (!m || !q)
    return false;

  const float r00 = m[0], r01 = m[4], r02 = m[8];
  const float r10 = m[1], r11 = m[5], r12 = m[9];
  const float r20 = m[2], r21 = m[6], r22 = m[10];
  const float determinant = r00 * (r11 * r22 - r12 * r21) - r01 * (r10 * r22 - r12 * r20) + r02 * (r10 * r21 - r11 * r20);
  if (!std::isfinite(determinant) || std::fabs(determinant - 1.0f) > 0.25f)
    return false;

  mp_quaternion result{};
  const float trace = r00 + r11 + r22;
  if (trace > 0) {
    const float scale = 2.0f * std::sqrt(trace + 1.0f);
    result.w = scale * 0.25f;
    result.x = (r21 - r12) / scale;
    result.y = (r02 - r20) / scale;
    result.z = (r10 - r01) / scale;
  } else if (r00 > r11 && r00 > r22) {
    const float scale = 2.0f * std::sqrt(1.0f + r00 - r11 - r22);
    result.w = (r21 - r12) / scale;
    result.x = scale * 0.25f;
    result.y = (r01 + r10) / scale;
    result.z = (r02 + r20) / scale;
  } else if (r11 > r22) {
    const float scale = 2.0f * std::sqrt(1.0f + r11 - r00 - r22);
    result.w = (r02 - r20) / scale;
    result.x = (r01 + r10) / scale;
    result.y = scale * 0.25f;
    result.z = (r12 + r21) / scale;
  } else {
    const float scale = 2.0f * std::sqrt(1.0f + r22 - r00 - r11);
    result.w = (r10 - r01) / scale;
    result.x = (r02 + r20) / scale;
    result.y = (r12 + r21) / scale;
    result.z = scale * 0.25f;
  }
  const float length = std::sqrt(result.x * result.x + result.y * result.y + result.z * result.z + result.w * result.w);
  if (!std::isfinite(length) || length < 0.00001f)
    return false;
  q->x = result.x / length;
  q->y = result.y / length;
  q->z = result.z / length;
  q->w = result.w / length;
  return true;
}

#endif
