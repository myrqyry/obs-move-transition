#include "../mediapipe-pose-math.h"
#include <cassert>
#include <cmath>
#include <cstdio>
int main() {
  const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  const float z90[16] = {0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  const float invalid[16] = {};
  const float face[16] = {0.941f, -0.059f, -0.332f, 0, 0.132f, 0.970f,  0.202f,   0,
                          0.310f, -0.234f, 0.921f,  0, 2.683f, -3.204f, -42.067f, 1};
  mp_quaternion q{};
  assert(mp_rotation_from_matrix(identity, &q));
  assert(std::fabs(q.w - 1.0f) < 0.0001f && std::fabs(q.z) < 0.0001f);
  assert(mp_rotation_from_matrix(z90, &q));
  assert(std::fabs(q.z - 0.7071068f) < 0.0001f && std::fabs(q.w - 0.7071068f) < 0.0001f);
  assert(!mp_rotation_from_matrix(invalid, &q));
  assert(mp_rotation_from_matrix(face, &q));
  const float mag = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  assert(std::fabs(mag - 1.0f) < 0.0001f);
  assert(std::fabs(q.x) > 0.02f && std::fabs(q.x) < 1.0f);
  printf("POSE_MATH_PASS x=%.4f y=%.4f z=%.4f w=%.4f\n", q.x, q.y, q.z, q.w);
  return 0;
}
