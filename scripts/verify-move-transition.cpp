// Isolated libobs regression probe for Move Transition rendering and HDR negotiation.
// No installed OBS plugin, scene collection, or configuration is modified.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <graphics/graphics.h>
#include <obs-module.h>
#include <obs.h>
#include <thread>
#include <vector>

static bool use_hdr = false;
struct TestInput {
  gs_texture_t *texture = nullptr;
};
static const char *input_name(void *) { return "Move transition verification input"; }
static uint32_t input_dimension(void *) { return 256; }
static enum gs_color_space input_color_space(void *, size_t, const enum gs_color_space *) {
  return use_hdr ? GS_CS_709_SCRGB : GS_CS_SRGB;
}
static void *input_create(obs_data_t *, obs_source_t *) {
  auto *input = new TestInput;
  obs_enter_graphics();
  if (use_hdr) {
    // RGBA16F: 2.0 red, 0.5 green, 0 blue, 1.0 alpha.
    // A red channel above 1.0 detects inadvertent HDR -> SDR clamping.
    std::vector<uint16_t> pixels(256 * 256 * 4);
    for (size_t i = 0; i < pixels.size(); i += 4) {
      pixels[i] = 0x4000;
      pixels[i + 1] = 0x3800;
      pixels[i + 2] = 0;
      pixels[i + 3] = 0x3c00;
    }
    input->texture = gs_texture_create(256, 256, GS_RGBA16F, 1, nullptr, GS_DYNAMIC);
    if (input->texture)
      gs_texture_set_image(input->texture, reinterpret_cast<uint8_t *>(pixels.data()), 256 * 8, false);
  } else {
    std::vector<uint8_t> pixels(256 * 256 * 4, 255);
    for (size_t i = 0; i < pixels.size(); i += 4) {
      pixels[i] = 224;
      pixels[i + 1] = 64;
      pixels[i + 2] = 32;
    }
    input->texture = gs_texture_create(256, 256, GS_RGBA, 1, nullptr, GS_DYNAMIC);
    if (input->texture)
      gs_texture_set_image(input->texture, pixels.data(), 256 * 4, false);
  }
  obs_leave_graphics();
  return input;
}
static void input_destroy(void *data) {
  auto *input = static_cast<TestInput *>(data);
  obs_enter_graphics();
  if (input->texture)
    gs_texture_destroy(input->texture);
  obs_leave_graphics();
  delete input;
}
static void input_render(void *data, gs_effect_t *) {
  auto *input = static_cast<TestInput *>(data);
  if (input->texture)
    obs_source_draw(input->texture, 0, 0, 256, 256, false);
}
static obs_source_info input_info = {
    .id = "move_verify_input",
    .type = OBS_SOURCE_TYPE_INPUT,
    .output_flags = OBS_SOURCE_VIDEO,
    .get_name = input_name,
    .create = input_create,
    .destroy = input_destroy,
    .get_width = input_dimension,
    .get_height = input_dimension,
    .video_render = input_render,
    .video_get_color_space = input_color_space,
};
int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "Usage: %s <move-transition.so> <plugin-data-dir> <sdr|hdr>\n", argv[0]);
    return 2;
  }
  setbuf(stdout, nullptr);
  use_hdr = std::strcmp(argv[3], "hdr") == 0;
  if (!use_hdr && std::strcmp(argv[3], "sdr"))
    return 2;
  if (!obs_startup("en-US", nullptr, nullptr))
    return 3;
  obs_video_info info{};
  info.graphics_module = "libobs-opengl";
  info.fps_num = 30;
  info.fps_den = 1;
  info.base_width = 256;
  info.base_height = 256;
  info.output_width = 256;
  info.output_height = 256;
  info.output_format = VIDEO_FORMAT_RGBA;
  info.scale_type = OBS_SCALE_DISABLE;
  if (obs_reset_video(&info) != OBS_VIDEO_SUCCESS)
    return 4;
  obs_register_source(&input_info);
  obs_module_t *module = nullptr;
  auto status = obs_open_module(&module, argv[1], argv[2]);
  if (status != MODULE_SUCCESS || !obs_init_module(module)) {
    std::fprintf(stderr, "Module loading failed, status=%d\n", (int)status);
    return 5;
  }
  auto *source = obs_source_create("move_verify_input", "moving-source", nullptr, nullptr);
  auto *from = obs_scene_create("move-verify-from");
  auto *to = obs_scene_create("move-verify-to");
  auto *item_a = obs_scene_add(from, source);
  auto *item_b = obs_scene_add(to, source);
  vec2 pos_a{}, pos_b{};
  vec2_set(&pos_a, 12.0f, 18.0f);
  vec2_set(&pos_b, 116.0f, 48.0f);
  obs_sceneitem_set_pos(item_a, &pos_a);
  obs_sceneitem_set_pos(item_b, &pos_b);
  auto *transition = obs_source_create_private("move_transition", "move-verify-transition", nullptr);
  if (!transition)
    return 6;
  obs_transition_set(transition, obs_scene_get_source(from));
  if (!obs_transition_start(transition, OBS_TRANSITION_MODE_MANUAL, 1000, obs_scene_get_source(to))) {
    std::fprintf(stderr, "Move transition failed to start\n");
    return 7;
  }
  const gs_color_space preferred[] = {GS_CS_SRGB, GS_CS_709_SCRGB};
  // OBS scenes use their own color-space negotiation. Verify scene rendering
  // separately from HDR propagation through a transition between HDR inputs.
  auto *second = obs_source_create("move_verify_input", "moving-source-second", nullptr, nullptr);
  auto *direct = obs_source_create_private("move_transition", "move-verify-direct", nullptr);
  if (!second || !direct)
    return 9;
  obs_transition_set(direct, source);
  if (!obs_transition_start(direct, OBS_TRANSITION_MODE_MANUAL, 1000, second))
    return 10;
  obs_set_output_source(0, transition);
  obs_set_output_source(1, direct);
  enum gs_color_space scene_actual = obs_source_get_color_space(transition, 2, preferred);
  enum gs_color_space actual = obs_source_get_color_space(direct, 2, preferred);
  enum gs_color_space expected = use_hdr ? GS_CS_709_SCRGB : GS_CS_SRGB;
  std::printf("COLOR_SPACE mode=%s input=%d scene=%d direct=%d expected=%d\n", argv[3],
              (int)obs_source_get_color_space(source, 2, preferred), (int)scene_actual, (int)actual, (int)expected);
  bool rendered = true;
  int first_alpha = -1, last_alpha = -1;
  int hdr_half_red = -1;
  for (float t : {0.10f, 0.40f, 0.75f}) {
    obs_transition_set_manual_time(transition, t);
    obs_transition_set_manual_time(direct, t);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::printf("CLOCK requested=%.2f actual_scene=%.3f actual_direct=%.3f\n", t, obs_transition_get_time(transition),
                obs_transition_get_time(direct));
    obs_enter_graphics();
    // The scene's output is SDR; sample a pixel to prove the source actually
    // changes position, not just that the render target initializes.
    auto *target = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
    bool ok = target && gs_texrender_begin_with_color_space(target, 256, 256, GS_CS_SRGB);
    int alpha = -1;
    if (ok) {
      struct vec4 clear_color {};
      gs_clear(GS_CLEAR_COLOR, &clear_color, 1.0f, 0);
      gs_projection_push();
      gs_ortho(0, 256, 0, 256, -100, 100);
      obs_source_video_render(transition);
      gs_projection_pop();
      gs_texrender_end(target);
      auto *staging = gs_stagesurface_create(256, 256, GS_RGBA);
      if (staging) {
        gs_stage_texture(staging, gs_texrender_get_texture(target));
        uint8_t *pixels = nullptr;
        uint32_t stride = 0;
        if (gs_stagesurface_map(staging, &pixels, &stride)) {
          alpha = pixels[70 * stride + 40 * 4 + 3];
          std::printf("PIXELS x0=%u x40RGBA=%u,%u,%u,%u x200=%u\n", (unsigned)pixels[70 * stride + 3],
                      (unsigned)pixels[70 * stride + 40 * 4], (unsigned)pixels[70 * stride + 40 * 4 + 1],
                      (unsigned)pixels[70 * stride + 40 * 4 + 2], (unsigned)alpha, (unsigned)pixels[70 * stride + 200 * 4 + 3]);
          gs_stagesurface_unmap(staging);
        }
        gs_stagesurface_destroy(staging);
      }
    }
    if (target)
      gs_texrender_destroy(target);
    // Render the HDR-aware source transition through an HDR target.
    auto *hdr_target = gs_texrender_create(use_hdr ? GS_RGBA16F : GS_RGBA, GS_ZS_NONE);
    bool hdr_ok = hdr_target && gs_texrender_begin_with_color_space(hdr_target, 256, 256, use_hdr ? GS_CS_709_SCRGB : GS_CS_SRGB);
    if (hdr_ok) {
      struct vec4 clear_color {};
      gs_clear(GS_CLEAR_COLOR, &clear_color, 1.0f, 0);
      gs_projection_push();
      gs_ortho(0, 256, 0, 256, -100, 100);
      obs_source_video_render(direct);
      gs_projection_pop();
      gs_texrender_end(hdr_target);
      if (use_hdr) {
        auto *stage = gs_stagesurface_create(256, 256, GS_RGBA16F);
        if (stage) {
          gs_stage_texture(stage, gs_texrender_get_texture(hdr_target));
          uint8_t *data = nullptr;
          uint32_t stride = 0;
          if (gs_stagesurface_map(stage, &data, &stride)) {
            uint16_t half = 0;
            std::memcpy(&half, data + 128 * stride + 128 * 8, sizeof(half));
            hdr_half_red = (int)half;
            gs_stagesurface_unmap(stage);
          }
          gs_stagesurface_destroy(stage);
        }
      }
    }
    if (hdr_target)
      gs_texrender_destroy(hdr_target);
    obs_leave_graphics();
    rendered &= ok && hdr_ok && alpha >= 0;
    if (t < 0.2f)
      first_alpha = alpha;
    if (t > 0.7f)
      last_alpha = alpha;
    std::printf("RENDER time=%.2f scene_target=%d hdr_target=%d sample_alpha=%d\n", t, (int)ok, (int)hdr_ok, alpha);
  }
  bool scene_moved = first_alpha > 180 && last_alpha < 30;
  bool hdr_preserved = !use_hdr || (hdr_half_red > 0x3c00 && hdr_half_red < 0x7c00);
  if (use_hdr)
    std::printf("HDR_PIXEL half_red=%d above_one=%d\n", hdr_half_red, (int)hdr_preserved);
  std::printf("SCENE_MOVED first_alpha=%d last_alpha=%d pass=%d\n", first_alpha, last_alpha, (int)scene_moved);
  obs_set_output_source(0, nullptr);
  obs_set_output_source(1, nullptr);
  // Give OBS's video thread a chance to drop its active-source references.
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  obs_transition_force_stop(transition);
  obs_transition_force_stop(direct);
  obs_transition_clear(transition);
  obs_transition_clear(direct);
  obs_source_release(transition);
  obs_source_release(direct);
  obs_sceneitem_remove(item_a);
  obs_sceneitem_remove(item_b);
  obs_scene_release(from);
  obs_scene_release(to);
  obs_source_release(second);
  obs_source_release(source);
  obs_shutdown();
  if (!rendered || !scene_moved || !hdr_preserved || actual != expected) {
    std::fprintf(stderr, "MOVE_VERIFY_FAIL rendered=%d scene_moved=%d hdr_preserved=%d color_space_match=%d\n", (int)rendered,
                 (int)scene_moved, (int)hdr_preserved, (int)(actual == expected));
    return 8;
  }
  std::puts("MOVE_VERIFY_PASS");
  return 0;
}
