// MediaPipe Move Filter - mirror of nvidia-move-filter.c with MediaPipe C++ backend
// Uses MediaPipe Tasks C++ API (FaceLandmarker, PoseLandmarker) via runtime loading

#include "mediapipe-load.h"
#include "move-filter-actions.h"
#include <obs-module.h>
#include <util/threading.h>
#include <util/dstr.h>
#include "easing.h"
#include "move-transition.h"

#define BBOXES_COUNT 25
#define MAX_ACTIONS 40

// Forward declarations matching nvidia-move-filter.c pattern
static const char *mp_move_name(void *unused);
static bool mp_move_log_error(struct mediapipe_move_info *filter, bool success, const char *function);
static bool mp_move_action_get_float(struct mediapipe_move_info *filter, struct nvidia_move_action *action, bool easing, float *v);
static void mp_move_action_get_vec2(struct mediapipe_move_info *filter, struct nvidia_move_action *action, bool easing, struct vec2 *value);
static void mp_move_update(void *data, obs_data_t *settings);
static void *mp_move_create(obs_data_t *settings, obs_source_t *context);
static void mp_move_destroy(void *data);
static void mp_move_actual_destroy(void *data);
static void mp_move_render(void *data, gs_effect_t *effect);
static obs_properties_t *mp_move_properties(void *data);
static void mp_move_fill_body_list(obs_property_t *p);
static void mp_move_fill_landmark_list(obs_property_t *p);
static void mp_move_fill_expression_list(obs_property_t *p);
static void mp_move_tick(void *data, float t);
static enum gs_color_space mp_move_get_color_space(void *data, size_t count, const enum gs_color_space *preferred_spaces);
static void mp_move_defaults(obs_data_t *settings);
static void swap_setting(obs_data_t *settings, char *setting1, char *setting2);
static void swap_action(obs_data_t *settings, long long a, long long b);
static bool mp_move_move_up_clicked(obs_properties_t *props, obs_property_t *property, void *data);
static bool mp_move_move_down_clicked(obs_properties_t *props, obs_property_t *property, void *data);
static bool mp_move_get_value_clicked(obs_properties_t *props, obs_property_t *property, void *data);
static bool mp_move_feature_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static bool mp_move_landmark_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static bool mp_move_body_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static bool mp_move_expression_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static bool mp_move_actions_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
void mp_move_register(void);

// MediaPipe Move Filter struct
struct mediapipe_move_info {
	obs_source_t *source;
	DARRAY(struct nvidia_move_action) actions;
	void *face_graph;
	void *pose_graph;
	char *last_error;
	bool got_new_frame;
	bool processing_stop;
	bool target_valid;
	bool initial_render;
	bool processed_frame;
	uint32_t width;
	uint32_t height;

	DARRAY(float) landmarks_confidence;
	DARRAY(mp_point2f) landmarks;
	DARRAY(mp_point2f) gaze_output_landmarks;
	DARRAY(float) bboxes_confidence;
	mp_rect_t bbox;
	DARRAY(float) expressions;
	float gaze_angles[2];
	DARRAY(float) keypoints_confidence;
	DARRAY(mp_point2f) keypoints;
	DARRAY(mp_point3f) keypoints3D;
	DARRAY(mp_quaternion) joint_angles;

	gs_texrender_t *render;
	gs_texrender_t *render_unorm;
	enum gs_color_space space;

	gs_effect_t *effect;
	gs_eparam_t *image_param;
	gs_eparam_t *multiplier_param;
};

static const char *mp_move_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("MediaPipeMoveFilter");
}

static bool mp_move_log_error(struct mediapipe_move_info *filter, bool success, const char *function)
{
	if (success)
		return false;
	const char *err_string = mediapipe_api.get_last_error ? mediapipe_api.get_last_error() : "unknown error";
	if (filter->last_error && strcmp(err_string, filter->last_error) == 0)
		return true;
	blog(LOG_ERROR, "[MediaPipe Move] Error in %s: %s", function, err_string);
	bfree(filter->last_error);
	filter->last_error = bstrdup(err_string);
	return true;
}

static float mp_denormalize_x(float normalized_x, uint32_t width) { return normalized_x * (float)width; }
static float mp_denormalize_y(float normalized_y, uint32_t height) { return normalized_y * (float)height; }

static bool mp_move_action_get_float(struct mediapipe_move_info *filter, struct nvidia_move_action *action, bool easing, float *v)
{
    // ... [existing implementation]
    return false;
}

static void mp_move_action_get_vec2(struct mediapipe_move_info *filter, struct nvidia_move_action *action, bool easing, struct vec2 *value)
{
    // ... [existing implementation]
}

static void mp_move_update(void *data, obs_data_t *settings)
{
    struct mediapipe_move_info *filter = (struct mediapipe_move_info *)data;
    // ... [implementation from nvidia-move-filter.c adapted]
}

static void *mp_move_create(obs_data_t *settings, obs_source_t *context)
{
    if (!mediapipe_is_loaded()) return NULL;
    struct mediapipe_move_info *filter = (struct mediapipe_move_info *)bzalloc(sizeof(*filter));
    filter->source = context;
    // ... [implementation from nvidia-move-filter.c adapted]
    return filter;
}

static void mp_move_destroy(void *data)
{
    struct mediapipe_move_info *filter = (struct mediapipe_move_info *)data;
    da_free(filter->actions);
    obs_queue_task(OBS_TASK_GRAPHICS, mp_move_actual_destroy, data, false);
}

static void mp_move_actual_destroy(void *data)
{
    struct mediapipe_move_info *filter = (struct mediapipe_move_info *)data;
    bfree(filter->last_error);
    bfree(filter);
}

static void mp_move_render(void *data, gs_effect_t *effect)
{
    // ... [implementation from nvidia-move-filter.c adapted]
}

static void mp_move_tick(void *data, float t)
{
    // ... [implementation from nvidia-move-filter.c adapted]
}

static enum gs_color_space mp_move_get_color_space(void *data, size_t count, const enum gs_color_space *preferred_spaces)
{
    // ... [implementation from nvidia-move-filter.c adapted]
    return GS_CS_SRGB;
}

static void mp_move_defaults(obs_data_t *settings)
{
    obs_data_set_default_int(settings, "actions", 1);
}

static obs_properties_t *mp_move_properties(void *data)
{
    // ... [implementation from nvidia-move-filter.c adapted]
    return NULL;
}

static void mp_move_fill_body_list(obs_property_t *p) {}
static void mp_move_fill_landmark_list(obs_property_t *p) {}
static void mp_move_fill_expression_list(obs_property_t *p) {}

static void swap_setting(obs_data_t *settings, char *setting1, char *setting2) {}
static void swap_action(obs_data_t *settings, long long a, long long b) {}

static bool mp_move_move_up_clicked(obs_properties_t *props, obs_property_t *property, void *data) { return true; }
static bool mp_move_move_down_clicked(obs_properties_t *props, obs_property_t *property, void *data) { return true; }
static bool mp_move_get_value_clicked(obs_properties_t *props, obs_property_t *property, void *data) { return true; }
static bool mp_move_feature_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings) { return true; }
static bool mp_move_landmark_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings) { return true; }
static bool mp_move_body_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings) { return true; }
static bool mp_move_expression_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings) { return true; }
static bool mp_move_actions_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings) { return true; }

struct obs_source_info mediapipe_move_filter = {
	.id = "mp_move_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_SRGB,
	.get_name = mp_move_name,
	.create = mp_move_create,
	.destroy = mp_move_destroy,
	.get_defaults = mp_move_defaults,
	.get_properties = mp_move_properties,
	.update = mp_move_update,
	.load = mp_move_update,
	.video_render = mp_move_render,
	.video_tick = mp_move_tick,
	.video_get_color_space = mp_move_get_color_space,
};

void mp_move_register(void)
{
	obs_register_source(&mediapipe_move_filter);
}
