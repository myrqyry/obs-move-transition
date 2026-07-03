// MediaPipe Move Filter - mirror of nvidia-move-filter.c with MediaPipe C++ backend
// Uses MediaPipe Tasks C++ API (FaceLandmarker, PoseLandmarker) via runtime loading

#include "mediapipe-load.h"
#include "move-filter-actions.h"
#include <obs-module.h>
#include <util/threading.h>
#include <util/dstr.h>
#include <util/platform.h>
#include <util/c99defs.h>
#include "easing.h"
#include "move-transition.h"

#include <graphics/vec2.h>
#include <graphics/matrix4.h>

#define BBOXES_COUNT 25
#define MAX_ACTIONS 40

// Forward declarations
static const char *mp_move_name(void *unused);
static bool mp_move_log_error(struct mediapipe_move_info *filter, bool success, const char *function);
static bool mp_move_action_get_float(struct mediapipe_move_info *filter, struct nvidia_move_action *action, bool easing, float *v);
static bool mp_move_action_get_vec2(struct mediapipe_move_info *filter, struct nvidia_move_action *action, bool easing, struct vec2 *value);
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
extern "C" void mp_move_register(void);

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

	// Face detection thresholds
	float face_detection_confidence;
	float face_presence_confidence;
	float face_tracking_confidence;

	DARRAY(float) landmarks_confidence;
	DARRAY(mp_point3f) landmarks;
	DARRAY(float) gaze_output_landmarks;
	DARRAY(float) bboxes_confidence;
	mp_rect_t bbox;
	DARRAY(float) expressions;
	float face_transform_matrix[16];
	DARRAY(float) keypoints_confidence;
	DARRAY(mp_point2f) keypoints;
	DARRAY(mp_point3f) keypoints3D;
	DARRAY(mp_quaternion) joint_angles;

	gs_texrender_t *render;
	gs_texrender_t *render_unorm;
	gs_stagesurf_t *stage_surf;
	enum gs_color_space space;

	gs_effect_t *effect;
	gs_eparam_t *image_param;
	gs_eparam_t *multiplier_param;

	// CPU pixel buffer for MediaPipe detection (readback from GPU)
	uint8_t *cpu_buffer;
	uint32_t cpu_buffer_size;
};

static const char *mp_move_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("MediaPipeMoveFilter");
}

static bool mp_move_log_error(struct mediapipe_move_info *filter, bool success, const char *function)
{
	UNUSED_PARAMETER(filter);
	if (success)
		return false;
	blog(LOG_ERROR, "[MediaPipe Move] Error in %s", function);
	return true;
}

static bool mp_check_model(const char *name)
{
	char *path = obs_module_file(name);
	bool exists = os_file_exists(path);
	if (!exists) {
		blog(LOG_ERROR, "[MediaPipe] Model file missing: %s", name);
	}
	bfree(path);
	return exists;
}

static void *mp_move_create(obs_data_t *settings, obs_source_t *context)
{
	if (!mediapipe_is_loaded())
		return NULL;

	struct mediapipe_move_info *filter =
		(struct mediapipe_move_info *)bzalloc(sizeof(*filter));
	filter->source = context;
	filter->space = GS_CS_SRGB;

	// Default confidence thresholds
	filter->face_detection_confidence = 0.5f;
	filter->face_presence_confidence = 0.5f;
	filter->face_tracking_confidence = 0.5f;

	char *effect_path = obs_module_file("effects/unorm.effect");
	obs_enter_graphics();
	filter->effect = gs_effect_create_from_file(effect_path, NULL);
	bfree(effect_path);
	if (filter->effect) {
		filter->image_param =
			gs_effect_get_param_by_name(filter->effect, "image");
		filter->multiplier_param =
			gs_effect_get_param_by_name(filter->effect, "multiplier");
	}
	obs_leave_graphics();

	// Handles created lazily in mp_move_update based on feature_flags
	obs_source_update(context, settings);
	return filter;
}

static void mp_move_destroy(void *data)
{
	struct mediapipe_move_info *filter =
		(struct mediapipe_move_info *)data;
	for (size_t i = 0; i < filter->actions.num; i++) {
		struct nvidia_move_action *action = filter->actions.array + i;
		obs_weak_source_release(action->target);
		action->target = NULL;
		bfree(action->name);
		action->name = NULL;
	}
	da_free(filter->actions);
	obs_queue_task(OBS_TASK_GRAPHICS, mp_move_actual_destroy, data, false);
}

static void mp_move_actual_destroy(void *data)
{
	struct mediapipe_move_info *filter =
		(struct mediapipe_move_info *)data;

	mp_api_t *api = mediapipe_get_api();

	// Destroy face landmarker handle if it exists
	if (filter->face_graph) {
		char *error_msg = NULL;
		int ret =
			api->face_landmarker_close(filter->face_graph, &error_msg);
		if (ret != 0 && error_msg) {
			blog(LOG_ERROR,
			     "[MediaPipe Move] Error closing face landmarker: %s",
			     error_msg);
			api->error_free(error_msg);
		}
		filter->face_graph = NULL;
	}

	// Destroy pose landmarker handle if it exists
	if (filter->pose_graph) {
		char *error_msg = NULL;
		int ret =
			api->pose_landmarker_close(filter->pose_graph, &error_msg);
		if (ret != 0 && error_msg) {
			blog(LOG_ERROR,
			     "[MediaPipe Move] Error closing pose landmarker: %s",
			     error_msg);
			api->error_free(error_msg);
		}
		filter->pose_graph = NULL;
	}

	// Free all DARRAY members
	da_free(filter->landmarks_confidence);
	da_free(filter->landmarks);
	da_free(filter->gaze_output_landmarks);
	da_free(filter->bboxes_confidence);
	da_free(filter->expressions);
	da_free(filter->keypoints_confidence);
	da_free(filter->keypoints);
	da_free(filter->keypoints3D);
	da_free(filter->joint_angles);

	// Destroy graphics resources (must be on graphics thread)
	obs_enter_graphics();
	gs_texrender_destroy(filter->render);
	filter->render = NULL;
	gs_texrender_destroy(filter->render_unorm);
	filter->render_unorm = NULL;
	if (filter->stage_surf) {
		gs_stagesurface_destroy(filter->stage_surf);
		filter->stage_surf = NULL;
	}
	if (filter->effect) {
		gs_effect_destroy(filter->effect);
		filter->effect = NULL;
	}
	obs_leave_graphics();

	bfree(filter->cpu_buffer);
	bfree(filter->last_error);
	bfree(filter);
}

// --- Render and detection ---

static void mp_move_draw_frame(struct mediapipe_move_info *filter,
			       uint32_t base_width, uint32_t base_height)
{
	obs_source_skip_video_filter(filter->source);
	UNUSED_PARAMETER(base_width);
	UNUSED_PARAMETER(base_height);
}

// --- Action evaluation and execution helpers ---

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Map attachment identifiers to MediaPipe Face Mesh landmark indices
static bool mp_move_get_attachment_pos(struct mediapipe_move_info *filter,
				       uint32_t attach_point, struct vec2 *pos)
{
	if (filter->landmarks.num < 474) // Safety check: requires face landmarks up to irises
		return false;

	mp_point3f *lms = filter->landmarks.array;
	float w = (float)filter->width;
	float h = (float)filter->height;

	// Lambda helper to project normalized landmark to absolute texture space coordinates
	auto get_pt = [&](int idx) -> struct vec2 {
		struct vec2 p;
		p.x = lms[idx].x * w;
		p.y = lms[idx].y * h;
		return p;
	};

	switch (attach_point) {
	case ATTACH_EYES: {
		struct vec2 le = get_pt(468); // Left iris center
		struct vec2 re = get_pt(473); // Right iris center
		pos->x = (le.x + re.x) * 0.5f;
		pos->y = (le.y + re.y) * 0.5f;
		return true;
	}
	case ATTACH_LEFT_EYE:
		*pos = get_pt(468);
		return true;
	case ATTACH_RIGHT_EYE:
		*pos = get_pt(473);
		return true;
	case ATTACH_EYEBROWS: {
		struct vec2 leb = get_pt(70);
		struct vec2 reb = get_pt(300);
		pos->x = (leb.x + reb.x) * 0.5f;
		pos->y = (leb.y + reb.y) * 0.5f;
		return true;
	}
	case ATTACH_LEFT_EYEBROW:
		*pos = get_pt(70);
		return true;
	case ATTACH_RIGHT_EYEBROW:
		*pos = get_pt(300);
		return true;
	case ATTACH_EARS: {
		struct vec2 le = get_pt(127);
		struct vec2 re = get_pt(356);
		pos->x = (le.x + re.x) * 0.5f;
		pos->y = (le.y + re.y) * 0.5f;
		return true;
	}
	case ATTACH_LEFT_EAR:
		*pos = get_pt(127);
		return true;
	case ATTACH_RIGHT_EAR:
		*pos = get_pt(356);
		return true;
	case ATTACH_NOSE:
		*pos = get_pt(4); // Nose tip
		return true;
	case ATTACH_MOUTH:
		*pos = get_pt(13); // Inner lip center
		return true;
	case ATTACH_UPPER_LIP:
		*pos = get_pt(0);
		return true;
	case ATTACH_LOWER_LIP:
		*pos = get_pt(17);
		return true;
	case ATTACH_CHIN:
		*pos = get_pt(152);
		return true;
	case ATTACH_JAW: {
		struct vec2 lj = get_pt(172);
		struct vec2 rj = get_pt(397);
		pos->x = (lj.x + rj.x) * 0.5f;
		pos->y = (lj.y + rj.y) * 0.5f;
		return true;
	}
	case ATTACH_FOREHEAD:
		*pos = get_pt(10);
		return true;
	default:
		break;
	}
	return false;
}

// Compute facial roll angle (degrees) based on Left/Right Iris vector
static float mp_move_get_head_roll(struct mediapipe_move_info *filter)
{
	if (filter->landmarks.num < 474)
		return 0.0f;
	mp_point3f *lms = filter->landmarks.array;
	float dx = lms[473].x - lms[468].x;
	float dy = lms[473].y - lms[468].y;
	return atan2f(dy, dx) * (180.0f / M_PI);
}

// Evaluates thresholds for filter switches / scene item visibility
static void mp_move_apply_threshold_action(struct mediapipe_move_info *filter,
					   struct nvidia_move_action *action,
					   float value)
{
	bool state = false;
	bool eval = false;

	switch (action->property) {
	case FEATURE_THRESHOLD_ENABLE_OVER:
		if (value > action->threshold) {
			state = true;
			eval = true;
		}
		break;
	case FEATURE_THRESHOLD_ENABLE_UNDER:
		if (value < action->threshold) {
			state = true;
			eval = true;
		}
		break;
	case FEATURE_THRESHOLD_DISABLE_OVER:
		if (value > action->threshold) {
			state = false;
			eval = true;
		}
		break;
	case FEATURE_THRESHOLD_DISABLE_UNDER:
		if (value < action->threshold) {
			state = false;
			eval = true;
		}
		break;
	case FEATURE_THRESHOLD_ENABLE_OVER_DISABLE_UNDER:
		state = (value > action->threshold);
		eval = true;
		break;
	case FEATURE_THRESHOLD_ENABLE_UNDER_DISABLE_OVER:
		state = (value < action->threshold);
		eval = true;
		break;
	}

	if (!eval)
		return;

	if (action->action == ACTION_ENABLE_FILTER) {
		obs_source_t *target_source =
			obs_weak_source_get_source(action->target);
		if (target_source) {
			obs_source_filter_set_enabled(target_source, state);
			obs_source_release(target_source);
		}
	} else if (action->action == ACTION_SOURCE_VISIBILITY) {
		obs_source_t *scene_source =
			obs_weak_source_get_source(action->target);
		if (scene_source) {
			obs_scene_t *scene =
				obs_scene_from_source(scene_source);
			if (!scene)
				scene = obs_group_from_source(scene_source);
			if (scene) {
				obs_sceneitem_t *item =
					obs_scene_find_source(scene,
							      action->name);
				if (item) {
					obs_sceneitem_set_visible(item,
								  state);
				}
			}
			obs_source_release(scene_source);
		}
	}
}

// Core action execution: parses current feature states and dispatches OBS API calls
static void mp_move_execute_actions(struct mediapipe_move_info *filter)
{
	for (size_t i = 0; i < filter->actions.num; i++) {
		struct nvidia_move_action *action =
			filter->actions.array + i;
		if (action->disabled || !action->target)
			continue;

		// --- 1. HANDLE SOURCE ATTACHMENTS ---
		if (action->action == ACTION_ATTACH_SOURCE) {
			obs_source_t *scene_source =
				obs_weak_source_get_source(action->target);
			if (!scene_source)
				continue;

			obs_scene_t *scene =
				obs_scene_from_source(scene_source);
			if (!scene)
				scene = obs_group_from_source(scene_source);

			if (scene) {
				obs_sceneitem_t *item =
					obs_scene_find_source(
						scene, action->name);
				if (item) {
					struct vec2 pos;
					if (mp_move_get_attachment_pos(
						    filter,
						    action->property,
						    &pos)) {
						// Apply scale factors and offsets
						pos.x = pos.x *
								action->factor +
							action->diff;
						pos.y = pos.y *
								action->factor2 +
							action->diff2;

						// Apply action easing over time
						if (action->easing > 0.0f) {
							pos.x = action->previous_vec2.x *
									action->easing +
								pos.x *
									(1.0f - action->easing);
							pos.y = action->previous_vec2.y *
									action->easing +
								pos.y *
									(1.0f - action->easing);
						}
						action->previous_vec2 = pos;

						obs_sceneitem_set_pos(item,
								      &pos);

						// Match facial roll rotation
						float roll =
							mp_move_get_head_roll(
								filter);
						obs_sceneitem_set_rot(item,
								      roll);
					}
				}
			}
			obs_source_release(scene_source);
			continue;
		}

		// --- 2. EVALUATE CONFIDENCE THRESHOLDS FOR GENERAL FEATURES ---
		if (action->feature == FEATURE_BOUNDINGBOX &&
		    filter->bbox.confidence < action->required_confidence)
			continue;

		if (action->feature == FEATURE_LANDMARK) {
			int32_t lm_idx = action->feature_number[0];
			if (lm_idx >= 0 &&
			    (size_t)lm_idx <
				    filter->landmarks_confidence.num) {
				if (filter->landmarks_confidence
					    .array[lm_idx] <
				    action->required_confidence)
					continue;
			}
		}

		if (action->feature == FEATURE_BODY) {
			int32_t kp_idx = action->feature_number[0];
			if (kp_idx >= 0 &&
			    (size_t)kp_idx <
				    filter->keypoints_confidence.num) {
				if (filter->keypoints_confidence
					    .array[kp_idx] <
				    action->required_confidence)
					continue;
			}
		}

		// --- 3. EXTRACT VALUES ---
		float value_f = 0.0f;
		struct vec2 value_v = {0.0f, 0.0f};
		bool has_f = mp_move_action_get_float(
			filter, action, (action->easing > 0.0f), &value_f);
		bool has_v = mp_move_action_get_vec2(
			filter, action, (action->easing > 0.0f), &value_v);

		// --- 4. EXECUTE ACTIONS BASED ON FEATURE VALUES ---
		if (action->action == ACTION_MOVE_SOURCE) {
			obs_source_t *scene_source =
				obs_weak_source_get_source(action->target);
			if (!scene_source)
				continue;

			obs_scene_t *scene =
				obs_scene_from_source(scene_source);
			if (!scene)
				scene = obs_group_from_source(scene_source);

			if (scene) {
				obs_sceneitem_t *item =
					obs_scene_find_source(
						scene, action->name);
				if (item) {
					switch (action->property) {
					case SCENEITEM_PROPERTY_POS:
						if (has_v) {
							struct vec2 pos;
							pos.x = value_v.x *
									action->factor +
								action->diff;
							pos.y = value_v.y *
									action->factor2 +
								action->diff2;
							obs_sceneitem_set_pos(
								item, &pos);
						}
						break;
					case SCENEITEM_PROPERTY_POSX:
						if (has_f) {
							struct vec2 pos;
							obs_sceneitem_get_pos(
								item, &pos);
							pos.x = value_f *
									action->factor +
								action->diff;
							obs_sceneitem_set_pos(
								item, &pos);
						}
						break;
					case SCENEITEM_PROPERTY_POSY:
						if (has_f) {
							struct vec2 pos;
							obs_sceneitem_get_pos(
								item, &pos);
							pos.y = value_f *
									action->factor +
								action->diff;
							obs_sceneitem_set_pos(
								item, &pos);
						}
						break;
					case SCENEITEM_PROPERTY_SCALE:
						if (has_v) {
							struct vec2 scale;
							scale.x = value_v.x *
									action->factor +
								action->diff;
							scale.y = value_v.y *
									action->factor2 +
								action->diff2;
							obs_sceneitem_set_scale(
								item, &scale);
						}
						break;
					case SCENEITEM_PROPERTY_SCALEX:
						if (has_f) {
							struct vec2 scale;
							obs_sceneitem_get_scale(
								item, &scale);
							scale.x = value_f *
									action->factor +
								action->diff;
							obs_sceneitem_set_scale(
								item, &scale);
						}
						break;
					case SCENEITEM_PROPERTY_SCALEY:
						if (has_f) {
							struct vec2 scale;
							obs_sceneitem_get_scale(
								item, &scale);
							scale.y = value_f *
									action->factor +
								action->diff;
							obs_sceneitem_set_scale(
								item, &scale);
						}
						break;
					case SCENEITEM_PROPERTY_ROT:
						if (has_f) {
							float rot = value_f *
								    action->factor +
							    action->diff;
							obs_sceneitem_set_rot(
								item, rot);
						}
						break;
					case SCENEITEM_PROPERTY_CROP_LEFT:
					case SCENEITEM_PROPERTY_CROP_RIGHT:
					case SCENEITEM_PROPERTY_CROP_BOTTOM:
					case SCENEITEM_PROPERTY_CROP_TOP:
						if (has_f) {
							struct obs_sceneitem_crop
								crop;
							obs_sceneitem_get_crop(
								item, &crop);
							int val =
								(int)(value_f *
									      action->factor +
								      action->diff);
							if (action->property ==
							    SCENEITEM_PROPERTY_CROP_LEFT)
								crop.left = val;
							else if (action->property ==
								 SCENEITEM_PROPERTY_CROP_RIGHT)
								crop.right =
									val;
							else if (action->property ==
								 SCENEITEM_PROPERTY_CROP_BOTTOM)
								crop.bottom =
									val;
							else if (action->property ==
								 SCENEITEM_PROPERTY_CROP_TOP)
								crop.top = val;
							obs_sceneitem_set_crop(
								item, &crop);
						}
						break;
					default:
						break;
					}
				}
			}
			obs_source_release(scene_source);

		} else if (action->action == ACTION_MOVE_VALUE) {
			obs_source_t *target_source =
				obs_weak_source_get_source(action->target);
			if (target_source && has_f) {
				obs_data_t *settings =
					obs_source_get_settings(
						target_source);
				float val = value_f * action->factor +
					    action->diff;
				if (action->is_int) {
					obs_data_set_int(
						settings, action->name,
						(long long)val);
				} else {
					obs_data_set_double(settings,
							    action->name,
							    (double)val);
				}
				obs_source_update(target_source, settings);
				obs_data_release(settings);
				obs_source_release(target_source);
			}

		} else if (action->action == ACTION_ENABLE_FILTER ||
			   action->action == ACTION_SOURCE_VISIBILITY) {
			if (has_f) {
				mp_move_apply_threshold_action(filter, action,
							       value_f);
			}
		}
	}
}

static void mp_move_render(void *data, gs_effect_t *effect)

{
	UNUSED_PARAMETER(effect);
	struct mediapipe_move_info *filter =
		(struct mediapipe_move_info *)data;

	if (filter->processing_stop) {
		obs_source_skip_video_filter(filter->source);
		return;
	}

	obs_source_t *const target = obs_filter_get_target(filter->source);
	obs_source_t *const parent = obs_filter_get_parent(filter->source);
	uint32_t base_width = obs_source_get_base_width(target);
	uint32_t base_height = obs_source_get_base_height(target);
	if (!base_width || !base_height || !target || !parent) {
		obs_source_skip_video_filter(filter->source);
		return;
	}

	if (filter->processed_frame) {
		mp_move_draw_frame(filter, base_width, base_height);
		return;
	}

	// Update width/height for detection
	filter->width = base_width;
	filter->height = base_height;

	// Color space handling
	const enum gs_color_space preferred_spaces[] = {
		GS_CS_SRGB,
		GS_CS_SRGB_16F,
		GS_CS_709_EXTENDED,
	};
	const enum gs_color_space space = obs_source_get_color_space(
		target, OBS_COUNTOF(preferred_spaces), preferred_spaces);
	if (filter->space != space) {
		filter->space = space;
		filter->initial_render = false;
	}
	const enum gs_color_format format = gs_get_format_from_space(space);

	// Render target to texrender
	if (!filter->render ||
	    gs_texrender_get_format(filter->render) != format) {
		gs_texrender_destroy(filter->render);
		filter->render =
			gs_texrender_create(format, GS_ZS_NONE);
	} else {
		gs_texrender_reset(filter->render);
	}

	if (!filter->render) {
		obs_source_skip_video_filter(filter->source);
		return;
	}

	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	if (gs_texrender_begin_with_color_space(filter->render, base_width,
						base_height, space)) {
		const float w = (float)base_width;
		const float h = (float)base_height;
		uint32_t parent_flags =
			obs_source_get_output_flags(target);
		bool custom_draw =
			(parent_flags & OBS_SOURCE_CUSTOM_DRAW) != 0;
		bool async = (parent_flags & OBS_SOURCE_ASYNC) != 0;
		struct vec4 clear_color;
		vec4_zero(&clear_color);
		gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0f, 0);
		gs_ortho(0.0f, w, 0.0f, h, -100.0f, 100.0f);

		if (target == parent && !custom_draw && !async)
			obs_source_default_render(target);
		else
			obs_source_video_render(target);
		gs_texrender_end(filter->render);
	}
	gs_blend_state_pop();

	// Convert to unorm format
	if (!filter->render_unorm) {
		filter->render_unorm =
			gs_texrender_create(GS_BGRA_UNORM, GS_ZS_NONE);
	} else {
		gs_texrender_reset(filter->render_unorm);
	}

	if (gs_texrender_begin_with_color_space(filter->render_unorm,
						base_width, base_height,
						GS_CS_SRGB)) {
		const bool previous = gs_framebuffer_srgb_enabled();
		gs_enable_framebuffer_srgb(true);
		gs_enable_blending(false);
		gs_ortho(0.0f, (float)filter->width, 0.0f,
			 (float)filter->height, -100.0f, 100.0f);

		const char *tech_name = "ConvertUnorm";
		float multiplier = 1.f;
		switch (space) {
		case GS_CS_709_EXTENDED:
			tech_name = "ConvertUnormTonemap";
			break;
		case GS_CS_709_SCRGB:
			tech_name = "ConvertUnormMultiplyTonemap";
			multiplier = 80.0f / obs_get_video_sdr_white_level();
		default:
			break;
		}

		gs_effect_set_texture_srgb(
			filter->image_param,
			gs_texrender_get_texture(filter->render));
		gs_effect_set_float(filter->multiplier_param, multiplier);

		while (gs_effect_loop(filter->effect, tech_name))
			gs_draw(GS_TRIS, 0, 3);

		gs_texrender_end(filter->render_unorm);
		gs_enable_blending(true);
		gs_enable_framebuffer_srgb(previous);
	}

	// --- Readback unorm texture to CPU for MediaPipe detection ---
	bool detection_done = false;
	gs_texture_t *unorm_tex =
		gs_texrender_get_texture(filter->render_unorm);
	if (unorm_tex && filter->width > 0 && filter->height > 0) {
		uint32_t required_size =
			filter->width * filter->height * 4; // BGRA
		if (!filter->cpu_buffer ||
		    filter->cpu_buffer_size < required_size) {
			bfree(filter->cpu_buffer);
			filter->cpu_buffer =
				(uint8_t *)bmalloc(required_size);
			filter->cpu_buffer_size = required_size;
		}

		if (filter->cpu_buffer) {
			// Use staging surface for GPU→CPU readback
			if (!filter->stage_surf ||
			    gs_stagesurface_get_width(filter->stage_surf) != filter->width ||
			    gs_stagesurface_get_height(filter->stage_surf) != filter->height) {
				if (filter->stage_surf) {
					gs_stagesurface_destroy(filter->stage_surf);
				}
				filter->stage_surf = gs_stagesurface_create(
					filter->width, filter->height,
					GS_BGRA_UNORM);
			}
			if (filter->stage_surf) {
				gs_stage_texture(filter->stage_surf,
						 unorm_tex);

				uint32_t mapped_linesize = 0;
				uint8_t *mapped_ptr = NULL;
				bool mapped = gs_stagesurface_map(
					filter->stage_surf,
					&mapped_ptr, &mapped_linesize);
				if (mapped && mapped_ptr) {
					// Copy GPU data to cpu_buffer row-by-row, handling stride
					uint32_t copy_width = filter->width * 4;
					uint32_t min_linesize = mapped_linesize < copy_width
									 ? mapped_linesize
									 : copy_width;
					for (uint32_t row = 0; row < filter->height; row++) {
						memcpy(filter->cpu_buffer + row * copy_width,
						       mapped_ptr + row * mapped_linesize,
						       min_linesize);
					}
					gs_stagesurface_unmap(filter->stage_surf);
				}

				mp_api_t *api = mediapipe_get_api();

				// --- Run face landmarker detection ---
				if (filter->face_graph) {
					MpImage *mp_image = api->image_create_from_uint8_data(
						(int)filter->width,
						(int)filter->height, 4,
						filter->cpu_buffer, NULL);
					if (mp_image) {
						FaceLandmarkerResultC face_result;
						memset(&face_result, 0,
						       sizeof(face_result));

						api->face_landmarker_detect_image(
							filter->face_graph,
							mp_image, NULL,
							&face_result);

						// Parse face landmarks
						if (face_result.face_landmarks_count > 0 &&
						    face_result.face_landmarks[0].landmarks_count > 0) {
							uint32_t num = face_result.face_landmarks[0].landmarks_count;
							da_resize(filter->landmarks, num);
							da_resize(filter->landmarks_confidence, num);
							for (uint32_t j = 0; j < num; j++) {
								NormalizedLandmarkC *lm =
									&face_result.face_landmarks[0].landmarks[j];
								filter->landmarks.array[j].x = lm->x;
								filter->landmarks.array[j].y = lm->y;
								filter->landmarks.array[j].z = lm->z;
								filter->landmarks_confidence.array[j] =
									lm->has_visibility ? lm->visibility : 1.0f;
							}
						}

						// Parse face blendshapes (expressions)
						if (face_result.face_blendshapes_count > 0 &&
						    face_result.face_blendshapes[0].categories_count > 0) {
							uint32_t num = face_result.face_blendshapes[0].categories_count;
							da_resize(filter->expressions, num);
							for (uint32_t j = 0; j < num; j++) {
								filter->expressions.array[j] =
									face_result.face_blendshapes[0].categories[j].score;
							}
						}

						// Parse facial transformation matrix
						if (face_result.facial_transformation_matrixes_count > 0 &&
						    face_result.facial_transformation_matrixes[0].data) {
							MatrixC *mat = &face_result.facial_transformation_matrixes[0];
							size_t matrix_size = (size_t)mat->rows *
									     (size_t)mat->columns * sizeof(float);
							if (matrix_size <= sizeof(filter->face_transform_matrix)) {
								memcpy(filter->face_transform_matrix,
								       mat->data, matrix_size);
							}
						}

						// Compute bounding box from face landmarks
						if (face_result.face_landmarks_count > 0 &&
						    face_result.face_landmarks[0].landmarks_count > 0) {
							uint32_t num = face_result.face_landmarks[0].landmarks_count;
							float min_x = 1.0f, min_y = 1.0f;
							float max_x = 0.0f, max_y = 0.0f;
							for (uint32_t j = 0; j < num; j++) {
								float lx = face_result.face_landmarks[0].landmarks[j].x;
								float ly = face_result.face_landmarks[0].landmarks[j].y;
								if (lx < min_x) min_x = lx;
								if (ly < min_y) min_y = ly;
								if (lx > max_x) max_x = lx;
								if (ly > max_y) max_y = ly;
							}
							filter->bbox.x = min_x;
							filter->bbox.y = min_y;
							filter->bbox.width = max_x - min_x;
							filter->bbox.height = max_y - min_y;
							filter->bbox.confidence = 1.0f;
						}

						api->face_landmarker_close_result(&face_result);
						api->image_free(mp_image);

						filter->got_new_frame = true;
						detection_done = true;
					} else {
						mp_move_log_error(filter, false,
								  "image_create_from_uint8_data");
					}
				}

				// --- Run pose landmarker detection ---
				if (filter->pose_graph) {
					MpImage *mp_image = api->image_create_from_uint8_data(
						(int)filter->width,
						(int)filter->height, 4,
						filter->cpu_buffer, NULL);
					if (mp_image) {
						PoseLandmarkerResultC pose_result;
						memset(&pose_result, 0,
						       sizeof(pose_result));

						char *mp_error = NULL;
						int ret = api->pose_landmarker_detect_image(
							filter->pose_graph,
							mp_image, NULL,
							&pose_result,
							&mp_error);
						if (ret != 0 && mp_error) {
							blog(LOG_ERROR,
							     "[MediaPipe Move] Error in pose_landmarker_detect_image: %s",
							     mp_error);
							api->error_free(mp_error);
							mp_error = NULL;
						}

						if (ret == 0) {
							// Parse 2D pose landmarks
							if (pose_result.pose_landmarks_count > 0 &&
							    pose_result.pose_landmarks[0].landmarks_count > 0) {
								uint32_t num = pose_result.pose_landmarks[0].landmarks_count;
								da_resize(filter->keypoints, num);
								da_resize(filter->keypoints_confidence, num);
								for (uint32_t j = 0; j < num; j++) {
									NormalizedLandmarkC *lm =
										&pose_result.pose_landmarks[0].landmarks[j];
									filter->keypoints.array[j].x = lm->x;
									filter->keypoints.array[j].y = lm->y;
									filter->keypoints_confidence.array[j] =
										lm->has_visibility ? lm->visibility : 1.0f;
								}
							}

							// Parse 3D pose landmarks
							if (pose_result.pose_world_landmarks_count > 0 &&
							    pose_result.pose_world_landmarks[0].landmarks_count > 0) {
								uint32_t num = pose_result.pose_world_landmarks[0].landmarks_count;
								da_resize(filter->keypoints3D, num);
								for (uint32_t j = 0; j < num; j++) {
									LandmarkC *lm =
										&pose_result.pose_world_landmarks[0].landmarks[j];
									filter->keypoints3D.array[j].x = lm->x;
									filter->keypoints3D.array[j].y = lm->y;
									filter->keypoints3D.array[j].z = lm->z;
								}
							}

							filter->got_new_frame = true;
							detection_done = true;
						}

						api->pose_landmarker_close_result(&pose_result);
						api->image_free(mp_image);
					} else {
						mp_move_log_error(filter, false,
								  "image_create_from_uint8_data");
					}
				}

			}
		}
	}

	// --- Run action evaluation when detections update ---
	if (detection_done) {
		mp_move_execute_actions(filter);
	}

	// Draw the output
	obs_source_skip_video_filter(filter->source);
}

// --- Action getters (copied from nvidia-move-filter.c pattern) ---

static bool mp_move_action_get_float(struct mediapipe_move_info *filter,
				     struct nvidia_move_action *action,
				     bool easing, float *v)
{
	bool success = false;
	*v = 0.0f;

	switch (action->feature) {
	case FEATURE_BOUNDINGBOX: {
		if (filter->bbox.width > 0.0f) {
			switch (action->feature_property) {
			case FEATURE_BOUNDINGBOX_LEFT:
				*v = filter->bbox.x;
				success = true;
				break;
			case FEATURE_BOUNDINGBOX_HORIZONTAL_CENTER:
				*v = filter->bbox.x + filter->bbox.width / 2.0f;
				success = true;
				break;
			case FEATURE_BOUNDINGBOX_RIGHT:
				*v = filter->bbox.x + filter->bbox.width;
				success = true;
				break;
			case FEATURE_BOUNDINGBOX_TOP:
				*v = filter->bbox.y;
				success = true;
				break;
			case FEATURE_BOUNDINGBOX_VERTICAL_CENTER:
				*v = filter->bbox.y + filter->bbox.height / 2.0f;
				success = true;
				break;
			case FEATURE_BOUNDINGBOX_BOTOM:
				*v = filter->bbox.y + filter->bbox.height;
				success = true;
				break;
			case FEATURE_BOUNDINGBOX_WIDTH:
				*v = filter->bbox.width;
				success = true;
				break;
			case FEATURE_BOUNDINGBOX_HEIGHT:
				*v = filter->bbox.height;
				success = true;
				break;
			case FEATURE_BOUNDINGBOX_SIZE:
				*v = filter->bbox.width * filter->bbox.height;
				success = true;
				break;
			default:
				break;
			}
		}
		break;
	}
	case FEATURE_LANDMARK: {
		if (filter->landmarks.num > 0 &&
		    action->feature_number[0] >= 0 &&
		    (uint32_t)action->feature_number[0] <
			    filter->landmarks.num) {
			mp_point3f *lm =
				filter->landmarks.array +
				action->feature_number[0];
			switch (action->feature_property) {
			case FEATURE_LANDMARK_X:
				*v = lm->x;
				success = true;
				break;
			case FEATURE_LANDMARK_Y:
				*v = lm->y;
				success = true;
				break;
			case FEATURE_LANDMARK_CONFIDENCE:
				if (action->feature_number[0] <
				    (int32_t)filter->landmarks_confidence
					    .num) {
					*v = filter->landmarks_confidence
						     .array[action->feature_number[0]];
					success = true;
				}
				break;
			case FEATURE_LANDMARK_DISTANCE:
			case FEATURE_LANDMARK_ROT:
			case FEATURE_LANDMARK_DIFF:
			case FEATURE_LANDMARK_POS:
				// Complex features - use vec2 getter
				success = false;
				break;
			default:
				break;
			}
		}
		break;
	}
	case FEATURE_EXPRESSION: {
		if (filter->expressions.num > 0 &&
		    action->feature_number[0] >= 0 &&
		    (uint32_t)action->feature_number[0] <
			    filter->expressions.num) {
			switch (action->feature_property) {
			case FEATURE_EXPRESSION_SINGLE:
				*v = filter->expressions
					     .array[action->feature_number[0]];
				success = true;
				break;
			case FEATURE_EXPRESSION_AVG:
			case FEATURE_EXPRESSION_ADD:
			case FEATURE_EXPRESSION_SUBSTRACT:
			case FEATURE_EXPRESSION_DISTANCE:
			case FEATURE_EXPRESSION_VECTOR:
				// Multi-expression features use vec2 pattern
				success = false;
				break;
			default:
				break;
			}
		}
		break;
	}
	case FEATURE_BODY: {
		if (filter->keypoints.num > 0 &&
		    action->feature_number[0] >= 0 &&
		    (uint32_t)action->feature_number[0] <
			    filter->keypoints.num) {
			mp_point3f *kp = NULL;
			if ((uint32_t)action->feature_number[0] <
			    filter->keypoints3D.num)
				kp = filter->keypoints3D.array +
				     action->feature_number[0];
			switch (action->feature_property) {
			case BODY_CONFIDENCE:
				if ((uint32_t)action->feature_number[0] <
				    filter->keypoints_confidence.num) {
					*v = filter->keypoints_confidence
						     .array[action->feature_number[0]];
					success = true;
				}
				break;
			case BODY_2D_POSX:
				*v = filter->keypoints.array[action->feature_number[0]].x;
				success = true;
				break;
			case BODY_2D_POSY:
				*v = filter->keypoints.array[action->feature_number[0]].y;
				success = true;
				break;
			case BODY_3D_POSX:
				if (kp) {
					*v = kp->x;
					success = true;
				}
				break;
			case BODY_3D_POSY:
				if (kp) {
					*v = kp->y;
					success = true;
				}
				break;
			case BODY_3D_POSZ:
				if (kp) {
					*v = kp->z;
					success = true;
				}
				break;
			case BODY_2D_DISTANCE:
			case BODY_2D_DIFF_X:
			case BODY_2D_DIFF_Y:
			case BODY_2D_DIFF:
			case BODY_2D_POS:
			case BODY_3D_DISTANCE:
			case BODY_3D_DIFF_X:
			case BODY_3D_DIFF_Y:
			case BODY_3D_DIFF_Z:
			case BODY_3D_POS:
			case BODY_3D_DIFF:
			case BODY_ANGLE_X:
			case BODY_ANGLE_Y:
			case BODY_ANGLE_Z:
			case BODY_ANGLE:
				success = false;
				break;
			default:
				break;
			}
		}
		break;
	}
	case FEATURE_POSE: {
		// Single-valued pose properties
		if (filter->face_transform_matrix[0] != 0.0f ||
		    filter->face_transform_matrix[5] != 0.0f) {
			switch (action->feature_property) {
			case FEATURE_POSE_X:
				*v = filter->face_transform_matrix[12];
				success = true;
				break;
			case FEATURE_POSE_Y:
				*v = filter->face_transform_matrix[13];
				success = true;
				break;
			case FEATURE_POSE_Z:
				*v = filter->face_transform_matrix[14];
				success = true;
				break;
			case FEATURE_POSE_W:
				*v = filter->face_transform_matrix[15];
				success = true;
				break;
			default:
				break;
			}
		}
		break;
	}
	default:
		break;
	}

	if (success && easing) {
		*v = action->previous_float * action->easing +
		     *v * (1.0f - action->easing);
		action->previous_float = *v;
	} else if (success) {
		action->previous_float = *v;
	}

	return success;
}

static bool mp_move_action_get_vec2(struct mediapipe_move_info *filter,
				    struct nvidia_move_action *action,
				    bool easing, struct vec2 *value)
{
	bool success = false;
	value->x = 0.0f;
	value->y = 0.0f;

	switch (action->feature) {
	case FEATURE_LANDMARK: {
		int32_t idx1 = action->feature_number[0];
		int32_t idx2 = action->feature_number[1];
		if (filter->landmarks.num > 0) {
			switch (action->feature_property) {
			case FEATURE_LANDMARK_DISTANCE:
				if (idx1 >= 0 && idx2 >= 0 &&
				    (uint32_t)idx1 < filter->landmarks.num &&
				    (uint32_t)idx2 < filter->landmarks.num) {
					float dx = filter->landmarks.array[idx1].x -
						   filter->landmarks.array[idx2].x;
					float dy = filter->landmarks.array[idx1].y -
						   filter->landmarks.array[idx2].y;
					value->x = dx;
					value->y = dy;
					success = true;
				}
				break;
			case FEATURE_LANDMARK_POS:
				if (idx1 >= 0 &&
				    (uint32_t)idx1 < filter->landmarks.num) {
					value->x = filter->landmarks.array[idx1].x;
					value->y = filter->landmarks.array[idx1].y;
					success = true;
				}
				break;
			case FEATURE_LANDMARK_DIFF:
				if (idx1 >= 0 && idx2 >= 0 &&
				    (uint32_t)idx1 < filter->landmarks.num &&
				    (uint32_t)idx2 < filter->landmarks.num) {
					value->x = filter->landmarks.array[idx2].x -
						   filter->landmarks.array[idx1].x;
					value->y = filter->landmarks.array[idx2].y -
						   filter->landmarks.array[idx1].y;
					success = true;
				}
				break;
			case FEATURE_LANDMARK_ROT:
				if (idx1 >= 0 && idx2 >= 0 &&
				    (uint32_t)idx1 < filter->landmarks.num &&
				    (uint32_t)idx2 < filter->landmarks.num) {
					float dx = filter->landmarks.array[idx2].x -
						   filter->landmarks.array[idx1].x;
					float dy = filter->landmarks.array[idx2].y -
						   filter->landmarks.array[idx1].y;
					value->x = dx;
					value->y = dy;
					success = true;
				}
				break;
			case FEATURE_LANDMARK_DIFF_X:
				if (idx1 >= 0 && idx2 >= 0 &&
				    (uint32_t)idx1 < filter->landmarks.num &&
				    (uint32_t)idx2 < filter->landmarks.num) {
					value->x = filter->landmarks.array[idx2].x -
						   filter->landmarks.array[idx1].x;
					success = true;
				}
				break;
			case FEATURE_LANDMARK_DIFF_Y:
				if (idx1 >= 0 && idx2 >= 0 &&
				    (uint32_t)idx1 < filter->landmarks.num &&
				    (uint32_t)idx2 < filter->landmarks.num) {
					value->x = filter->landmarks.array[idx2].y -
						   filter->landmarks.array[idx1].y;
					success = true;
				}
				break;
			default:
				break;
			}
		}
		break;
	}
	case FEATURE_EXPRESSION: {
		int32_t idx1 = action->feature_number[0];
		int32_t idx2 = action->feature_number[1];
		if (filter->expressions.num > 0) {
			switch (action->feature_property) {
			case FEATURE_EXPRESSION_AVG:
				if (idx1 >= 0 && idx2 >= 0 &&
				    (uint32_t)idx1 < filter->expressions.num &&
				    (uint32_t)idx2 < filter->expressions.num) {
					value->x = (filter->expressions.array[idx1] +
						    filter->expressions.array[idx2]) /
						   2.0f;
					success = true;
				}
				break;
			case FEATURE_EXPRESSION_ADD:
				if (idx1 >= 0 && idx2 >= 0 &&
				    (uint32_t)idx1 < filter->expressions.num &&
				    (uint32_t)idx2 < filter->expressions.num) {
					value->x = filter->expressions.array[idx1] +
						   filter->expressions.array[idx2];
					success = true;
				}
				break;
			case FEATURE_EXPRESSION_SUBSTRACT:
				if (idx1 >= 0 && idx2 >= 0 &&
				    (uint32_t)idx1 < filter->expressions.num &&
				    (uint32_t)idx2 < filter->expressions.num) {
					value->x = filter->expressions.array[idx1] -
						   filter->expressions.array[idx2];
					success = true;
				}
				break;
			case FEATURE_EXPRESSION_DISTANCE:
				if (idx1 >= 0 && idx2 >= 0 &&
				    (uint32_t)idx1 < filter->expressions.num &&
				    (uint32_t)idx2 < filter->expressions.num) {
					value->x = fabsf(filter->expressions.array[idx1] -
							  filter->expressions.array[idx2]);
					success = true;
				}
				break;
			case FEATURE_EXPRESSION_VECTOR:
				if (idx1 >= 0 &&
				    (uint32_t)idx1 < filter->expressions.num) {
					value->x = filter->expressions.array[idx1];
					success = true;
				}
				break;
			default:
				break;
			}
		}
		break;
	}
	case FEATURE_BOUNDINGBOX: {
		switch (action->feature_property) {
		case FEATURE_BOUNDINGBOX_TOP_LEFT:
			if (filter->bbox.width > 0.0f) {
				value->x = filter->bbox.x;
				value->y = filter->bbox.y;
				success = true;
			}
			break;
		case FEATURE_BOUNDINGBOX_TOP_CENTER:
			if (filter->bbox.width > 0.0f) {
				value->x = filter->bbox.x +
					   filter->bbox.width / 2.0f;
				value->y = filter->bbox.y;
				success = true;
			}
			break;
		case FEATURE_BOUNDINGBOX_TOP_RIGHT:
			if (filter->bbox.width > 0.0f) {
				value->x = filter->bbox.x +
					   filter->bbox.width;
				value->y = filter->bbox.y;
				success = true;
			}
			break;
		case FEATURE_BOUNDINGBOX_CENTER_RIGHT:
			if (filter->bbox.width > 0.0f) {
				value->x = filter->bbox.x +
					   filter->bbox.width;
				value->y = filter->bbox.y +
					   filter->bbox.height / 2.0f;
				success = true;
			}
			break;
		case FEATURE_BOUNDINGBOX_BOTTOM_RIGHT:
			if (filter->bbox.width > 0.0f) {
				value->x = filter->bbox.x +
					   filter->bbox.width;
				value->y = filter->bbox.y +
					   filter->bbox.height;
				success = true;
			}
			break;
		case FEATURE_BOUNDINGBOX_BOTTOM_CENTER:
			if (filter->bbox.width > 0.0f) {
				value->x = filter->bbox.x +
					   filter->bbox.width / 2.0f;
				value->y = filter->bbox.y +
					   filter->bbox.height;
				success = true;
			}
			break;
		case FEATURE_BOUNDINGBOX_BOTTOM_LEFT:
			if (filter->bbox.width > 0.0f) {
				value->x = filter->bbox.x;
				value->y = filter->bbox.y +
					   filter->bbox.height;
				success = true;
			}
			break;
		case FEATURE_BOUNDINGBOX_CENTER_LEFT:
			if (filter->bbox.width > 0.0f) {
				value->x = filter->bbox.x;
				value->y = filter->bbox.y +
					   filter->bbox.height / 2.0f;
				success = true;
			}
			break;
		case FEATURE_BOUNDINGBOX_CENTER:
			if (filter->bbox.width > 0.0f) {
				value->x = filter->bbox.x +
					   filter->bbox.width / 2.0f;
				value->y = filter->bbox.y +
					   filter->bbox.height / 2.0f;
				success = true;
			}
			break;
		case FEATURE_BOUNDINGBOX_LEFT:
			value->x = filter->bbox.x;
			success = true;
			break;
		case FEATURE_BOUNDINGBOX_RIGHT:
			value->x = filter->bbox.x +
				   filter->bbox.width;
			success = true;
			break;
		case FEATURE_BOUNDINGBOX_TOP:
			value->y = filter->bbox.y;
			success = true;
			break;
		case FEATURE_BOUNDINGBOX_BOTOM:
			value->y = filter->bbox.y +
				   filter->bbox.height;
			success = true;
			break;
		default:
			break;
		}
		break;
	}
	case FEATURE_BODY: {
		int32_t idx1 = action->feature_number[0];
		int32_t idx2 = action->feature_number[1];
		if (filter->keypoints.num > 0) {
			switch (action->feature_property) {
			case BODY_2D_POS:
				if (idx1 >= 0 &&
				    (uint32_t)idx1 < filter->keypoints.num) {
					value->x = filter->keypoints.array[idx1].x;
					value->y = filter->keypoints.array[idx1].y;
					success = true;
				}
				break;
			case BODY_2D_DISTANCE:
				if (idx1 >= 0 && idx2 >= 0 &&
				    (uint32_t)idx1 < filter->keypoints.num &&
				    (uint32_t)idx2 < filter->keypoints.num) {
					value->x = filter->keypoints.array[idx2].x -
						   filter->keypoints.array[idx1].x;
					value->y = filter->keypoints.array[idx2].y -
						   filter->keypoints.array[idx1].y;
					success = true;
				}
				break;
			case BODY_2D_ROT:
				if (idx1 >= 0 && idx2 >= 0 &&
				    (uint32_t)idx1 < filter->keypoints.num &&
				    (uint32_t)idx2 < filter->keypoints.num) {
					value->x = filter->keypoints.array[idx2].x -
						   filter->keypoints.array[idx1].x;
					value->y = filter->keypoints.array[idx2].y -
						   filter->keypoints.array[idx1].y;
					success = true;
				}
				break;
			case BODY_2D_DIFF_X:
				if (idx1 >= 0 && idx2 >= 0 &&
				    (uint32_t)idx1 < filter->keypoints.num &&
				    (uint32_t)idx2 < filter->keypoints.num) {
					value->x = filter->keypoints.array[idx2].x -
						   filter->keypoints.array[idx1].x;
					success = true;
				}
				break;
			case BODY_2D_DIFF_Y:
				if (idx1 >= 0 && idx2 >= 0 &&
				    (uint32_t)idx1 < filter->keypoints.num &&
				    (uint32_t)idx2 < filter->keypoints.num) {
					value->x = filter->keypoints.array[idx2].y -
						   filter->keypoints.array[idx1].y;
					success = true;
				}
				break;
			case BODY_2D_DIFF:
				if (idx1 >= 0 && idx2 >= 0 &&
				    (uint32_t)idx1 < filter->keypoints.num &&
				    (uint32_t)idx2 < filter->keypoints.num) {
					value->x = filter->keypoints.array[idx2].x -
						   filter->keypoints.array[idx1].x;
					value->y = filter->keypoints.array[idx2].y -
						   filter->keypoints.array[idx1].y;
					success = true;
				}
				break;
			case BODY_3D_POS:
				if (idx1 >= 0 &&
				    (uint32_t)idx1 < filter->keypoints3D.num) {
					value->x = filter->keypoints3D.array[idx1].x;
					value->y = filter->keypoints3D.array[idx1].y;
					success = true;
				}
				break;
			case BODY_3D_DISTANCE:
			case BODY_3D_DIFF_X:
			case BODY_3D_DIFF_Y:
			case BODY_3D_DIFF_Z:
			case BODY_3D_DIFF:
			case BODY_ANGLE_X:
			case BODY_ANGLE_Y:
			case BODY_ANGLE_Z:
			case BODY_ANGLE:
				success = false;
				break;
			default:
				break;
			}
		}
		break;
	}
	default:
		break;
	}

	if (success && easing) {
		value->x = action->previous_vec2.x * action->easing +
			   value->x * (1.0f - action->easing);
		action->previous_vec2.x = value->x;
		value->y = action->previous_vec2.y * action->easing +
			   value->y * (1.0f - action->easing);
		action->previous_vec2.y = value->y;
	} else if (success) {
		action->previous_vec2 = *value;
	}

	return success;
}

// --- Tick and color space ---

static void mp_move_tick(void *data, float t)
{
	UNUSED_PARAMETER(t);
	struct mediapipe_move_info *filter =
		(struct mediapipe_move_info *)data;

	if (filter->processing_stop)
		return;

	if (filter->initial_render) {
		filter->initial_render = false;
	}

	// Reset got_new_frame flag each tick
	filter->got_new_frame = false;
}

static enum gs_color_space mp_move_get_color_space(
	void *data, size_t count,
	const enum gs_color_space *preferred_spaces)
{
	UNUSED_PARAMETER(data);
	UNUSED_PARAMETER(count);
	UNUSED_PARAMETER(preferred_spaces);
	return GS_CS_SRGB;
}

// --- Properties (skeleton — to be filled from nvidia-move-filter.c) ---

static void mp_move_defaults(obs_data_t *settings)
{
	UNUSED_PARAMETER(settings);
	obs_data_set_default_int(settings, "actions", 1);
	obs_data_set_default_double(settings, "face_detection_confidence", 0.5);
	obs_data_set_default_double(settings, "face_presence_confidence", 0.5);
	obs_data_set_default_double(settings, "face_tracking_confidence", 0.5);
}

static obs_properties_t *mp_move_properties(void *data)
{
	UNUSED_PARAMETER(data);
	obs_properties_t *props = obs_properties_create();

	obs_properties_add_int(props, "actions", "Actions", 1, MAX_ACTIONS, 1);

	obs_properties_add_float_slider(props, "face_detection_confidence",
					"Face Detection Confidence", 0.0,
					1.0, 0.01);
	obs_properties_add_float_slider(props, "face_presence_confidence",
					"Face Presence Confidence", 0.0, 1.0,
					0.01);
	obs_properties_add_float_slider(props, "face_tracking_confidence",
					"Face Tracking Confidence", 0.0, 1.0,
					0.01);
	UNUSED_PARAMETER(data);
	return props;
}

static void mp_move_fill_body_list(obs_property_t *p)
{
	UNUSED_PARAMETER(p);
}

static void mp_move_fill_landmark_list(obs_property_t *p)
{
	UNUSED_PARAMETER(p);
}

static void mp_move_fill_expression_list(obs_property_t *p)
{
	UNUSED_PARAMETER(p);
}

static void swap_setting(obs_data_t *settings, char *setting1,
			 char *setting2)
{
	UNUSED_PARAMETER(settings);
	UNUSED_PARAMETER(setting1);
	UNUSED_PARAMETER(setting2);
}

static void swap_action(obs_data_t *settings, long long a, long long b)
{
	UNUSED_PARAMETER(settings);
	UNUSED_PARAMETER(a);
	UNUSED_PARAMETER(b);
}

static bool mp_move_move_up_clicked(obs_properties_t *props,
				    obs_property_t *property, void *data)
{
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(property);
	UNUSED_PARAMETER(data);
	return true;
}

static bool mp_move_move_down_clicked(obs_properties_t *props,
				      obs_property_t *property, void *data)
{
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(property);
	UNUSED_PARAMETER(data);
	return true;
}

static bool mp_move_get_value_clicked(obs_properties_t *props,
				      obs_property_t *property, void *data)
{
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(property);
	UNUSED_PARAMETER(data);
	return true;
}

static bool mp_move_feature_changed(void *priv, obs_properties_t *props,
				    obs_property_t *property,
				    obs_data_t *settings)
{
	UNUSED_PARAMETER(priv);
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(property);
	UNUSED_PARAMETER(settings);
	return true;
}

static bool mp_move_landmark_changed(void *priv, obs_properties_t *props,
				     obs_property_t *property,
				     obs_data_t *settings)
{
	UNUSED_PARAMETER(priv);
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(property);
	UNUSED_PARAMETER(settings);
	return true;
}

static bool mp_move_body_changed(void *priv, obs_properties_t *props,
				 obs_property_t *property,
				 obs_data_t *settings)
{
	UNUSED_PARAMETER(priv);
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(property);
	UNUSED_PARAMETER(settings);
	return true;
}

static bool mp_move_expression_changed(void *priv, obs_properties_t *props,
				       obs_property_t *property,
				       obs_data_t *settings)
{
	UNUSED_PARAMETER(priv);
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(property);
	UNUSED_PARAMETER(settings);
	return true;
}

static bool mp_move_actions_changed(void *priv, obs_properties_t *props,
				    obs_property_t *property,
				    obs_data_t *settings)
{
	UNUSED_PARAMETER(priv);
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(property);
	UNUSED_PARAMETER(settings);
	return true;
}

// --- mp_move_update: full action parsing and handle lifecycle ---

static void mp_move_update(void *data, obs_data_t *settings)
{
	struct mediapipe_move_info *filter =
		(struct mediapipe_move_info *)data;

	// --- Resize actions array ---
	size_t actions = (size_t)obs_data_get_int(settings, "actions");
	if (actions < filter->actions.num) {
		for (size_t i = actions; i < filter->actions.num; i++) {
			struct nvidia_move_action *action =
				filter->actions.array + i;
			obs_weak_source_release(action->target);
			action->target = NULL;
			bfree(action->name);
			action->name = NULL;
		}
		da_resize(filter->actions, actions);
	} else if (actions > filter->actions.num) {
		size_t old_num = filter->actions.num;
		da_resize(filter->actions, actions);
		for (size_t i = old_num; i < actions; i++) {
			struct nvidia_move_action *action =
				filter->actions.array + i;
			memset(action, 0,
			       sizeof(struct nvidia_move_action));
		}
	}

	// Read confidence threshold settings
	filter->face_detection_confidence = (float)obs_data_get_double(
		settings, "face_detection_confidence");
	filter->face_presence_confidence = (float)obs_data_get_double(
		settings, "face_presence_confidence");
	filter->face_tracking_confidence = (float)obs_data_get_double(
		settings, "face_tracking_confidence");

	// Parse each action from settings
	uint64_t feature_flags = 0;
	struct dstr name = {0};
	for (size_t i = 1; i <= actions; i++) {
		struct nvidia_move_action *action =
			filter->actions.array + i - 1;
		dstr_printf(&name, "action_%zu_disabled", i);
		action->disabled = obs_data_get_bool(settings, name.array);
		dstr_printf(&name, "action_%zu_action", i);
		action->action =
			(uint32_t)obs_data_get_int(settings, name.array);
		obs_weak_source_release(action->target);
		action->target = NULL;
		bfree(action->name);
		action->name = NULL;

		if (action->action == ACTION_MOVE_SOURCE ||
		    action->action == ACTION_SOURCE_VISIBILITY ||
		    action->action == ACTION_ATTACH_SOURCE) {
			dstr_printf(&name, "action_%zu_canvas", i);
			const char *canvas_name =
				obs_data_get_string(settings, name.array);
			obs_canvas_t *canvas =
				canvas_name[0] == '\0'
					? obs_get_main_canvas()
					: obs_get_canvas_by_name(
						  canvas_name);
			dstr_printf(&name, "action_%zu_scene", i);
			const char *scene_name =
				obs_data_get_string(settings, name.array);
			obs_source_t *source = NULL;
			if (canvas) {
				source = obs_canvas_get_source_by_name(
					canvas, scene_name);
				obs_canvas_release(canvas);
			}
			if (!source)
				source = obs_get_source_by_name(scene_name);
			if (source) {
				if (obs_source_is_scene(source) ||
				    obs_source_is_group(source))
					action->target = obs_source_get_weak_source(
						source);
				obs_source_release(source);
			}
			dstr_printf(&name, "action_%zu_sceneitem", i);
			action->name = bstrdup(
				obs_data_get_string(settings, name.array));
		}

		if (action->action == ACTION_MOVE_SOURCE) {
			dstr_printf(&name, "action_%zu_sceneitem_property",
				    i);
			action->property = (uint32_t)obs_data_get_int(
				settings, name.array);
		} else if (action->action == ACTION_ENABLE_FILTER ||
			   action->action == ACTION_SOURCE_VISIBILITY) {
			dstr_printf(&name, "action_%zu_enable", i);
			action->property = (uint32_t)obs_data_get_int(
				settings, name.array);
			dstr_printf(&name, "action_%zu_threshold", i);
			action->threshold = (float)obs_data_get_double(
				settings, name.array);
		} else if (action->action == ACTION_ATTACH_SOURCE) {
			dstr_printf(&name, "action_%zu_attach", i);
			action->property = (uint32_t)obs_data_get_int(
				settings, name.array);
		}

		if (action->action == ACTION_MOVE_VALUE ||
		    action->action == ACTION_ENABLE_FILTER) {
			dstr_printf(&name, "action_%zu_source", i);
			const char *source_name =
				obs_data_get_string(settings, name.array);
			obs_source_t *source =
				obs_get_source_by_name(source_name);
			if (source) {
				dstr_printf(&name, "action_%zu_filter", i);
				const char *filter_name =
					obs_data_get_string(settings,
							    name.array);
				obs_source_t *f =
					obs_source_get_filter_by_name(
						source, filter_name);
				if (f ||
				    action->action == ACTION_ENABLE_FILTER) {
					obs_source_release(source);
					source = f;
				}
				action->target =
					obs_source_get_weak_source(source);

				if (action->action == ACTION_MOVE_VALUE) {
					dstr_printf(&name,
						    "action_%zu_property",
						    i);
					action->name = bstrdup(
						obs_data_get_string(
							settings,
							name.array));

					obs_properties_t *sp =
						obs_source_properties(
							source);
					obs_property_t *p =
						obs_properties_get(
							sp, action->name);
					action->is_int =
						(obs_property_get_type(p) ==
						 OBS_PROPERTY_INT);
					obs_properties_destroy(sp);
				}
				obs_source_release(source);
			}
		}

		if (action->action == ACTION_ATTACH_SOURCE) {
			action->feature = FEATURE_LANDMARK;
		} else {
			dstr_printf(&name, "action_%zu_feature", i);
			action->feature = (uint32_t)obs_data_get_int(
				settings, name.array);
		}
		feature_flags |= (1ull << action->feature);

		if (action->feature == FEATURE_BOUNDINGBOX) {
			dstr_printf(&name, "action_%zu_bounding_box", i);
			action->feature_property =
				(uint32_t)obs_data_get_int(settings,
							   name.array);
			dstr_printf(&name,
				    "action_%zu_required_confidence", i);
			action->required_confidence =
				(float)obs_data_get_double(settings,
							   name.array);
		} else if (action->feature == FEATURE_LANDMARK) {
			dstr_printf(&name, "action_%zu_landmark", i);
			action->feature_property =
				(uint32_t)obs_data_get_int(settings,
							   name.array);
			dstr_printf(&name, "action_%zu_landmark_1", i);
			if (obs_data_get_int(settings, name.array))
				action->feature_number[0] =
					(int32_t)obs_data_get_int(
						settings, name.array) -
					1;
			dstr_printf(&name, "action_%zu_landmark_2", i);
			if (obs_data_get_int(settings, name.array))
				action->feature_number[1] =
					(int32_t)obs_data_get_int(
						settings, name.array) -
					1;
			dstr_printf(&name,
				    "action_%zu_required_confidence", i);
			action->required_confidence =
				(float)obs_data_get_double(settings,
							   name.array);
		} else if (action->feature == FEATURE_POSE) {
			dstr_printf(&name, "action_%zu_pose", i);
			action->feature_property =
				(uint32_t)obs_data_get_int(settings,
							   name.array);
		} else if (action->feature == FEATURE_EXPRESSION) {
			dstr_printf(&name,
				    "action_%zu_expression_property", i);
			action->feature_property =
				(uint32_t)obs_data_get_int(settings,
							   name.array);
			dstr_printf(&name, "action_%zu_expression", i);
			long long old = obs_data_get_int(settings,
							 name.array);
			if (old) {
				obs_data_unset_user_value(settings,
							  name.array);
				dstr_printf(&name,
					    "action_%zu_expression_1", i);
				obs_data_set_int(settings, name.array,
						 old);
			}
			dstr_printf(&name, "action_%zu_expression_1", i);
			action->feature_number[0] =
				(int32_t)obs_data_get_int(settings,
							   name.array) -
				1;
			dstr_printf(&name, "action_%zu_expression_2", i);
			action->feature_number[1] =
				(int32_t)obs_data_get_int(settings,
							   name.array) -
				1;
		} else if (action->feature == FEATURE_GAZE) {
			dstr_printf(&name, "action_%zu_gaze", i);
			action->feature_property =
				(int32_t)obs_data_get_int(settings,
							  name.array);
		} else if (action->feature == FEATURE_BODY) {
			dstr_printf(&name, "action_%zu_body", i);
			action->feature_property =
				(int32_t)obs_data_get_int(settings,
							  name.array);
			dstr_printf(&name, "action_%zu_body_1", i);
			action->feature_number[0] =
				(int32_t)obs_data_get_int(settings,
							   name.array);
			dstr_printf(&name, "action_%zu_body_2", i);
			action->feature_number[1] =
				(int32_t)obs_data_get_int(settings,
							   name.array);
			dstr_printf(&name,
				    "action_%zu_required_confidence", i);
			action->required_confidence =
				(float)obs_data_get_double(settings,
							   name.array);
		}

		// Factor/diff defaults and reads
		dstr_printf(&name, "action_%zu_factor2", i);
		obs_data_set_default_double(settings, name.array, 100.0f);
		dstr_printf(&name, "action_%zu_factor", i);
		obs_data_set_default_double(settings, name.array, 100.0f);
		action->factor = (float)obs_data_get_double(settings,
							    name.array) /
				 100.0f;
		dstr_printf(&name, "action_%zu_factor2", i);
		action->factor2 =
			(float)obs_data_get_double(settings, name.array) /
			100.0f;
		dstr_printf(&name, "action_%zu_diff", i);
		action->diff =
			(float)obs_data_get_double(settings, name.array);
		dstr_printf(&name, "action_%zu_diff2", i);
		action->diff2 =
			(float)obs_data_get_double(settings, name.array);

		dstr_printf(&name, "action_%zu_easing", i);
		action->easing = ExponentialEaseOut(
			(float)obs_data_get_double(settings, name.array) /
			100.0f);

		// Pre-calculate initial values
		if (action->action != ACTION_ATTACH_SOURCE) {
			mp_move_action_get_float(filter, action, false,
						 &action->previous_float);
			mp_move_action_get_vec2(filter, action, false,
						&action->previous_vec2);
		}
	}
	dstr_free(&name);

	// Clear last_error on settings update
	bfree(filter->last_error);
	filter->last_error = NULL;

	// --- Feature-flag-driven handle lifecycle ---
	bool needs_face =
		(feature_flags &
		 ((1ull << FEATURE_BOUNDINGBOX) |
		  (1ull << FEATURE_LANDMARK) | (1ull << FEATURE_POSE) |
		  (1ull << FEATURE_EXPRESSION) |
		  (1ull << FEATURE_GAZE))) != 0;

	bool needs_body =
		(feature_flags & (1ull << FEATURE_BODY)) != 0;

	mp_api_t *api = mediapipe_get_api();

	// --- Face graph lifecycle ---
	if (needs_face) {
		if (!filter->face_graph) {
			if (!mp_check_model("face_landmarker.task")) {
				blog(LOG_ERROR,
				     "[MediaPipe Move] face_landmarker.task missing, face features disabled");
			} else {
				char *model_path = obs_module_file(
					"face_landmarker.task");
				FaceLandmarkerOptionsC opts;
				memset(&opts, 0, sizeof(opts));
				opts.base_options.model_asset_path =
					model_path;
				opts.base_options.delegate =
					0; // CPU
				opts.running_mode = 1; // IMAGE mode
				opts.num_faces = 1;
				opts.min_face_detection_confidence =
					filter->face_detection_confidence;
				opts.min_face_presence_confidence =
					filter->face_presence_confidence;
				opts.min_tracking_confidence =
					filter->face_tracking_confidence;
				opts.output_face_blendshapes =
					(feature_flags &
					 (1ull << FEATURE_EXPRESSION)) !=
					0;
				opts.output_facial_transformation_matrixes =
					(feature_flags &
					 ((1ull << FEATURE_POSE) |
					  (1ull <<
					   FEATURE_BOUNDINGBOX))) != 0;

				char *error_msg = NULL;
				int ret = api->face_landmarker_create(
					&opts, &filter->face_graph,
					&error_msg);
				if (ret == 0) {
					blog(LOG_INFO,
					     "[MediaPipe Move] Created face landmarker (blendshapes=%d, transform=%d)",
					     opts.output_face_blendshapes,
					     opts.output_facial_transformation_matrixes);
				} else {
					blog(LOG_ERROR,
					     "[MediaPipe Move] Failed to create face landmarker: %s",
					     error_msg ? error_msg
						       : "unknown");
					if (error_msg)
						api->error_free(error_msg);
				}
				bfree(model_path);
			}
		}
	} else if (filter->face_graph) {
		char *error_msg = NULL;
		int ret = api->face_landmarker_close(
			filter->face_graph, &error_msg);
		if (ret != 0 && error_msg) {
			blog(LOG_ERROR,
			     "[MediaPipe Move] Error closing face landmarker: %s",
			     error_msg);
			api->error_free(error_msg);
		}
		filter->face_graph = NULL;
		blog(LOG_DEBUG,
		     "[MediaPipe Move] Destroyed face landmarker (no longer needed)");
	}

	// --- Pose (body) graph lifecycle ---
	if (needs_body) {
		if (!filter->pose_graph) {
			if (!mp_check_model("pose_landmarker.task")) {
				blog(LOG_ERROR,
				     "[MediaPipe Move] pose_landmarker.task missing, body features disabled");
			} else {
				char *model_path = obs_module_file(
					"pose_landmarker.task");
				PoseLandmarkerOptionsC opts;
				memset(&opts, 0, sizeof(opts));
				opts.base_options.model_asset_path =
					model_path;
				opts.base_options.delegate =
					0; // CPU
 				opts.running_mode = 1; // IMAGE mode
 				opts.num_poses = 1;
				opts.min_pose_detection_confidence =
					0.5f;
				opts.min_pose_presence_confidence =
					0.5f;
				opts.min_tracking_confidence = 0.5f;
				opts.output_segmentation_masks = false;

				char *error_msg = NULL;
				int ret = api->pose_landmarker_create(
					&opts, &filter->pose_graph,
					&error_msg);
				if (ret == 0) {
					blog(LOG_INFO,
					     "[MediaPipe Move] Created pose landmarker");
				} else {
					blog(LOG_ERROR,
					     "[MediaPipe Move] Failed to create pose landmarker: %s",
					     error_msg ? error_msg
						       : "unknown");
					if (error_msg)
						api->error_free(error_msg);
				}
				bfree(model_path);
			}
		}
	} else if (filter->pose_graph) {
		char *error_msg = NULL;
		int ret = api->pose_landmarker_close(
			filter->pose_graph, &error_msg);
		if (ret != 0 && error_msg) {
			blog(LOG_ERROR,
			     "[MediaPipe Move] Error closing pose landmarker: %s",
			     error_msg);
			api->error_free(error_msg);
		}
		filter->pose_graph = NULL;
		blog(LOG_DEBUG,
		     "[MediaPipe Move] Destroyed pose landmarker (no longer needed)");
	}

	// --- Resize output arrays based on which features are active ---
	if (feature_flags &
	    ((1ull << FEATURE_LANDMARK) | (1ull << FEATURE_BOUNDINGBOX) |
	     (1ull << FEATURE_POSE) | (1ull << FEATURE_GAZE))) {
		if (!filter->landmarks_confidence.num)
			da_resize(filter->landmarks_confidence, 478);
		if (!filter->landmarks.num)
			da_resize(filter->landmarks, 478);
	}
	if (feature_flags & (1ull << FEATURE_BOUNDINGBOX)) {
		if (!filter->bboxes_confidence.num)
			da_resize(filter->bboxes_confidence,
				  BBOXES_COUNT);
	}
	if (feature_flags & (1ull << FEATURE_EXPRESSION)) {
		if (!filter->expressions.num)
			da_resize(filter->expressions, 52);
	}
	if (feature_flags & (1ull << FEATURE_BODY)) {
		if (!filter->keypoints_confidence.num)
			da_resize(filter->keypoints_confidence, 33);
		if (!filter->keypoints.num)
			da_resize(filter->keypoints, 33);
		if (!filter->keypoints3D.num)
			da_resize(filter->keypoints3D, 33);
		if (!filter->joint_angles.num)
			da_resize(filter->joint_angles, 33);
	}
}

// --- Registration ---

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
	.video_tick = mp_move_tick,
	.video_render = mp_move_render,
	.filter_video = NULL,
	.load = mp_move_update,
	.video_get_color_space = mp_move_get_color_space,
};

extern "C" void mp_move_register(void)
{
	obs_register_source(&mediapipe_move_filter);
}
