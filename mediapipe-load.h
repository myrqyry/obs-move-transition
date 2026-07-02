// MediaPipe dynamic load shim - mirror pattern of nvar-load.h

#ifndef MEDIAPIPE_MOVE_FILTER_H
#define MEDIAPIPE_MOVE_FILTER_H

#include <stdbool.h>
#include <util/darray.h>

// MediaPipe graph handles for different feature types
typedef struct mp_graph_t mp_graph_t;
typedef struct mp_face_landmarker_t mp_face_landmarker_t;
typedef struct mp_pose_landmarker_t mp_pose_landmarker_t;

// Point/rect types - mirrors NvAR_Point2f / NvAR_Point3f / NvAR_BBoxes
typedef struct {
	float x;
	float y;
} mp_point2f;

typedef struct {
	float x;
	float y;
	float z;
} mp_point3f;

typedef struct {
	float x;
	float y;
	float z;
	float w;
} mp_quaternion;

typedef struct {
	float x;
	float y;
	float width;
	float height;
	float confidence;
} mp_rect_t;

// Result containers
typedef struct {
	DARRAY(mp_point3f) landmarks;	// MediaPipe outputs 478 3D landmarks
	DARRAY(float) landmarks_visibility;
	DARRAY(float) blendshape_coefficients;
	mp_rect_t bounding_box;
	float face_transform_matrix[16]; // 4x4 row-major, enabled via output_transform_matrix
} mp_face_result_t;

typedef struct {
	DARRAY(mp_point2f) keypoints;
	DARRAY(mp_point3f) keypoints_3d;
	DARRAY(float) visibility;
} mp_pose_result_t;

typedef struct {
    int    num_faces;                    // default 1
    float  min_detection_confidence;    // default 0.5
    float  min_presence_confidence;     // default 0.5
    float  min_tracking_confidence;     // default 0.5
    bool   output_blendshapes;          // must be true for FEATURE_EXPRESSION
    bool   output_transform_matrix;     // true for FEATURE_POSE
    void (*result_callback)(const mp_face_result_t *result, void *userdata);
    void  *callback_userdata;
} mp_face_landmarker_options_t;

// Dynamic API table
typedef struct {
	bool loaded;
	void *handle;

	// MediaPipe Tasks API functions
	bool (*create_face_landmarker)(const mp_face_landmarker_options_t *options, const char *model_path, mp_face_landmarker_t **landmarker);
	bool (*detect_face)(mp_face_landmarker_t *landmarker, const void *input, mp_face_result_t *result);
	void (*destroy_face_landmarker)(mp_face_landmarker_t *landmarker);
	bool (*create_pose_landmarker)(const char *model_path, mp_pose_landmarker_t **landmarker);
	bool (*detect_pose)(mp_pose_landmarker_t *landmarker, const void *input, mp_pose_result_t *result);
	void (*destroy_pose_landmarker)(mp_pose_landmarker_t *landmarker);

	// Utility functions
	const char *(*get_version)(void);
	const char *(*get_last_error)(void);
} mp_api_t;

// Global state
extern bool mediapipe_loaded;
extern mp_api_t mediapipe_api;

// Function prototypes
bool mediapipe_load_library(void);
void mediapipe_unload_library(void);
bool mediapipe_is_loaded(void);
mp_api_t *mediapipe_get_api(void);

#endif // MEDIAPIPE_MOVE_FILTER_H
