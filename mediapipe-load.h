// MediaPipe dynamic load shim for real MediaPipe Tasks C API
// Loads libmediapipe.so and exposes the C API via function pointers.

#ifndef MEDIAPIPE_MOVE_FILTER_H
#define MEDIAPIPE_MOVE_FILTER_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <util/darray.h>

#ifdef __cplusplus
extern "C" {
#endif

// ----- Opaque handles -----
typedef struct MpImage MpImage;
// The structs BaseOptionsC, NormalizedLandmarkC, NormalizedLandmarksC, LandmarkC,
// LandmarksC, CategoryC, CategoriesC, MatrixC, FaceLandmarkerResultC,
// PoseLandmarkerResultC, ImageProcessingOptionsC, FaceLandmarkerOptionsC,
// PoseLandmarkerOptionsC are fully defined below.
// Forward declarations are not needed since definitions precede usage.

// ----- Point/rect types (internal use, not MediaPipe API) -----
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

// ----- Real MediaPipe Tasks C API structs -----

// BaseOptionsC: model loading and delegate configuration.
// ABI pinned to the installed MediaPipe Tasks C library: 56 bytes on x86_64.
// Newer versions add fields, so do not substitute their header blindly.
typedef struct {
	const char *model_asset_buffer;
	unsigned int model_asset_buffer_count;
	const char *model_asset_path;
	int delegate;         // 0=CPU, 1=GPU
	int host_environment;
	int host_system;
	const char *host_version;
	const char *ca_bundle_path;
} BaseOptionsC;

// Single normalized landmark
typedef struct {
	float x;
	float y;
	float z;
	bool has_visibility;
	float visibility;
	bool has_presence;
	float presence;
	char *name;
} NormalizedLandmarkC;

// Array of normalized landmarks
typedef struct {
	NormalizedLandmarkC *landmarks;
	uint32_t landmarks_count;
} NormalizedLandmarksC;

// Single landmark (world coordinates)
typedef struct {
	float x;
	float y;
	float z;
	bool has_visibility;
	float visibility;
	bool has_presence;
	float presence;
	char *name;
} LandmarkC;

// Array of landmarks
typedef struct {
	LandmarkC *landmarks;
	uint32_t landmarks_count;
} LandmarksC;

// Single category (for blendshapes)
typedef struct {
	int index;
	float score;
	char *category_name;
	char *display_name;
} CategoryC;

// Array of categories
typedef struct {
	CategoryC *categories;
	uint32_t categories_count;
} CategoriesC;

// Matrix (for facial transformation)
typedef struct {
	uint32_t rows;
	uint32_t columns;
	float *data;
} MatrixC;

// FaceLandmarkerResult
typedef struct {
	NormalizedLandmarksC *face_landmarks;
	uint32_t face_landmarks_count;
	CategoriesC *face_blendshapes;
	uint32_t face_blendshapes_count;
	MatrixC *facial_transformation_matrixes;
	uint32_t facial_transformation_matrixes_count;
} FaceLandmarkerResultC;

// PoseLandmarkerResult
typedef struct {
	void **segmentation_masks;
	uint32_t segmentation_masks_count;
	NormalizedLandmarksC *pose_landmarks;
	uint32_t pose_landmarks_count;
	LandmarksC *pose_world_landmarks;
	uint32_t pose_world_landmarks_count;
} PoseLandmarkerResultC;

// ImageProcessingOptions
typedef struct {
	int has_region_of_interest;
	float left, top, right, bottom;
	int rotation_degrees;
} ImageProcessingOptionsC;

// FaceLandmarkerOptions: configuration for creating a FaceLandmarker
typedef struct {
	BaseOptionsC base_options;
	int running_mode; // 0=IMAGE, 1=VIDEO, 2=LIVE_STREAM
	int num_faces;
	float min_face_detection_confidence;
	float min_face_presence_confidence;
	float min_tracking_confidence;
	bool output_face_blendshapes;
	bool output_facial_transformation_matrixes;
	void *result_callback;
} FaceLandmarkerOptionsC;

// PoseLandmarkerOptions: configuration for creating a PoseLandmarker
typedef struct {
	BaseOptionsC base_options;
	int running_mode; // 0=IMAGE, 1=VIDEO, 2=LIVE_STREAM
	int num_poses;
	float min_pose_detection_confidence;
	float min_pose_presence_confidence;
	float min_tracking_confidence;
	bool output_segmentation_masks;
	void *result_callback;
} PoseLandmarkerOptionsC;

#if defined(__cplusplus) && defined(__x86_64__) && defined(__linux__)
// These layouts match the locally installed July 2026 libmediapipe.so.
// Its CppFaceLandmarkerCreate reads running_mode at offset 0x38.
static_assert(sizeof(BaseOptionsC) == 56, "MediaPipe BaseOptions ABI mismatch");
static_assert(offsetof(FaceLandmarkerOptionsC, running_mode) == 56,
              "MediaPipe FaceLandmarkerOptions ABI mismatch");
static_assert(offsetof(PoseLandmarkerOptionsC, running_mode) == 56,
              "MediaPipe PoseLandmarkerOptions ABI mismatch");
static_assert(sizeof(MatrixC) == 16, "MediaPipe matrix ABI mismatch");
#endif

// ----- Dynamic API function table -----

typedef struct mp_api_t {
	bool loaded;
	void *handle;

	// MpImage functions
	// MediaPipe Tasks C: returns MpStatus and fills an out pointer.
	int (*image_create_from_uint8_data)(int format, int width, int height,
					     const unsigned char *data, int data_size,
					     MpImage **out, char **error_msg);
	void (*image_free)(MpImage *image);

	// Error handling
	void (*error_free)(char *error_msg);

	// Face Landmarker
	int (*face_landmarker_create)(FaceLandmarkerOptionsC *options,
				     void **landmarker, char **error_msg);
	int (*face_landmarker_detect_image)(void *landmarker, MpImage *image,
					    const ImageProcessingOptionsC *options,
					    FaceLandmarkerResultC *result, char **error_msg);
	void (*face_landmarker_close_result)(FaceLandmarkerResultC *result);
	int (*face_landmarker_close)(void *landmarker, char **error_msg);

	// Pose Landmarker
	int (*pose_landmarker_create)(PoseLandmarkerOptionsC *options,
				      void **landmarker, char **error_msg);
	int (*pose_landmarker_detect_image)(void *landmarker, MpImage *image,
					    ImageProcessingOptionsC *options,
					    PoseLandmarkerResultC *result,
					    char **error_msg);
	void (*pose_landmarker_close_result)(PoseLandmarkerResultC *result);
	int (*pose_landmarker_close)(void *landmarker, char **error_msg);
} mp_api_t;

// Global state
extern bool mediapipe_loaded;
extern mp_api_t mediapipe_api;

// Exported load/unload functions
bool mediapipe_load_library(void);
void mediapipe_unload_library(void);
bool mediapipe_is_loaded(void);
mp_api_t *mediapipe_get_api(void);

#ifdef __cplusplus
}
#endif

#endif // MEDIAPIPE_MOVE_FILTER_H
