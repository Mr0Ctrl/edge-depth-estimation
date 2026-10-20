#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace spatial {

/**
 * @brief Representation of an individual 3D Gaussian ellipsoid.
 */
struct Gaussian3D {
    float x, y, z;          // 3D position in camera space
    float scaleX, scaleY, scaleZ; // Covariance semi-axes
    float r, g, b;          // Color [0.0, 1.0]
    float opacity;          // Alpha opacity [0.0, 1.0]
};

/**
 * @brief Camera Intrinsic Calibration parameters.
 */
struct CameraIntrinsics {
    float fx;   // Focal length in pixels (X)
    float fy;   // Focal length in pixels (Y)
    float cx;   // Principal point (center X)
    float cy;   // Principal point (center Y)

    static CameraIntrinsics defaultFromSize(int width, int height) {
        // Standard pin-hole approximation (approx 55 deg FOV)
        float f = width * 1.0f;
        return {f, f, width * 0.5f, height * 0.5f};
    }
};

/**
 * @brief Unprojects 2D image + depth map into a 3D Gaussian Splatting scene
 * and exports to universal .ply format.
 */
class GaussianSplatEngine {
public:
    GaussianSplatEngine();
    ~GaussianSplatEngine();

    /**
     * @brief Unprojects RGB image and Depth map into 3D Gaussians.
     * @param image BGR image (CV_8UC3).
     * @param depthMap Normalized linear metric depth (CV_32FC1).
     * @param intrinsics Camera intrinsic parameters.
     * @param maxDepthMeters Physical maximum distance in meters (e.g. 10.0m).
     * @param step Subsampling step (e.g. 1 for full resolution, 2 for half).
     * @return Vector of generated 3D Gaussians.
     */
    std::vector<Gaussian3D> generate(
        const cv::Mat& image,
        const cv::Mat& depthMap,
        const CameraIntrinsics& intrinsics,
        float maxDepthMeters = 10.0f,
        int step = 2
    );

    /**
     * @brief Exports 3D Gaussians to universal binary or ASCII .ply file
     * compatible with SuperSplat, WebGL viewers, and Nerfstudio.
     * @param filename Destination file path (e.g. "output.ply").
     * @param gaussians List of 3D Gaussians to write.
     * @return true on success.
     */
    bool exportPLY(const std::string& filename, const std::vector<Gaussian3D>& gaussians);
};

} // namespace spatial
