---
title: "Chapter 11: Advanced Perception with Isaac"
description: "Implementing AI-powered perception algorithms with NVIDIA Isaac"
sidebar_position: 2
---

# Chapter 11: Advanced Perception with Isaac

## Learning Objectives

After completing this chapter, you should be able to:

- Implement AI-powered perception algorithms using Isaac SDK
- Apply object detection and recognition in robotics applications
- Process depth estimation and point cloud data using GPU acceleration
- Optimize perception algorithms for real-time performance

## AI-Powered Perception in Robotics

### The Role of Perception in Robotics

Perception is the foundation of autonomous robot behavior, enabling robots to:
- Understand their environment
- Identify objects and obstacles
- Navigate safely through complex spaces
- Interact with objects and people

Traditional perception approaches often struggle with:
- Variability in lighting conditions
- Changes in object appearance
- Dynamic environments
- Real-time processing requirements

AI-powered perception addresses these challenges through:
- Deep learning models trained on large datasets
- Real-time processing enabled by GPU acceleration
- Adaptability to different scenarios
- Robustness to environmental variations

### Isaac's Perception Capabilities

NVIDIA Isaac provides specialized tools for AI-powered perception:

1. **TensorRT Integration**: Optimized inference for deep learning models
2. **CUDA Acceleration**: Parallel processing for sensor data
3. **Pre-trained Models**: Ready-to-use perception models
4. **Synthetic Data Generation**: Tools to create training datasets
5. **Sensor Fusion**: Combining data from multiple sensors

## Object Detection and Recognition

### Overview of Object Detection

Object detection involves:
- Localizing objects in images (bounding boxes)
- Classifying objects into categories
- Determining object properties (size, orientation, etc.)

Isaac provides GPU-accelerated object detection using TensorRT:

```cpp
// Isaac codelet for object detection
#include "engine/alice/alice.hpp"
#include "messages/image.capnp.h"
#include "messages/detections.capnp.h"
#include "packages/tensorrt/CodeletTensorRT.hpp"

namespace isaac {
namespace perception {

class TensorRTObjectDetector : public TensorRTCodelet {
 public:
  void start() override {
    // Subscribe to camera images
    rx_image().subscribe(&TensorRTObjectDetector::on_image, this);
    
    // Initialize TensorRT engine
    initialize_tensorrt_engine(get_model_path());
  }

  void tick() override {
    // Process images when they arrive
  }

 private:
  void on_image(const messages::ImageProto::Reader& image_reader) {
    // Convert Isaac image format to TensorRT input
    auto input = convert_image_for_tensorrt(image_reader);
    
    // Run inference with TensorRT
    auto detections = run_object_detection(input);
    
    // Publish detections
    publish_detections(detections);
  }

  std::vector<DetectedObject> run_object_detection(const Tensor& input) {
    // Execute TensorRT engine
    std::vector<float> output_buffer(output_size_);
    execute_engine(input.data(), output_buffer.data());
    
    // Process raw detections into structured format
    auto detections = postprocess_detections(output_buffer);
    
    return detections;
  }

  void publish_detections(const std::vector<DetectedObject>& detections) {
    // Create detection message
    auto detection_msg = tx_detections().initProto();
    
    // Set detection data
    auto detections_list = detection_msg.initDetections(detections.size());
    for (size_t i = 0; i < detections.size(); ++i) {
      auto detection = detections_list[i];
      detection.setClassIndex(detections[i].class_index);
      detection.setConfidence(detections[i].confidence);
      
      // Set bounding box
      auto bbox = detection.initBbox();
      bbox.setXMin(detections[i].bbox.x_min);
      bbox.setYMin(detections[i].bbox.y_min);
      bbox.setXMax(detections[i].bbox.x_max);
      bbox.setYMax(detections[i].bbox.y_max);
    }
    
    // Publish message
    tx_detections().publish();
  }

  std::string get_model_path() {
    return getParameter<std::string>("model_path", "");
  }

  ISAAC_PROTO_RX(ImageProto, "image");
  ISAAC_PROTO_TX(DetectionsProto, "detections");

  int input_size_;
  int output_size_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
};

}  // namespace perception
}  // namespace isaac

ISAAC_REGISTER_CODELET(isaac::perception::TensorRTObjectDetector)
```

### Implementing Custom Object Detection Models

#### Model Preparation

To use custom models with Isaac, you need to:

1. **Train the model** using a framework like PyTorch or TensorFlow
2. **Export to ONNX format** for interoperability
3. **Convert to TensorRT** for optimized inference
4. **Integrate into Isaac** as a codelet

```cpp
// Example of integrating a custom model
#include "engine/alice/alice.hpp"
#include "nv_onnx_parser.h"

namespace isaac {
namespace perception {

class CustomObjectDetector : public TensorRTCodelet {
 public:
  void start() override {
    // Set up TensorRT with custom model
    create_custom_tensorrt_engine();
    
    // Subscribe to input
    rx_image().subscribe(&CustomObjectDetector::on_image, this);
  }

 private:
  void create_custom_tensorrt_engine() {
    // Create TensorRT builder and network definition
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0U));
    
    // Parse ONNX model
    auto parser = nvonnxparser::createParser(*network, logger_);
    if (!parser->parseFromFile(onnx_model_path_.c_str(), 
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
      reportError("Failed to parse ONNX model");
      return;
    }
    
    // Configure builder for optimization
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMaxWorkspaceSize(1_GiB);  // Set workspace size
    
    // Build CUDA engine
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config));
    
    if (!engine_) {
      reportError("Failed to build TensorRT engine");
      return;
    }
    
    // Create execution context
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  }

  nvinfer1::ILogger logger_;
  std::string onnx_model_path_ = "models/custom_model.onnx";
};

}  // namespace perception
}  // namespace isaac
```

### Multi-Object Tracking

For applications requiring tracking of objects over time:

```cpp
// Object tracking codelet
#include "engine/alice/alice.hpp"
#include "messages/detections.capnp.h"

namespace isaac {
namespace tracking {

class MultiObjectTracker : public Codelet {
 public:
  void start() override {
    rx_detections().subscribe(&MultiObjectTracker::on_detections, this);
    tickPeriodically();
  }

  void tick() override {
    // Update tracked objects
    update_tracking();
    
    // Publish tracked objects
    publish_tracked_objects();
  }

 private:
  void on_detections(const messages::DetectionsProto::Reader& detections) {
    // Get current detections
    auto new_detections = detections.getDetections();
    
    // Associate new detections with existing tracks
    std::vector<int> assignment = associate_detections_to_tracks(new_detections);
    
    // Update existing tracks
    for (size_t i = 0; i < assignment.size(); ++i) {
      if (assignment[i] >= 0) {
        // Update existing track
        tracks_[assignment[i]].update(new_detections[i]);
      } else {
        // Create new track
        tracks_.push_back(Track(new_detections[i]));
      }
    }
    
    // Handle lost tracks
    for (size_t i = 0; i < tracks_.size(); ++i) {
      if (tracks_[i].is_lost()) {
        tracks_.erase(tracks_.begin() + i);
        i--;  // Adjust index after removal
      }
    }
  }

  std::vector<int> associate_detections_to_tracks(
      const capnp::List<messages::DetectionProto>::Reader& detections) {
    // Use Hungarian algorithm or other association method
    // Calculate cost matrix based on position, appearance, etc.
    // Return assignment of detections to existing tracks
    return std::vector<int>();
  }

  void update_tracking() {
    // Predict object movement using Kalman filters
    for (auto& track : tracks_) {
      track.predict();
    }
  }

  void publish_tracked_objects() {
    // Create and publish tracked objects message
    auto tracked_msg = tx_tracked_objects().initProto();
    auto tracked_objects = tracked_msg.initObjects(tracks_.size());
    
    for (size_t i = 0; i < tracks_.size(); ++i) {
      auto obj = tracked_objects[i];
      obj.setId(tracks_[i].get_id());
      auto bbox = tracks_[i].get_bounding_box();
      obj.setBbox(bbox);
      obj.setVelocity(tracks_[i].get_velocity());
    }
    
    tx_tracked_objects().publish();
  }

  ISAAC_PROTO_RX(DetectionsProto, "detections");
  ISAAC_PROTO_TX(TrackedObjectsProto, "tracked_objects");

  std::vector<Track> tracks_;
};

}  // namespace tracking
}  // namespace isaac

ISAAC_REGISTER_CODELET(isaac::tracking::MultiObjectTracker)
```

## Depth Estimation and Point Cloud Processing

### Depth Estimation with Neural Networks

Depth estimation algorithms predict distance from a single image or stereo pair:

```cpp
// Depth estimation codelet
#include "engine/alice/alice.hpp"
#include "messages/image.capnp.h"
#include "messages/depth.capnp.h"
#include "packages/tensorrt/CodeletTensorRT.hpp"

namespace isaac {
namespace depth_estimation {

class MonocularDepthEstimator : public TensorRTCodelet {
 public:
  void start() override {
    rx_image().subscribe(&MonocularDepthEstimator::on_image, this);
  }

  void on_image(const messages::ImageProto::Reader& image_reader) {
    // Preprocess image for depth estimation
    auto input = preprocess_image_for_depth(image_reader);
    
    // Run depth estimation model
    auto depth_map = estimate_depth(input);
    
    // Publish depth map
    publish_depth_map(depth_map);
  }

 private:
  cv::Mat estimate_depth(const cv::Mat& input_image) {
    // Convert to TensorRT format
    auto tensor_input = image_to_tensor(input_image);
    
    // Run inference
    std::vector<float> output_buffer(output_size_);
    execute_engine(tensor_input.data(), output_buffer.data());
    
    // Convert output to depth map
    cv::Mat depth_map = tensor_to_depth_map(output_buffer);
    
    return depth_map;
  }

  void publish_depth_map(const cv::Mat& depth_map) {
    // Create depth message
    auto depth_msg = tx_depth().initProto();
    
    // Set depth map data
    depth_msg.setWidth(depth_map.cols);
    depth_msg.setHeight(depth_map.rows);
    
    // Convert Mat to array
    auto depth_data = depth_msg.initData(depth_map.rows * depth_map.cols);
    for (int i = 0; i < depth_map.rows; ++i) {
      for (int j = 0; j < depth_map.cols; ++j) {
        depth_data[i * depth_map.cols + j] = depth_map.at<float>(i, j);
      }
    }
    
    tx_depth().publish();
  }

  ISAAC_PROTO_RX(ImageProto, "image");
  ISAAC_PROTO_TX(DepthProto, "depth");
};

}  // namespace depth_estimation
}  // namespace isaac

ISAAC_REGISTER_CODELET(isaac::depth_estimation::MonocularDepthEstimator)
```

### Point Cloud Processing

Point clouds from LiDAR or depth sensors provide 3D environmental information:

```cpp
// Point cloud processing codelet
#include "engine/alice/alice.hpp"
#include "messages/point_cloud.capnp.h"
#include "engine/core/point_cloud.hpp"

namespace isaac {
namespace point_cloud {

class PointCloudProcessor : public Codelet {
 public:
  void start() override {
    rx_point_cloud().subscribe(&PointCloudProcessor::on_point_cloud, this);
  }

  void on_point_cloud(const messages::PointCloudProto::Reader& cloud_reader) {
    // Convert message to internal point cloud format
    auto cloud = point_cloud_from_proto(cloud_reader);
    
    // Filter point cloud (remove noise, crop)
    auto filtered_cloud = filter_point_cloud(cloud);
    
    // Segment ground plane
    auto segmented_cloud = segment_ground_plane(filtered_cloud);
    
    // Detect obstacles
    auto obstacles = detect_obstacles(segmented_cloud);
    
    // Publish processed results
    publish_results(obstacles);
  }

 private:
  PointCloud filter_point_cloud(const PointCloud& input) {
    PointCloud output;
    
    // Apply voxel grid filter for noise reduction
    float voxel_size = 0.1f; // 10cm voxels
    std::map<VoxelCoord, std::vector<Point3>> voxel_map;
    
    // Group points by voxel
    for (const auto& point : input.points) {
      VoxelCoord voxel = {
        static_cast<int>(point.x / voxel_size),
        static_cast<int>(point.y / voxel_size),
        static_cast<int>(point.z / voxel_size)
      };
      
      voxel_map[voxel].push_back(point);
    }
    
    // Take centroid of each voxel
    for (const auto& [voxel, points] : voxel_map) {
      Point3 centroid(0, 0, 0);
      for (const auto& point : points) {
        centroid.x += point.x;
        centroid.y += point.y;
        centroid.z += point.z;
      }
      centroid.x /= points.size();
      centroid.y /= points.size();
      centroid.z /= points.size();
      
      output.points.push_back(centroid);
    }
    
    return output;
  }

  PointCloud segment_ground_plane(const PointCloud& input) {
    // Use RANSAC to find ground plane
    auto ground_model = fit_plane_ransac(input.points, max_iterations_=1000, threshold_=0.1);
    
    PointCloud non_ground_points;
    for (const auto& point : input.points) {
      float distance = point_to_plane_distance(point, ground_model);
      if (distance > 0.2f) { // Not on ground plane (20cm threshold)
        non_ground_points.points.push_back(point);
      }
    }
    
    return non_ground_points;
  }

  std::vector<Object3D> detect_obstacles(const PointCloud& input) {
    // Cluster points to detect individual objects
    auto clusters = cluster_points(input.points, cluster_tolerance_=0.5);
    
    std::vector<Object3D> obstacles;
    for (const auto& cluster : clusters) {
      // Calculate bounding box for cluster
      auto bbox = calculate_bounding_box(cluster);
      
      // Create obstacle object
      Object3D obstacle;
      obstacle.bounding_box = bbox;
      obstacle.centroid = calculate_centroid(cluster);
      obstacle.point_count = cluster.size();
      
      obstacles.push_back(obstacle);
    }
    
    return obstacles;
  }

  void publish_results(const std::vector<Object3D>& obstacles) {
    // Create and publish obstacle message
    auto obstacle_msg = tx_obstacles().initProto();
    auto obstacles_list = obstacle_msg.initObstacles(obstacles.size());
    
    for (size_t i = 0; i < obstacles.size(); ++i) {
      auto obstacle = obstacles_list[i];
      
      // Set bounding box
      auto bbox = obstacle.initBbox();
      bbox.setCenterX(obstacles[i].centroid.x);
      bbox.setCenterY(obstacles[i].centroid.y);
      bbox.setCenterZ(obstacles[i].centroid.z);
      bbox.setSizeX(obstacles[i].bounding_box.size_x);
      bbox.setSizeY(obstacles[i].bounding_box.size_y);
      bbox.setSizeZ(obstacles[i].bounding_box.size_z);
      
      obstacle.setPointCount(obstacles[i].point_count);
    }
    
    tx_obstacles().publish();
  }

  ISAAC_PROTO_RX(PointCloudProto, "point_cloud");
  ISAAC_PROTO_TX(ObstaclesProto, "obstacles");
};

}  // namespace point_cloud
}  // namespace isaac

ISAAC_REGISTER_CODELET(isaac::point_cloud::PointCloudProcessor)
```

## Real-time Performance Optimization

### GPU Memory Management

Efficient GPU memory management is crucial for real-time perception:

```cpp
// Optimized memory management for perception pipeline
#include "engine/alice/alice.hpp"
#include <cuda_runtime.h>

namespace isaac {
namespace optimization {

class PerceptionMemoryManager {
 public:
  PerceptionMemoryManager(size_t max_batch_size, size_t max_tensor_size) 
      : max_batch_size_(max_batch_size), max_tensor_size_(max_tensor_size) {
    // Allocate persistent GPU memory
    cudaMalloc(&input_buffer_, max_batch_size_ * max_tensor_size_);
    cudaMalloc(&output_buffer_, max_batch_size_ * max_tensor_size_);
    
    // Create CUDA streams for overlapping operations
    cudaStreamCreate(&inference_stream_);
    cudaStreamCreate(&copy_stream_);
  }
  
  ~PerceptionMemoryManager() {
    cudaFree(input_buffer_);
    cudaFree(output_buffer_);
    cudaStreamDestroy(inference_stream_);
    cudaStreamDestroy(copy_stream_);
  }

  // Asynchronous data transfer
  void async_copy_to_gpu(const void* host_data, size_t size, cudaStream_t stream) {
    cudaMemcpyAsync(input_buffer_, host_data, size, cudaMemcpyHostToDevice, stream);
  }

  void async_copy_from_gpu(void* host_data, size_t size, cudaStream_t stream) {
    cudaMemcpyAsync(host_data, output_buffer_, size, cudaMemcpyDeviceToHost, stream);
  }

  // Memory pools for tensor objects
  template<typename T>
  T* get_tensor_memory(size_t elements) {
    // Return memory from pool or allocate new if needed
    if (tensor_pool_.size() > 0) {
      T* tensor = tensor_pool_.back();
      tensor_pool_.pop_back();
      return tensor;
    }
    
    // Allocate new tensor
    T* new_tensor = new T[elements];
    return new_tensor;
  }

  template<typename T>
  void release_tensor_memory(T* tensor) {
    // Return tensor to pool for reuse
    tensor_pool_.push_back(tensor);
  }

 private:
  void* input_buffer_;
  void* output_buffer_;
  cudaStream_t inference_stream_;
  cudaStream_t copy_stream_;
  size_t max_batch_size_;
  size_t max_tensor_size_;
  
  // Memory pools for reuse
  std::vector<void*> tensor_pool_;
};

}  // namespace optimization
}  // namespace isaac
```

### Pipeline Optimization

Creating efficient processing pipelines for perception:

```cpp
// Perception pipeline with overlapping operations
#include "engine/alice/alice.hpp"
#include <thread>
#include <queue>
#include <mutex>

namespace isaac {
namespace pipeline {

class OptimizedPerceptionPipeline : public Codelet {
 public:
  void start() override {
    // Start processing thread
    processing_thread_ = std::thread(&OptimizedPerceptionPipeline::processing_loop, this);
    
    // Start receiving images
    rx_image().subscribe(&OptimizedPerceptionPipeline::on_image, this);
  }

  void stop() override {
    stop_processing_ = true;
    if (processing_thread_.joinable()) {
      processing_thread_.join();
    }
  }

 private:
  void on_image(const messages::ImageProto::Reader& image_reader) {
    // Add image to processing queue
    std::lock_guard<std::mutex> lock(queue_mutex_);
    image_queue_.push(image_reader);
    
    // Notify processing thread
    processing_cv_.notify_one();
  }

  void processing_loop() {
    while (!stop_processing_) {
      // Wait for new image
      messages::ImageProto::Reader image;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        processing_cv_.wait(lock, [this] { return !image_queue_.empty() || stop_processing_; });
        
        if (!image_queue_.empty()) {
          image = image_queue_.front();
          image_queue_.pop();
        }
      }
      
      if (stop_processing_) break;
      
      // Preprocess image
      auto preprocessed = preprocess_image(image);
      
      // Run object detection
      auto detections = run_object_detection(preprocessed);
      
      // Run depth estimation
      auto depth = run_depth_estimation(preprocessed);
      
      // Publish results
      publish_perception_results(detections, depth);
    }
  }

  // GPU-accelerated preprocessing
  Tensor preprocess_image(const messages::ImageProto::Reader& image) {
    // Convert and preprocess on GPU
    // Resize, normalize, convert color spaces
    return Tensor(); // Placeholder
  }

  std::vector<DetectedObject> run_object_detection(const Tensor& input) {
    // Run object detection model on GPU
    return std::vector<DetectedObject>(); // Placeholder
  }

  cv::Mat run_depth_estimation(const Tensor& input) {
    // Run depth estimation model on GPU
    return cv::Mat(); // Placeholder
  }

  void publish_perception_results(
      const std::vector<DetectedObject>& detections,
      const cv::Mat& depth_map) {
    // Publish all perception results
  }

  ISAAC_PROTO_RX(ImageProto, "image");

  std::queue<messages::ImageProto::Reader> image_queue_;
  std::mutex queue_mutex_;
  std::condition_variable processing_cv_;
  std::thread processing_thread_;
  std::atomic<bool> stop_processing_{false};
};

}  // namespace pipeline
}  // namespace isaac

ISAAC_REGISTER_CODELET(isaac::pipeline::OptimizedPerceptionPipeline)
```

## Sensor Fusion Techniques

### Combining Multiple Sensors

Fusing data from different sensors improves perception robustness:

```cpp
// Sensor fusion codelet combining camera and LiDAR
#include "engine/alice/alice.hpp"
#include "messages/image.capnp.h"
#include "messages/point_cloud.capnp.h"
#include "messages/camera.capnp.h"

namespace isaac {
namespace fusion {

class CameraLidarFusion : public Codelet {
 public:
  void start() override {
    // Subscribe to both sensors
    rx_camera_image().subscribe(&CameraLidarFusion::on_camera_image, this);
    rx_point_cloud().subscribe(&CameraLidarFusion::on_point_cloud, this);
    
    // Initialize calibration (camera to LiDAR transform)
    initialize_calibration();
  }

  void on_camera_image(const messages::ImageProto::Reader& image_reader) {
    // Store image with timestamp
    latest_image_ = image_reader;
    image_timestamp_ = image_reader.getTimestamp();
    
    // Try to fuse with available LiDAR data
    try_fusion();
  }

  void on_point_cloud(const messages::PointCloudProto::Reader& cloud_reader) {
    // Store point cloud with timestamp
    latest_point_cloud_ = cloud_reader;
    cloud_timestamp_ = cloud_reader.getTimestamp();
    
    // Try to fuse with available camera data
    try_fusion();
  }

 private:
  void initialize_calibration() {
    // Load camera intrinsics and extrinsics (camera to LiDAR transform)
    camera_intrinsics_ = get_camera_intrinsics();
    camera_to_lidar_ = get_transform("camera_frame", "lidar_frame");
  }

  void try_fusion() {
    // Check if we have both data types and they're close in time
    if (!latest_image_.hasValue() || !latest_point_cloud_.hasValue()) {
      return; // Wait for both
    }
    
    double time_diff = std::abs(image_timestamp_ - cloud_timestamp_);
    if (time_diff > max_time_diff_) {
      return; // Too much time difference
    }
    
    // Perform fusion
    auto fused_data = perform_fusion(latest_image_, latest_point_cloud_);
    
    // Publish fused results
    publish_fused_data(fused_data);
    
    // Clear processed data to avoid duplication
    latest_image_ = messages::ImageProto::Reader(); 
    latest_point_cloud_ = messages::PointCloudProto::Reader();
  }

  FusedPerceptionData perform_fusion(
      const messages::ImageProto::Reader& image,
      const messages::PointCloudProto::Reader& cloud) {
    FusedPerceptionData result;
    
    // Project 3D points to 2D image space
    auto projected_points = project_lidar_to_image(cloud, camera_intrinsics_, camera_to_lidar_);
    
    // Perform camera-based object detection
    auto camera_detections = run_object_detection(image);
    
    // Perform LiDAR-based object detection
    auto lidar_detections = run_lidar_detection(cloud);
    
    // Associate 2D and 3D detections
    result.associations = associate_detections(camera_detections, lidar_detections, projected_points);
    
    // Create enhanced detections
    result.enhanced_detections = create_enhanced_detections(
        camera_detections, lidar_detections, result.associations);
    
    return result;
  }

  std::vector<DetectionAssociation> associate_detections(
      const std::vector<DetectedObject>& camera_detections,
      const std::vector<DetectedObject3D>& lidar_detections,
      const std::vector<Point2>& projected_points) {
    std::vector<DetectionAssociation> associations;
    
    // Associate detections based on spatial proximity
    for (const auto& cam_det : camera_detections) {
      for (const auto& lidar_det : lidar_detections) {
        // Calculate overlap in projected space
        float overlap = calculate_overlap(cam_det.bbox, lidar_det.projected_bbox);
        
        if (overlap > association_threshold_) {
          associations.push_back({
            cam_det.id, 
            lidar_det.id, 
            overlap
          });
        }
      }
    }
    
    return associations;
  }

  ISAAC_PROTO_RX(ImageProto, "camera_image");
  ISAAC_PROTO_RX(PointCloudProto, "point_cloud");
  ISAAC_PROTO_TX(FusedPerceptionProto, "fused_detections");
  
  capnp::Orphan<capnp::List<messages::ImageProto>> latest_image_;
  capnp::Orphan<capnp::List<messages::PointCloudProto>> latest_point_cloud_;
  
  int64_t image_timestamp_ = 0;
  int64_t cloud_timestamp_ = 0;
  
  messages::CameraIntrinsicsProto camera_intrinsics_;
  Pose3 camera_to_lidar_;
  
  double max_time_diff_ = 0.1; // 100ms tolerance
  float association_threshold_ = 0.3; // 30% overlap threshold
};

}  // namespace fusion
}  // namespace isaac

ISAAC_REGISTER_CODELET(isaac::fusion::CameraLidarFusion)
```

## Practical Implementation Example

Here's a complete implementation combining multiple perception techniques:

```cpp
// Complete perception pipeline example
#include "engine/alice/alice.hpp"
#include "messages/image.capnp.h"
#include "messages/point_cloud.capnp.h"
#include "messages/detections.capnp.h"
#include "packages/tensorrt/CodeletTensorRT.hpp"

namespace isaac {
namespace perception {

class CompletePerceptionNode : public Codelet {
 public:
  void start() override {
    // Set up subscriptions
    rx_camera_image().subscribe(&CompletePerceptionNode::on_camera_image, this);
    rx_lidar_point_cloud().subscribe(&CompletePerceptionNode::on_lidar_data, this);
    
    // Initialize components
    initialize_perception_components();
  }

 private:
  void initialize_perception_components() {
    // Initialize object detector
    object_detector_.initialize();
    
    // Initialize depth estimator
    depth_estimator_.initialize();
    
    // Initialize point cloud processor
    point_cloud_processor_.initialize();
  }

  void on_camera_image(const messages::ImageProto::Reader& image_reader) {
    // Preprocess image
    auto processed_image = preprocess_image(image_reader);
    
    // Run object detection
    auto detections = object_detector_.detect(processed_image);
    
    // Run depth estimation
    auto depth_map = depth_estimator_.estimate_depth(processed_image);
    
    // Enhance detections with depth information
    auto enhanced_detections = enhance_detections_with_depth(detections, depth_map);
    
    // Store results and publish if fused with LiDAR
    camera_detections_ = enhanced_detections;
    try_publish_fused_detections();
  }

  void on_lidar_data(const messages::PointCloudProto::Reader& cloud_reader) {
    // Process point cloud
    auto obstacles = point_cloud_processor_.process(cloud_reader);
    
    // Store results and publish if fused with camera data
    lidar_obstacles_ = obstacles;
    try_publish_fused_detections();
  }

  void try_publish_fused_detections() {
    if (!camera_detections_.empty() && !lidar_obstacles_.empty()) {
      // Fuse camera and LiDAR detections
      auto fused_detections = fuse_camera_lidar_detections(
          camera_detections_, lidar_obstacles_);
      
      // Publish results
      publish_detections(fused_detections);
      
      // Clear processed data
      camera_detections_.clear();
      lidar_obstacles_.clear();
    }
  }

  std::vector<EnhancedDetection> enhance_detections_with_depth(
      const std::vector<DetectedObject>& detections,
      const cv::Mat& depth_map) {
    std::vector<EnhancedDetection> enhanced;
    
    for (const auto& det : detections) {
      EnhancedDetection enhanced_det;
      enhanced_det.base_detection = det;
      
      // Calculate depth at object center
      int center_x = (det.bbox.x_min + det.bbox.x_max) / 2;
      int center_y = (det.bbox.y_min + det.bbox.y_max) / 2;
      
      // Average depth in bounding box region
      float avg_depth = 0;
      int count = 0;
      for (int y = det.bbox.y_min; y <= det.bbox.y_max; ++y) {
        for (int x = det.bbox.x_min; x <= det.bbox.x_max; ++x) {
          if (x >= 0 && x < depth_map.cols && y >= 0 && y < depth_map.rows) {
            float depth = depth_map.at<float>(y, x);
            if (depth > 0) { // Valid depth
              avg_depth += depth;
              count++;
            }
          }
        }
      }
      
      enhanced_det.distance = (count > 0) ? avg_depth / count : -1;
      enhanced.push_back(enhanced_det);
    }
    
    return enhanced;
  }

  std::vector<FusedDetection> fuse_camera_lidar_detections(
      const std::vector<EnhancedDetection>& camera_detections,
      const std::vector<DetectedObject3D>& lidar_detections) {
    std::vector<FusedDetection> fused_detections;
    
    // Associate 2D and 3D detections based on spatial overlap
    for (const auto& cam_det : camera_detections) {
      FusedDetection fused_det;
      fused_det.camera_detection = cam_det;
      
      // Find closest matching 3D detection
      for (const auto& lidar_det : lidar_detections) {
        float distance = calculate_distance(cam_det, lidar_det);
        if (distance < fusion_distance_threshold_) {
          fused_det.lidar_detection = lidar_det;
          fused_det.confidence = (cam_det.confidence + lidar_det.confidence) / 2;
          break; // Found match
        }
      }
      
      fused_detections.push_back(fused_det);
    }
    
    return fused_detections;
  }

  void publish_detections(const std::vector<FusedDetection>& detections) {
    // Create message
    auto detection_msg = tx_fused_detections().initProto();
    auto detections_list = detection_msg.initDetections(detections.size());
    
    for (size_t i = 0; i < detections.size(); ++i) {
      auto detection = detections_list[i];
      
      // Set 2D detection data
      detection.setClassIndex(detections[i].camera_detection.base_detection.class_index);
      detection.setConfidence(detections[i].confidence);
      
      // Set 3D position from LiDAR
      if (detections[i].lidar_detection.has_value()) {
        auto position = detection.initPosition();
        position.setX(detections[i].lidar_detection.value().position.x);
        position.setY(detections[i].lidar_detection.value().position.y);
        position.setZ(detections[i].lidar_detection.value().position.z);
      }
    }
    
    tx_fused_detections().publish();
  }

  ISAAC_PROTO_RX(ImageProto, "camera_image");
  ISAAC_PROTO_RX(PointCloudProto, "lidar_point_cloud");
  ISAAC_PROTO_TX(FusedDetectionsProto, "fused_detections");

  // Perception components
  ObjectDetector object_detector_;
  DepthEstimator depth_estimator_;
  PointCloudProcessor point_cloud_processor_;
  
  // Results storage
  std::vector<EnhancedDetection> camera_detections_;
  std::vector<DetectedObject3D> lidar_obstacles_;
  
  // Configuration
  float fusion_distance_threshold_ = 1.0; // 1 meter threshold for fusion
};

}  // namespace perception
}  // namespace isaac

ISAAC_REGISTER_CODELET(isaac::perception::CompletePerceptionNode)
```

## Exercises

1. **Object Detection Exercise**: Implement a complete object detection pipeline using Isaac SDK that processes camera images and publishes detection results with confidence scores.

2. **Depth Estimation Exercise**: Create a depth estimation node that takes stereo camera input and produces depth maps, then use the depth information to enhance object detection results.

3. **Sensor Fusion Exercise**: Implement a sensor fusion node that combines camera-based object detection with LiDAR-based object detection to create more robust perception results.

## Summary

This chapter covered advanced perception techniques using the NVIDIA Isaac platform, including AI-powered object detection and recognition, depth estimation, point cloud processing, and sensor fusion. We explored how GPU acceleration enables real-time processing of complex perception algorithms and how to implement these techniques using Isaac's codelet architecture.

The key takeaways include:
- Isaac provides optimized tools for AI-powered perception using TensorRT
- Depth estimation and point cloud processing benefit significantly from GPU acceleration
- Real-time performance requires careful memory management and pipeline optimization
- Sensor fusion combines multiple data sources for more robust perception
- Isaac's modular architecture enables flexible perception pipeline design

## Cross-references

For foundational concepts about Physical AI, see [Chapter 1: Introduction to Physical AI](../part-01-foundations/chapter-1). For sensor systems, see [Chapter 2: The Robotic Sensorium](../part-01-foundations/chapter-2). For Isaac platform basics, see [Chapter 10: NVIDIA Isaac Platform Overview](../part-04-isaac/chapter-10).