---
title: "Chapter 13: Navigation and Path Planning"
description: "Implementing Nav2 for robot navigation and path planning in Isaac"
sidebar_position: 4
---

# Chapter 13: Navigation and Path Planning

## Learning Objectives

After completing this chapter, you should be able to:

- Understand the Nav2 architecture and its components
- Implement path planning algorithms for bipedal robots
- Configure obstacle avoidance strategies for dynamic environments
- Design dynamic replanning capabilities for changing conditions

## Introduction to Nav2 Architecture

### Overview of Navigation Stack

Navigation is a critical capability for autonomous robots. The Navigation2 (Nav2) stack provides a complete solution for robot navigation, building upon the ROS/ROS 2 ecosystem and integrating with the Isaac platform for enhanced perception and planning.

### Key Components of Nav2

The Nav2 stack consists of several integrated components:

1. **Map Server**: Loads and manages static maps
2. **Local Planner**: Plans short-term trajectories in local space
3. **Global Planner**: Plans long-term paths in global space
4. **Controller**: Executes planned trajectories
5. **Recovery**: Handles navigation failures
6. **Lifecycle Manager**: Manages navigation system lifecycle
7. **Behavior Tree Navigator**: Coordinates navigation tasks

### Nav2 with Isaac Integration

The integration with Isaac provides:

1. **Enhanced Perception**: GPU-accelerated perception for better obstacle detection
2. **Simulation**: High-fidelity simulation environment with Isaac Sim
3. **Deep Learning**: AI-powered navigation behaviors
4. **Hardware Acceleration**: Optimized navigation algorithms for NVIDIA platforms

## Nav2 Architecture and Components

### Main Architecture

The Nav2 architecture follows a client-server model with lifecycle management:

```cpp
// Example of Nav2 server codelet in Isaac
#include "nav2_util/lifecycle_node.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_core/global_planner.hpp"
#include "nav2_core/local_planner.hpp"
#include "nav2_behavior_tree/behavior_tree_engine.hpp"

namespace nav2_isaac {

class Nav2IsaacServer : public nav2_util::LifecycleNode {
public:
  explicit Nav2IsaacServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~Nav2IsaacServer();

protected:
  // Lifecycle methods
  nav2_util::CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  nav2_util::CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
  nav2_util::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
  nav2_util::CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;
  nav2_util::CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override;
  nav2_util::CallbackReturn on_error(const rclcpp_lifecycle::State & state) override;

private:
  // Core navigation components
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> global_costmap_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> local_costmap_;
  std::shared_ptr<nav2_core::GlobalPlanner> global_planner_;
  std::shared_ptr<nav2_core::LocalPlanner> local_planner_;
  std::shared_ptr<nav2_core::Controller> controller_;
  
  // Isaac-specific components
  std::shared_ptr<IsaacPerceptionInterface> perception_interface_;
  std::shared_ptr<IsaacPathOptimizer> path_optimizer_;
  
  // Behavior tree and navigation tasks
  std::shared_ptr<nav2_behavior_tree::BehaviorTreeEngine> bt_engine_;
  std::shared_ptr<nav2_behavior_tree::NavigationTask> navigation_task_;
};

} // namespace nav2_isaac
```

### Costmap Configuration

Costmaps represent the environment with different cost levels for navigation:

```cpp
// Costmap configuration for Nav2 with Isaac
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"

namespace nav2_isaac {

class IsaacCostmapLayer : public nav2_costmap_2d::Layer {
public:
  IsaacCostmapLayer();

  virtual void onInitialize();
  virtual void updateBounds(
    double origin_x, double origin_y, double origin_yaw, 
    double* min_x, double* min_y, double* max_x, double* max_y);
  virtual void updateCosts(
    nav2_costmap_2d::Costmap2D& master_grid, 
    int min_i, int min_j, int max_i, int max_j);

private:
  void addStaticLayer(
    double* min_x, double* min_y, double* max_x, double* max_y);
  void addObstacleLayer(
    double* min_x, double* min_y, double* max_x, double* max_y);
  void addVoxelLayer(
    double* min_x, double* min_y, double* max_x, double* max_y);
  void addInflationLayer(
    double* min_x, double* min_y, double* max_x, double* max_y);

  // Isaac-specific inputs
  std::shared_ptr<IsaacPerceptionData> perception_data_;
  double static_map_resolution_;
  int static_map_width_, static_map_height_;
  std::vector<char> static_map_data_;
};

void IsaacCostmapLayer::updateBounds(
  double origin_x, double origin_y, double origin_yaw,
  double* min_x, double* min_y, double* max_x, double* max_y)
{
  // Get bounds from static map
  addStaticLayer(min_x, min_y, max_x, max_y);
  
  // Add dynamic obstacle bounds from Isaac perception
  if (perception_data_ && perception_data_->has_obstacles) {
    for (const auto& obstacle : perception_data_->obstacles) {
      double ox = obstacle.position.x;
      double oy = obstacle.position.y;
      double radius = obstacle.radius;
      
      *min_x = std::min(*min_x, ox - radius);
      *min_y = std::min(*min_y, oy - radius);
      *max_x = std::max(*max_x, ox + radius);
      *max_y = std::max(*max_y, oy + radius);
    }
  }
  
  // Add inflation bounds
  addInflationLayer(min_x, min_y, max_x, max_y);
}

void IsaacCostmapLayer::updateCosts(
  nav2_costmap_2d::Costmap2D& master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  // Add static map costs
  addStaticCosts(master_grid, min_i, min_j, max_i, max_j);
  
  // Add dynamic obstacles from Isaac perception
  if (perception_data_ && perception_data_->has_obstacles) {
    for (const auto& obstacle : perception_data_->obstacles) {
      addDynamicObstacleCosts(
        master_grid, min_i, min_j, max_i, max_j, 
        obstacle.position, obstacle.radius);
    }
  }
  
  // Add inflation costs
  addInflationCosts(master_grid, min_i, min_j, max_i, max_j);
}

} // namespace nav2_isaac
```

### Behavior Tree Integration

Nav2 uses behavior trees to coordinate complex navigation tasks:

```xml
<!-- Example behavior tree for navigation with Isaac -->
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <Sequence name="NavigateWithRecovery">
      <GoalUpdated/>
      <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
      <IsPathValid path="{path}"/>
      <FollowPath path="{path}" controller_id="FollowPath"/>
      <RecoveryNode number_of_retries="2">
        <ClearEntireCostmap name="ClearLocalCostmap" service_name="local_costmap/clear_entirely_local_costmap"/>
        <ClearEntireCostmap name="ClearGlobalCostmap" service_name="global_costmap/clear_entirely_global_costmap"/>
      </RecoveryNode>
    </Sequence>
  </BehaviorTree>

  <BehaviorTree ID="IsaacPerceptionSequence">
    <Sequence name="IsaacPerception">
      <IsaacGetSemanticMap/>
      <IsaacObjectDetection/>
      <IsaacDynamicObstacleDetection/>
      <IsaacUpdateCostmap/>
    </Sequence>
  </BehaviorTree>
</root>
```

## Path Planning Algorithms for Bipedal Robots

### Challenges of Bipedal Navigation

Bipedal robots face unique navigation challenges:

1. **Stability**: Maintaining balance during movement
2. **Footstep Planning**: Planning where to place feet for stable walking
3. **Dynamic Obstacles**: Dealing with moving obstacles in crowded environments
4. **Terrain Adaptation**: Handling uneven surfaces and stairs
5. **Energy Efficiency**: Optimizing for battery life

### Footstep Planning Integration

```cpp
// Footstep planning for bipedal robots in Nav2
#include "nav2_core/global_planner.hpp"
#include "footstep_planner/footstep_planner.hpp"

namespace nav2_isaac {

class BipedalPlanner : public nav2_core::GlobalPlanner {
public:
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  nav_msgs::msg::Path createPath(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal) override;

private:
  std::shared_ptr<footstep_planner::FootstepPlanner> footstep_planner_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  std::vector<footstep_planner::Footstep> footsteps_;
  
  // Bipedal-specific parameters
  double foot_separation_;
  double step_length_limit_;
  double step_height_limit_;
  bool enable_terrain_analysis_;
};

nav_msgs::msg::Path BipedalPlanner::createPath(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  nav_msgs::msg::Path path;
  
  // Plan footsteps between start and goal
  footstep_planner::FootstepPlan plan;
  
  // Set up planning constraints
  footstep_planner::PlanningParams params;
  params.step_length_limit = step_length_limit_;
  params.step_height_limit = step_height_limit_;
  params.foot_separation = foot_separation_;
  params.enable_terrain_analysis = enable_terrain_analysis_;
  
  // Get costmap data for planning
  auto costmap = costmap_ros_->getCostmap();
  
  // Plan footsteps
  bool success = footstep_planner_->planFootsteps(
    start.pose.position, goal.pose.position, 
    costmap, params, plan);
  
  if (success) {
    // Convert footsteps to navigation path
    for (const auto& step : plan.footsteps) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header.frame_id = "map";
      pose.pose.position.x = step.x;
      pose.pose.position.y = step.y;
      pose.pose.position.z = step.z;
      
      // Calculate orientation based on path direction
      if (&step != &plan.footsteps[0]) {  // Not first step
        auto prev_step = *(std::prev(&step));
        double angle = atan2(step.y - prev_step.y, step.x - prev_step.x);
        pose.pose.orientation = tf2::toMsg(
          tf2::Quaternion(tf2::Vector3(0, 0, 1), angle));
      }
      
      path.poses.push_back(pose);
    }
  }
  
  return path;
}

} // namespace nav2_isaac
```

### Hybrid A* for Bipedal Robots

```cpp
// Hybrid A* implementation for bipedal navigation
#include "nav2_core/global_planner.hpp"
#include <queue>

namespace nav2_isaac {

struct HybridState {
  double x, y, theta;
  int g_cost;  // Path cost
  int h_cost;  // Heuristic cost
  
  bool operator>(const HybridState& other) const {
    return (g_cost + h_cost) > (other.g_cost + other.h_cost);
  }
};

class HybridAStarBipedal : public nav2_core::GlobalPlanner {
public:
  nav_msgs::msg::Path createPath(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal) override;

private:
  double calculateHeuristic(const HybridState& state, 
                           const geometry_msgs::msg::Point& goal);
  std::vector<HybridState> getSuccessors(const HybridState& state,
                                        const nav2_costmap_2d::Costmap2D* costmap);
  bool isStateValid(const HybridState& state,
                   const nav2_costmap_2d::Costmap2D* costmap);
  nav_msgs::msg::Path extractPath(const std::map<int, HybridState>& came_from,
                                 const HybridState& goal_state);
};

double HybridAStarBipedal::calculateHeuristic(
  const HybridState& state, 
  const geometry_msgs::msg::Point& goal)
{
  // Euclidean distance
  double dist = sqrt(pow(state.x - goal.x, 2) + pow(state.y - goal.y, 2));
  
  // Add orientation penalty to encourage straighter paths
  double orientation_penalty = 0.1 * abs(state.theta);
  
  return dist + orientation_penalty;
}

std::vector<HybridState> HybridAStarBipedal::getSuccessors(
  const HybridState& state,
  const nav2_costmap_2d::Costmap2D* costmap)
{
  std::vector<HybridState> successors;
  
  // Define possible actions for bipedal robot
  std::vector<std::pair<double, double>> actions = {
    {0.1, 0.0},      // Move forward
    {0.05, 0.2},     // Move forward and turn right
    {0.05, -0.2},    // Move forward and turn left
    {0.0, 0.15},     // Turn in place right
    {0.0, -0.15}     // Turn in place left
  };
  
  for (const auto& action : actions) {
    HybridState new_state = state;
    
    // Apply action
    double dt = 0.1;  // Time step
    new_state.x += action.first * cos(state.theta) * dt;
    new_state.y += action.first * sin(state.theta) * dt;
    new_state.theta += action.second * dt;
    
    // Add cost based on movement
    new_state.g_cost = state.g_cost + 
                      static_cast<int>(action.first * 10); // Convert to int cost
    
    if (isStateValid(new_state, costmap)) {
      successors.push_back(new_state);
    }
  }
  
  return successors;
}

bool HybridAStarBipedal::isStateValid(
  const HybridState& state,
  const nav2_costmap_2d::Costmap2D* costmap)
{
  // Check if position is in costmap bounds
  unsigned int mx, my;
  if (!costmap->worldToMap(state.x, state.y, mx, my)) {
    return false;  // Outside costmap bounds
  }
  
  // Check cost - don't allow paths through obstacles
  unsigned char cost = costmap->getCost(mx, my);
  if (cost >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
    return false;
  }
  
  // For bipedal robots, also check if terrain is traversable
  // (e.g., avoid stairs, steep slopes)
  if (enable_terrain_analysis_) {
    if (isTraversableTerrain(state, costmap)) {
      return false;
    }
  }
  
  return true;
}

nav_msgs::msg::Path HybridAStarBipedal::createPath(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  nav_msgs::msg::Path path;
  
  // Initialize start state
  HybridState start_state;
  start_state.x = start.pose.position.x;
  start_state.y = start.pose.position.y;
  start_state.theta = tf2::getYaw(start.pose.orientation);
  start_state.g_cost = 0;
  start_state.h_cost = static_cast<int>(
    calculateHeuristic(start_state, goal.pose.position) * 10);
  
  // Priority queue for A* search
  std::priority_queue<HybridState, std::vector<HybridState>, 
                     std::greater<HybridState>> open_set;
  open_set.push(start_state);
  
  // Keep track of visited states and parent relationships
  std::set<std::string> closed_set;
  std::map<std::string, HybridState> came_from;
  
  auto costmap = costmap_ros_->getCostmap();
  
  while (!open_set.empty()) {
    HybridState current = open_set.top();
    open_set.pop();
    
    // Create unique key for state
    std::string state_key = std::to_string(static_cast<int>(current.x * 10)) + "," +
                           std::to_string(static_cast<int>(current.y * 10)) + "," +
                           std::to_string(static_cast<int>(current.theta * 10));
    
    // Check if we've reached the goal
    double dist_to_goal = sqrt(pow(current.x - goal.pose.position.x, 2) + 
                              pow(current.y - goal.pose.position.y, 2));
    if (dist_to_goal < 0.5) {  // Goal tolerance
      // Reconstruct path
      path = extractPath(came_from, current);
      break;
    }
    
    // Skip if already processed
    if (closed_set.count(state_key)) {
      continue;
    }
    
    closed_set.insert(state_key);
    
    // Get possible next states
    auto successors = getSuccessors(current, costmap);
    
    for (const auto& successor : successors) {
      std::string succ_key = std::to_string(static_cast<int>(successor.x * 10)) + "," +
                            std::to_string(static_cast<int>(successor.y * 10)) + "," +
                            std::to_string(static_cast<int>(successor.theta * 10));
      
      if (!closed_set.count(succ_key)) {
        successor.h_cost = static_cast<int>(
          calculateHeuristic(successor, goal.pose.position) * 10);
        
        open_set.push(successor);
        came_from[succ_key] = current;
      }
    }
  }
  
  return path;
}

nav_msgs::msg::Path HybridAStarBipedal::extractPath(
  const std::map<std::string, HybridState>& came_from,
  const HybridState& goal_state)
{
  nav_msgs::msg::Path path;
  path.header.frame_id = "map";
  
  std::vector<HybridState> path_states;
  HybridState current = goal_state;
  
  // Reconstruct path by following parent pointers
  while (true) {
    path_states.push_back(current);
    
    std::string state_key = std::to_string(static_cast<int>(current.x * 10)) + "," +
                           std::to_string(static_cast<int>(current.y * 10)) + "," +
                           std::to_string(static_cast<int>(current.theta * 10));
    
    auto it = came_from.find(state_key);
    if (it == came_from.end()) {
      break;  // Reached start state
    }
    
    current = it->second;
  }
  
  // Add poses to path (in reverse order since we built it backwards)
  for (auto it = path_states.rbegin(); it != path_states.rend(); ++it) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = "map";
    pose.pose.position.x = it->x;
    pose.pose.position.y = it->y;
    pose.pose.position.z = 0.0;  // Assuming 2D navigation
    
    tf2::Quaternion quat;
    quat.setRPY(0.0, 0.0, it->theta);
    pose.pose.orientation = tf2::toMsg(quat);
    
    path.poses.push_back(pose);
  }
  
  return path;
}

} // namespace nav2_isaac
```

## Obstacle Avoidance Strategies

### Dynamic Obstacle Detection with Isaac

```cpp
// Dynamic obstacle detection and avoidance
#include "nav2_core/controller.hpp"
#include "isaac_perception/obstacle_detector.hpp"

namespace nav2_isaac {

class DynamicObstacleController : public nav2_core::Controller {
public:
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  geometry_msgs::msg::Twist computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped& pose,
    const geometry_msgs::msg::Twist& velocity,
    nav2_core::GoalChecker * goal_checker) override;

private:
  std::shared_ptr<IsaacObstacleDetector> obstacle_detector_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  
  // Dynamic obstacle tracking
  std::vector<ObstacleTrack> tracked_obstacles_;
  rclcpp::Time last_detection_time_;
  
  // Avoidance parameters
  double safety_distance_;
  double prediction_horizon_;
  double max_angular_vel_;
  double obstacle_velocity_threshold_;
};

geometry_msgs::msg::Twist DynamicObstacleController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped& pose,
  const geometry_msgs::msg::Twist& velocity,
  nav2_core::GoalChecker * goal_checker)
{
  geometry_msgs::msg::Twist cmd_vel;
  
  // Get dynamic obstacles from Isaac perception
  IsaacPerceptionData perception_data = obstacle_detector_->getPerceptionData();
  
  // Update obstacle tracking
  updateObstacleTracks(perception_data);
  
  // Get preferred velocity toward goal
  geometry_msgs::msg::Twist preferred_vel = computePreferredVelocity(pose);
  
  // Check for conflicts with dynamic obstacles
  geometry_msgs::msg::Twist safe_vel = preferred_vel;
  
  for (const auto& obstacle : tracked_obstacles_) {
    if (obstacle.is_conflicting(pose.pose.position, velocity)) {
      // Compute avoidance velocity
      geometry_msgs::msg::Twist avoidance_vel = 
        computeAvoidanceVelocity(pose.pose.position, velocity, obstacle);
      
      // Select best velocity (minimize deviation from preferred while avoiding collision)
      safe_vel = selectSafeVelocity(preferred_vel, avoidance_vel);
    }
  }
  
  return safe_vel;
}

void DynamicObstacleController::updateObstacleTracks(
  const IsaacPerceptionData& data)
{
  rclcpp::Time current_time = this->now();
  
  // Match new detections to existing tracks
  std::vector<bool> matched(tracked_obstacles_.size(), false);
  
  for (const auto& detection : data.obstacles) {
    int best_match = -1;
    double min_distance = std::numeric_limits<double>::max();
    
    // Find best matching track
    for (size_t i = 0; i < tracked_obstacles_.size(); ++i) {
      if (matched[i]) continue;
      
      double dist = distance(detection.position, tracked_obstacles_[i].predicted_position);
      if (dist < min_distance && dist < 0.5) {  // 50cm max association distance
        min_distance = dist;
        best_match = static_cast<int>(i);
      }
    }
    
    if (best_match >= 0) {
      // Update existing track
      matched[best_match] = true;
      tracked_obstacles_[best_match].update(detection, current_time);
    } else {
      // Create new track
      tracked_obstacles_.emplace_back(detection, current_time);
    }
  }
  
  // Remove unmatched old tracks
  for (size_t i = 0; i < tracked_obstacles_.size(); ) {
    if (!matched[i] && 
        (current_time - tracked_obstacles_[i].last_seen_time).seconds() > 2.0) {
      tracked_obstacles_.erase(tracked_obstacles_.begin() + i);
    } else {
      ++i;
    }
  }
}

geometry_msgs::msg::Twist DynamicObstacleController::computeAvoidanceVelocity(
  const geometry_msgs::msg::Point& robot_pos,
  const geometry_msgs::msg::Twist& robot_vel,
  const ObstacleTrack& obstacle)
{
  geometry_msgs::msg::Twist avoidance_cmd;
  
  // Predict obstacle future position
  auto predicted_obstacle_pos = obstacle.predictPosition(prediction_horizon_);
  
  // Calculate avoidance direction (perpendicular to robot-obstacle vector)
  double dx = predicted_obstacle_pos.x - robot_pos.x;
  double dy = predicted_obstacle_pos.y - robot_pos.y;
  double dist = sqrt(dx*dx + dy*dy);
  
  if (dist < safety_distance_) {
    // Compute perpendicular vector for avoidance
    double perp_x = -dy / dist;  // Perpendicular direction
    double perp_y = dx / dist;
    
    // Normalize and scale based on distance
    double avoidance_strength = std::max(0.0, (safety_distance_ - dist) / safety_distance_);
    
    avoidance_cmd.linear.x = perp_x * avoidance_strength;
    avoidance_cmd.linear.y = perp_y * avoidance_strength;
    
    // Add angular component for turning away
    double desired_theta = atan2(perp_y, perp_x);
    avoidance_cmd.angular.z = std::min(max_angular_vel_, 
                                      std::max(-max_angular_vel_, desired_theta * 2.0));
  }
  
  return avoidance_cmd;
}

} // namespace nav2_isaac
```

### Vector Field Histogram for Bipedal Robots

```cpp
// Vector Field Histogram implementation adapted for bipedal robots
#include "nav2_core/controller.hpp"

namespace nav2_isaac {

class BipedalVFHController : public nav2_core::Controller {
public:
  geometry_msgs::msg::Twist computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped& pose,
    const geometry_msgs::msg::Twist& velocity,
    nav2_core::GoalChecker * goal_checker) override;

private:
  void updatePolarHistogram(const nav2_costmap_2d::Costmap2D& costmap,
                           const geometry_msgs::msg::Point& robot_pos);
  std::vector<double> computeTargetDirection(const geometry_msgs::msg::Point& goal_pos);
  std::vector<double> selectSafeDirection(const std::vector<double>& target_directions);
  geometry_msgs::msg::Twist directionToVelocity(const std::vector<double>& direction);

  // VFH parameters
  double sector_angle_;           // Angle of each sector in polar histogram (radians)
  double max_sensor_range_;       // Maximum range for obstacle detection
  double min_distance_threshold_; // Minimum distance that triggers obstacle response
  double safety_factor_;          // Factor for determining safety threshold
  int window_size_;              // Size of smoothing window for histogram
  
  std::vector<double> polar_histogram_;  // Polar histogram of obstacles
};

void BipedalVFHController::updatePolarHistogram(
  const nav2_costmap_2d::Costmap2D& costmap,
  const geometry_msgs::msg::Point& robot_pos)
{
  // Initialize histogram
  int num_sectors = static_cast<int>(2 * M_PI / sector_angle_);
  polar_histogram_.resize(num_sectors, 0.0);
  
  // Iterate through costmap cells in circular area around robot
  double max_range_sq = max_sensor_range_ * max_sensor_range_;
  
  // Determine area to scan based on robot position and sensor range
  int robot_mx, robot_my;
  if (costmap.worldToMap(robot_pos.x, robot_pos.y, robot_mx, robot_my)) {
    int range_cells = static_cast<int>(max_sensor_range_ / costmap.getResolution());
    
    for (int dx = -range_cells; dx <= range_cells; dx++) {
      for (int dy = -range_cells; dy <= range_cells; dy++) {
        int mx = robot_mx + dx;
        int my = robot_my + dy;
        
        if (mx >= 0 && mx < costmap.getSizeInCellsX() && 
            my >= 0 && my < costmap.getSizeInCellsY()) {
          
          // Calculate distance from robot
          double wx, wy;
          costmap.mapToWorld(mx, my, wx, wy);
          double dist_sq = (wx - robot_pos.x) * (wx - robot_pos.x) + 
                          (wy - robot_pos.y) * (wy - robot_pos.y);
          
          if (dist_sq <= max_range_sq) {
            // Calculate angle from robot heading
            double angle = atan2(wy - robot_pos.y, wx - robot_pos.x);
            // Normalize angle to [0, 2*PI)
            if (angle < 0) angle += 2 * M_PI;
            
            // Determine sector
            int sector = static_cast<int>(angle / sector_angle_);
            if (sector >= num_sectors) sector = 0;  // Handle wrap-around
            
            // Get cost and add to histogram
            unsigned char cost = costmap.getCost(mx, my);
            if (cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {
              // For bipedal robots, consider cost based on traversability
              double distance_factor = 1.0 - sqrt(dist_sq) / max_sensor_range_;
              polar_histogram_[sector] += cost * distance_factor;
            }
          }
        }
      }
    }
    
    // Apply smoothing to reduce noise
    std::vector<double> smoothed = polar_histogram_;
    for (int i = 0; i < num_sectors; i++) {
      double sum = 0;
      int count = 0;
      
      for (int j = -window_size_/2; j <= window_size_/2; j++) {
        int idx = (i + j + num_sectors) % num_sectors;
        sum += polar_histogram_[idx];
        count++;
      }
      
      smoothed[i] = sum / count;
    }
    
    polar_histogram_ = smoothed;
  }
}

std::vector<double> BipedalVFHController::computeTargetDirection(
  const geometry_msgs::msg::Point& goal_pos)
{
  std::vector<double> target_directions;
  
  // Calculate direction to goal
  double dx = goal_pos.x - robot_pos_.x;
  double dy = goal_pos.y - robot_pos_.y;
  double target_angle = atan2(dy, dx);
  
  // Normalize to [0, 2*PI)
  if (target_angle < 0) target_angle += 2 * M_PI;
  
  // Find corresponding sector
  int target_sector = static_cast<int>(target_angle / sector_angle_);
  if (target_sector >= polar_histogram_.size()) target_sector = 0;
  
  // Consider adjacent sectors as potential directions
  for (int offset = -2; offset <= 2; offset++) {
    int sector = (target_sector + offset + polar_histogram_.size()) % polar_histogram_.size();
    if (polar_histogram_[sector] < min_distance_threshold_) {
      // This sector is relatively safe
      double angle = sector * sector_angle_;
      target_directions.push_back(angle);
    }
  }
  
  return target_directions;
}

geometry_msgs::msg::Twist BipedalVFHController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped& pose,
  const geometry_msgs::msg::Twist& velocity,
  nav2_core::GoalChecker * goal_checker)
{
  // Update robot position
  robot_pos_ = pose.pose.position;
  
  // Update polar histogram based on current costmap
  auto costmap = costmap_ros_->getCostmap();
  updatePolarHistogram(*costmap, robot_pos_);
  
  // Get target directions
  std::vector<double> target_dirs = computeTargetDirection(goal_pos_);
  
  // Select the best safe direction
  std::vector<double> best_dir = selectSafeDirection(target_dirs);
  
  // Convert direction to velocity command
  geometry_msgs::msg::Twist cmd_vel = directionToVelocity(best_dir);
  
  return cmd_vel;
}

std::vector<double> BipedalVFHController::selectSafeDirection(
  const std::vector<double>& target_directions)
{
  std::vector<double> best_direction;
  
  if (target_directions.empty()) {
    // No safe directions found to goal, try to move away from obstacles
    // Find the direction with minimum obstacle density
    int best_sector = 0;
    double min_cost = polar_histogram_[0];
    
    for (size_t i = 1; i < polar_histogram_.size(); i++) {
      if (polar_histogram_[i] < min_cost) {
        min_cost = polar_histogram_[i];
        best_sector = static_cast<int>(i);
      }
    }
    
    best_direction.push_back(best_sector * sector_angle_);
  } else {
    // Choose direction closest to goal direction
    best_direction = target_directions[0];
  }
  
  return best_direction;
}

geometry_msgs::msg::Twist BipedalVFHController::directionToVelocity(
  const std::vector<double>& direction)
{
  geometry_msgs::msg::Twist cmd_vel;
  
  if (!direction.empty()) {
    double target_angle = direction[0];
    
    // Calculate robot's current heading
    double robot_angle = tf2::getYaw(robot_pos_.orientation);
    
    // Calculate angle difference
    double angle_diff = target_angle - robot_angle;
    
    // Normalize angle difference to [-PI, PI]
    while (angle_diff > M_PI) angle_diff -= 2 * M_PI;
    while (angle_diff < -M_PI) angle_diff += 2 * M_PI;
    
    // Set velocities based on angle difference
    cmd_vel.linear.x = 0.5 * cos(angle_diff);  // Forward speed based on alignment
    cmd_vel.angular.z = 2.0 * angle_diff;       // Angular correction
    
    // Limit velocities for bipedal stability
    cmd_vel.linear.x = std::max(-max_linear_speed_, 
                               std::min(max_linear_speed_, cmd_vel.linear.x));
    cmd_vel.angular.z = std::max(-max_angular_speed_, 
                                std::min(max_angular_speed_, cmd_vel.angular.z));
  }
  
  return cmd_vel;
}

} // namespace nav2_isaac
```

## Dynamic Replanning Capabilities

### Adaptive Path Replanning

```cpp
// Dynamic replanning for changing environments
#include "nav2_core/global_planner.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"

namespace nav2_isaac {

class DynamicReplanPlanner : public nav2_core::GlobalPlanner {
public:
  nav_msgs::msg::Path createPath(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal) override;

  void setReplanCallback(std::function<void()> callback) {
    replan_callback_ = callback;
  }

private:
  struct PathSegment {
    nav_msgs::msg::Path path;
    double validity_time;  // When this segment becomes invalid
    int start_index;       // Index in the global path
  };

  std::vector<PathSegment> path_segments_;
  rclcpp::Time last_replan_time_;
  std::function<void()> replan_callback_;
  
  // Replanning parameters
  double replan_frequency_;
  double min_path_validity_duration_;
  double costmap_update_threshold_;
  
  bool shouldReplan();
  void invalidateAffectedSegments(const std::vector<Obstacle>& new_obstacles);
  nav_msgs::msg::Path generateNewPathSegment(int start_index, 
                                           const geometry_msgs::msg::PoseStamped& goal);
};

bool DynamicReplanPlanner::shouldReplan()
{
  auto costmap = costmap_ros_->getCostmap();
  
  // Check if enough time has passed since last replan
  auto now = this->now();
  if ((now - last_replan_time_).seconds() < 1.0/replan_frequency_) {
    return false;
  }
  
  // Check if any path segments are about to become invalid
  for (const auto& segment : path_segments_) {
    double time_to_invalid = segment.validity_time - now.seconds();
    if (time_to_invalid < min_path_validity_duration_) {
      return true;
    }
  }
  
  // Check for significant changes in costmap that might affect path
  if (hasSignificantCostmapChanges()) {
    return true;
  }
  
  return false;
}

void DynamicReplanPlanner::invalidateAffectedSegments(
  const std::vector<Obstacle>& new_obstacles)
{
  // Mark segments that would pass through new obstacles as invalid
  for (auto& segment : path_segments_) {
    for (const auto& obstacle : new_obstacles) {
      // Check if obstacle is in segment path
      for (size_t i = segment.start_index; i < std::min(segment.start_index + segment.path.poses.size(), 
                                                     path_segments_.back().start_index + path_segments_.back().path.poses.size()); i++) {
        if (distance(obstacle.position, segment.path.poses[i].pose.position) < obstacle.radius) {
          // Mark this segment for replanning
          segment.validity_time = 0;  // Immediate replanning needed
        }
      }
    }
  }
}

nav_msgs::msg::Path DynamicReplanPlanner::createPath(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  nav_msgs::msg::Path complete_path;
  
  if (shouldReplan()) {
    // Perform complete replanning
    auto new_path = planCompletePath(start, goal);
    
    // Update path segments
    path_segments_.clear();
    PathSegment segment;
    segment.path = new_path;
    segment.start_index = 0;
    segment.validity_time = this->now().seconds() + 10.0; // Valid for 10 seconds initially
    path_segments_.push_back(segment);
    
    last_replan_time_ = this->now();
    
    if (replan_callback_) {
      replan_callback_();
    }
    
    return new_path;
  }
  
  // Use existing path if still valid
  for (const auto& segment : path_segments_) {
    complete_path.poses.insert(complete_path.poses.end(),
                              segment.path.poses.begin(),
                              segment.path.poses.end());
  }
  
  return complete_path;
}

nav_msgs::msg::Path DynamicReplanPlanner::planCompletePath(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  // Use the appropriate planner based on environment type
  if (isStaticEnvironment()) {
    // Use A* for static environments
    return planWithAStar(start, goal);
  } else {
    // Use D* Lite or other dynamic planning algorithm
    return planWithDynamicPlanner(start, goal);
  }
}

} // namespace nav2_isaac
```

### Costmap-Based Replanning Trigger

```cpp
// Replanning triggered by costmap changes
#include "nav2_costmap_2d/costmap_layer.hpp"

namespace nav2_isaac {

class ReplanningMonitor : public nav2_costmap_2d::Layer {
public:
  ReplanningMonitor(std::function<void()> replan_callback) 
    : replan_callback_(replan_callback) {}

  void onInitialize() override;
  void updateBounds(double origin_x, double origin_y, double origin_yaw,
                   double* min_x, double* min_y, double* max_x, double* max_y) override;
  void updateCosts(nav2_costmap_2d::Costmap2D& master_grid,
                  int min_i, int min_j, int max_i, int max_j) override;

private:
  void checkForChanges(int min_i, int min_j, int max_i, int max_j);
  bool detectSignificantChanges(int min_i, int min_j, int max_i, int max_j);
  
  std::function<void()> replan_callback_;
  std::vector<unsigned char> previous_costmap_;
  double change_threshold_;
  rclcpp::Time last_check_time_;
};

void ReplanningMonitor::updateCosts(
  nav2_costmap_2d::Costmap2D& master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  // Check for significant changes in costmap
  if (detectSignificantChanges(min_i, min_j, max_i, max_j)) {
    RCLCPP_INFO(node_->get_logger(), "Significant environmental change detected, triggering replan");
    
    // Trigger replanning
    if (replan_callback_) {
      replan_callback_();
    }
  }
  
  // Store current costmap for next comparison
  previous_costmap_.resize(master_grid.getSizeInCellsX() * master_grid.getSizeInCellsY());
  for (int j = min_j; j < max_j; ++j) {
    for (int i = min_i; i < max_i; ++i) {
      int idx = master_grid.getIndex(i, j);
      previous_costmap_[idx] = master_grid.getCost(i, j);
    }
  }
}

bool ReplanningMonitor::detectSignificantChanges(
  int min_i, int min_j, int max_i, int max_j)
{
  auto costmap = layered_costmap_->getCostmap();
  
  if (previous_costmap_.empty()) {
    // First time, no comparison possible
    return false;
  }
  
  int significant_changes = 0;
  int total_cells = 0;
  
  for (int j = min_j; j < max_j; ++j) {
    for (int i = min_i; i < max_i; ++i) {
      int idx = costmap->getIndex(i, j);
      unsigned char current_cost = costmap->getCost(i, j);
      
      if (idx < static_cast<int>(previous_costmap_.size())) {
        unsigned char previous_cost = previous_costmap_[idx];
        
        // Check for significant cost change
        if (abs(static_cast<int>(current_cost) - static_cast<int>(previous_cost)) > 50) {
          significant_changes++;
        }
        
        total_cells++;
      }
    }
  }
  
  // Trigger replan if significant percentage of cells have changed
  return (total_cells > 0) && (static_cast<double>(significant_changes) / total_cells > 0.1); // 10% threshold
}

} // namespace nav2_isaac
```

## Practical Implementation Example

Here's a complete example combining navigation components:

```cpp
// Complete navigation system integration example
#include "nav2_util/lifecycle_node.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_behavior_tree/behavior_tree_engine.hpp"

namespace nav2_isaac {

class IntegratedNavigationSystem : public nav2_util::LifecycleNode {
public:
  explicit IntegratedNavigationSystem(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~IntegratedNavigationSystem();

protected:
  nav2_util::CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  nav2_util::CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
  nav2_util::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
  nav2_util::CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;

private:
  // Core navigation components
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> global_costmap_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> local_costmap_;
  std::shared_ptr<nav2_core::GlobalPlanner> global_planner_;
  std::shared_ptr<nav2_core::LocalPlanner> local_planner_;
  std::shared_ptr<nav2_core::Controller> controller_;
  
  // Isaac-specific components
  std::shared_ptr<IsaacPerceptionInterface> perception_interface_;
  std::shared_ptr<IsaacCostmapLayer> isaac_costmap_layer_;
  std::shared_ptr<ReplanningMonitor> replan_monitor_;
  
  // Bipedal-specific components
  std::shared_ptr<BipedalPlanner> bipedal_planner_;
  std::shared_ptr<BipedalVFHController> bipedal_controller_;
  
  // Behavior tree and navigation tasks
  std::shared_ptr<nav2_behavior_tree::BehaviorTreeEngine> bt_engine_;
  std::shared_ptr<nav2_behavior_tree::NavigationTask> navigation_task_;
  
  // Action servers
  rclcpp_action::Server<nav2_msgs::action::NavigateToPose>::SharedPtr navigate_action_server_;
  
  // Callback functions
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp::TimerBase::SharedPtr server_control_timer_;
  
  // Navigation parameters
  std::string global_frame_;
  std::string robot_base_frame_;
  double transform_tolerance_;
  
  // Replanning management
  bool is_replanning_needed_;
  rclcpp::Time last_global_plan_time_;
  std::mutex navigation_mutex_;
  
  // Action server callbacks
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const nav2_msgs::action::NavigateToPose::Goal> goal);
    
  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<nav2_msgs::action::NavigateToPose>> goal_handle);
    
  void handle_accepted(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<nav2_msgs::action::NavigateToPose>> goal_handle);
    
  // Navigation execution
  void executeNavigation(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<nav2_msgs::action::NavigateToPose>> goal_handle);
    
  void triggerReplanning();
};

IntegratedNavigationSystem::IntegratedNavigationSystem(const rclcpp::NodeOptions & options)
  : nav2_util::LifecycleNode("integrated_navigation_system", options)
{
  // Initialize parameters
  this->declare_parameter("global_frame", "map");
  this->declare_parameter("robot_base_frame", "base_link");
  this->declare_parameter("transform_tolerance", 0.1);
  
  // Get parameters
  this->get_parameter("global_frame", global_frame_);
  this->get_parameter("robot_base_frame", robot_base_frame_);
  this->get_parameter("transform_tolerance", transform_tolerance_);
}

nav2_util::CallbackReturn IntegratedNavigationSystem::on_configure(const rclcpp_lifecycle::State & state)
{
  // Create costmaps
  global_costmap_ = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "global_costmap", 
    std::shared_ptr<rclcpp::Node>(this, [](auto){}), 
    std::string{get_name()}, 
    global_frame_);
    
  local_costmap_ = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "local_costmap", 
    std::shared_ptr<rclcpp::Node>(this, [](auto){}), 
    std::string{get_name()}, 
    global_frame_);
  
  // Initialize Isaac perception interface
  perception_interface_ = std::make_shared<IsaacPerceptionInterface>();
  perception_interface_->initialize();
  
  // Create Isaac costmap layer
  isaac_costmap_layer_ = std::make_shared<IsaacCostmapLayer>();
  global_costmap_->getLayeredCostmap()->addPlugin(isaac_costmap_layer_);
  local_costmap_->getLayeredCostmap()->addPlugin(isaac_costmap_layer_);
  
  // Initialize replanning monitor
  replan_monitor_ = std::make_shared<ReplanningMonitor>(
    std::bind(&IntegratedNavigationSystem::triggerReplanning, this));
  global_costmap_->getLayeredCostmap()->addPlugin(replan_monitor_);
  
  // Initialize bipedal-specific components
  bipedal_planner_ = std::make_shared<BipedalPlanner>();
  bipedal_controller_ = std::make_shared<BipedalVFHController>();
  
  // Initialize behavior tree engine
  bt_engine_ = std::make_shared<nav2_behavior_tree::BehaviorTreeEngine>();
  
  // Create action server
  callback_group_ = this->create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive, false);
    
  navigate_action_server_ = rclcpp_action::create_server<nav2_msgs::action::NavigateToPose>(
    this->get_node_base_interface(),
    this->get_node_clock_interface(),
    this->get_node_logging_interface(),
    this->get_node_waitables_interface(),
    "navigate_to_pose",
    std::bind(&IntegratedNavigationSystem::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
    std::bind(&IntegratedNavigationSystem::handle_cancel, this, std::placeholders::_1),
    std::bind(&IntegratedNavigationSystem::handle_accepted, this, std::placeholders::_1),
    rcl_action_server_get_default_options(),
    callback_group_);
  
  RCLCPP_INFO(get_logger(), "Integrated Navigation System configured");
  return nav2_util::CallbackReturn::SUCCESS;
}

void IntegratedNavigationSystem::executeNavigation(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<nav2_msgs::action::NavigateToPose>> goal_handle)
{
  auto goal = goal_handle->get_goal();
  
  // Get start pose
  geometry_msgs::msg::PoseStamped start;
  getRobotPose(start);
  
  // Plan path
  nav_msgs::msg::Path path;
  try {
    path = bipedal_planner_->createPath(start, goal->pose);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "Failed to create path: %s", e.what());
    goal_handle->terminate_goals();
    return;
  }
  
  // Execute navigation
  geometry_msgs::msg::Twist cmd_vel;
  auto result = std::make_shared<nav2_msgs::action::NavigateToPose::Result>();
  
  while (rclcpp::ok() && !goal_handle->is_canceling()) {
    // Check if we need to replan
    if (is_replanning_needed_) {
      RCLCPP_INFO(get_logger(), "Replanning path due to environmental changes");
      path = bipedal_planner_->createPath(start, goal->pose);
      is_replanning_needed_ = false;
    }
    
    // Get current robot pose
    geometry_msgs::msg::PoseStamped current_pose;
    getRobotPose(current_pose);
    
    // Calculate velocity command
    cmd_vel = bipedal_controller_->computeVelocityCommands(current_pose, cmd_vel, nullptr);
    
    // Send velocity command to robot
    publishVelocity(cmd_vel);
    
    // Check if goal is reached
    if (isGoalReached(current_pose, goal->pose)) {
      RCLCPP_INFO(get_logger(), "Goal reached successfully");
      goal_handle->succeed(result);
      break;
    }
    
    // Sleep briefly to allow perception updates
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  
  if (goal_handle->is_canceling()) {
    goal_handle->canceled(result);
  }
}

void IntegratedNavigationSystem::triggerReplanning()
{
  std::lock_guard<std::mutex> lock(navigation_mutex_);
  is_replanning_needed_ = true;
  RCLCPP_INFO(get_logger(), "Replanning triggered by environmental changes");
}

} // namespace nav2_isaac
```

## Exercises

1. **Navigation Implementation Exercise**: Implement a complete navigation system using Nav2 that includes costmap configuration, local and global planners, and a controller for a simple mobile robot.

2. **Bipedal Navigation Exercise**: Modify the navigation system to work with a simulated bipedal robot, taking into account footstep planning and balance constraints.

3. **Dynamic Replanning Exercise**: Implement dynamic replanning that responds to moving obstacles detected by Isaac perception, and evaluate how well the navigation system adapts to changing environments.

## Summary

This chapter covered navigation and path planning for robotics using the Nav2 framework integrated with Isaac. We explored the Nav2 architecture, implemented path planning algorithms specifically designed for bipedal robots, developed obstacle avoidance strategies using Isaac's perception capabilities, and created dynamic replanning systems that respond to changing environments.

The key takeaways include:
- Nav2 provides a complete navigation framework with modular components
- Bipedal robots require specialized planning considering balance and footstep constraints
- Isaac's perception capabilities enhance navigation with real-time obstacle detection
- Dynamic replanning is crucial for adapting to environmental changes
- Behavior trees effectively coordinate complex navigation tasks

## Cross-references

For foundational concepts about Physical AI, see [Chapter 1: Introduction to Physical AI](../part-01-foundations/chapter-1). For Isaac platform basics, see [Chapter 10: NVIDIA Isaac Platform Overview](../part-04-isaac/chapter-10). For perception systems that support navigation, see [Chapter 11: Advanced Perception with Isaac](../part-04-isaac/chapter-11). For RL-based navigation approaches, see [Chapter 12: Reinforcement Learning for Robot Control](../part-04-isaac/chapter-12).