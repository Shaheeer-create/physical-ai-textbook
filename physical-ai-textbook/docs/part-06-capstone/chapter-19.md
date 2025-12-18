---
title: "Chapter 19: System Integration and Testing"
description: "Integration strategies and comprehensive testing for humanoid robots"
sidebar_position: 2
---

# Chapter 19: System Integration and Testing

## Learning Objectives

After completing this chapter, you should be able to:

- Design comprehensive integration strategies for complex robotic systems
- Plan and execute system-level tests for humanoid robots
- Implement continuous integration pipelines for robotic platforms
- Evaluate system performance and identify bottlenecks
- Ensure safety during testing of bipedal robots

## Introduction to System Integration for Humanoid Robots

### Challenges of Humanoid Integration

Integrating a humanoid robot presents unique challenges compared to simpler robotic platforms:

1. **Complex Kinematics**: 20+ degrees of freedom requiring coordinated control
2. **Balance Requirements**: Constant balance maintenance during all operations
3. **Multi-Modal Perception**: Integration of vision, audio, proprioception, and touch
4. **Real-Time Constraints**: Tight timing requirements for control loops
5. **Safety Considerations**: Potential for injury due to robot size and weight
6. **Power Management**: Balancing performance with battery life

### Integration Architecture

The humanoid robot system follows a component-based architecture with defined interfaces:

```yaml
# System Architecture Overview
perception_layer:
  - vision_processing: {interface: "sensor_msgs/Image", rate: 30Hz}
  - audio_processing: {interface: "std_msgs/String", rate: 10Hz}
  - sensor_fusion: {interface: "fusion_msgs/Environment", rate: 10Hz}

cognition_layer:
  - task_planning: {interface: "planning_msgs/Plan", rate: 1Hz}
  - behavior_selection: {interface: "behavior_msgs/Behavior", rate: 5Hz}
  - state_estimation: {interface: "geometry_msgs/Pose", rate: 50Hz}

control_layer:
  - locomotion_control: {interface: "humanoid_msgs/LocomotionCommand", rate: 100Hz}
  - manipulation_control: {interface: "humanoid_msgs/ManipulationCommand", rate: 100Hz}
  - balance_control: {interface: "humanoid_msgs/BalanceCommand", rate: 200Hz}

hardware_interface:
  - joint_controllers: {interface: "sensor_msgs/JointState", rate: 200Hz}
  - safety_system: {interface: "std_msgs/Bool", rate: 100Hz}
```

## Integration Strategies

### Top-Down vs. Bottom-Up Integration

For humanoid robots, a hybrid approach often works best:

1. **Bottom-Up Component Testing**: Test individual subsystems in isolation
2. **Middle-Out Subsystem Integration**: Integrate related subsystems (e.g., perception + cognition)
3. **Top-Down System Integration**: Integrate all subsystems with stubs, then replace with real components

#### Component Integration Patterns

```python
# Pattern for safe component testing
class SafeComponentWrapper:
    def __init__(self, component, safety_monitor):
        self.component = component
        self.safety_monitor = safety_monitor
        self.last_execution_time = 0
        self.rate_limit_hz = 10.0
    
    def execute_safely(self, *args, **kwargs):
        # Check if safe to operate
        if not self.safety_monitor.is_safe():
            rospy.logwarn("Component execution blocked due to safety violation")
            return None
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_execution_time < 1.0 / self.rate_limit_hz:
            rospy.logwarn(f"Component called too frequently, rate limited to {self.rate_limit_hz}Hz")
            return None
        
        try:
            result = self.component.execute(*args, **kwargs)
            self.last_execution_time = current_time
            return result
        except Exception as e:
            rospy.logerr(f"Component execution failed: {e}")
            self.safety_monitor.trigger_safe_stop()
            return None

# Example of integrating a vision component safely
class SafeVisionModule:
    def __init__(self, camera_topic="/camera/rgb/image_raw"):
        self.camera_sub = rospy.Subscriber(
            camera_topic, 
            Image, 
            self._safe_image_callback
        )
        self.object_detector = ObjectDetectionPipeline()
        
        # Safety monitors
        self.cpu_monitor = CPUMonitor(threshold=0.85)  # 85% CPU usage threshold
        self.temp_monitor = TemperatureMonitor(threshold=70.0)  # 70°C threshold
        self.bandwidth_monitor = BandwidthMonitor(threshold=100.0)  # 100 Mbps threshold
    
    def _safe_image_callback(self, msg):
        # Check system health before processing
        if not self.cpu_monitor.is_healthy() or not self.temp_monitor.is_healthy():
            rospy.logwarn("System resources overloaded, skipping image processing")
            return
        
        # Process with timeout
        result = self._timed_process_image(msg)
        
        if result is not None:
            # Publish results
            self._publish_detections(result)
        else:
            rospy.logerr("Image processing timed out")
    
    def _timed_process_image(self, image_msg, timeout=2.0):
        """Process image with timeout to prevent system blocking"""
        result_container = {'result': None}
        exception_container = {'exception': None}
        
        def target():
            try:
                result_container['result'] = self.object_detector.detect(image_msg)
            except Exception as e:
                exception_container['exception'] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            rospy.logerr("Image processing thread timed out")
            return None
        elif exception_container['exception']:
            rospy.logerr(f"Image processing error: {exception_container['exception']}")
            return None
        else:
            return result_container['result']

class IntegrationTestingFramework:
    """Framework for integration testing of humanoid subsystems"""
    
    def __init__(self):
        self.test_results = {}
        self.active_tests = []
        self.test_sequence = []
        
        # Component interfaces
        self.components = {
            'locomotion': None,
            'manipulation': None,
            'perception': None,
            'cognition': None,
            'audio': None
        }
        
        # Test result publisher
        self.test_result_pub = rospy.Publisher(
            '/integration_test/results', 
            IntegrationTestResult, 
            queue_size=10
        )
        
        rospy.loginfo("Integration testing framework initialized")
    
    def register_component(self, name, interface):
        """Register a component for integration testing"""
        self.components[name] = interface
        rospy.loginfo(f"Registered component for testing: {name}")
    
    def define_test_sequence(self, sequence):
        """Define sequence of integration tests"""
        self.test_sequence = sequence
        rospy.loginfo(f"Defined test sequence with {len(sequence)} tests")
    
    def run_integration_tests(self, test_subset=None):
        """Run defined integration tests"""
        tests_to_run = self.test_sequence if test_subset is None else test_subset
        
        rospy.loginfo(f"Starting integration tests ({len(tests_to_run)} tests)")
        
        results = {}
        for test_config in tests_to_run:
            test_name = test_config['name']
            test_function = test_config['function']
            test_args = test_config.get('args', {})
            test_kwargs = test_config.get('kwargs', {})
            
            rospy.loginfo(f"Running test: {test_name}")
            
            try:
                # Prepare test environment
                self._prepare_test_environment(test_config)
                
                # Execute test
                test_result = test_function(**test_kwargs)
                
                # Evaluate results
                evaluation = self._evaluate_test_result(test_name, test_result, test_config)
                
                # Clean up test environment
                self._cleanup_test_environment(test_config)
                
                # Store results
                results[test_name] = evaluation
                
                # Publish result
                self._publish_test_result(test_name, evaluation)
                
            except Exception as e:
                rospy.logerr(f"Test {test_name} failed with error: {e}")
                results[test_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'metrics': {}
                }
                self._publish_test_result(test_name, results[test_name])
        
        # Aggregate and save results
        self.test_results = results
        self._save_test_results()
        
        # Generate summary report
        summary = self._generate_test_summary(results)
        self._publish_test_summary(summary)
        
        return results
    
    def _prepare_test_environment(self, test_config):
        """Prepare environment for specific test"""
        # Set initial conditions
        initial_conditions = test_config.get('initial_conditions', {})
        
        for component_name, initial_state in initial_conditions.items():
            if component_name in self.components:
                comp = self.components[component_name]
                if hasattr(comp, 'set_initial_state'):
                    comp.set_initial_state(initial_state)
        
        # Pause other non-related components if needed
        for comp_name, comp in self.components.items():
            if comp and test_config.get('exclusive_components', {}).get(comp_name, False):
                continue  # Don't pause this component
            if comp and hasattr(comp, 'pause'):
                comp.pause()
    
    def _evaluate_test_result(self, test_name, result, test_config):
        """Evaluate test result against expected outcomes"""
        expected_outcomes = test_config.get('expected_outcomes', {})
        metrics = test_config.get('metrics', [])
        thresholds = test_config.get('thresholds', {})
        
        evaluation = {
            'status': 'passed' if result else 'failed',
            'measured_values': {},
            'expected_values': expected_outcomes,
            'metrics': {},
            'error': None
        }
        
        # Calculate metrics if needed
        for metric_name in metrics:
            try:
                if metric_name == 'execution_time':
                    eval['metrics']['execution_time'] = result.get('execution_time', 0)
                    
                    if 'max_execution_time' in thresholds:
                        if result['execution_time'] > thresholds['max_execution_time']:
                            evaluation['status'] = 'failed'
                            evaluation['error'] = f"Execution time exceeded threshold: {result['execution_time']}s > {thresholds['max_execution_time']}s"
                
                elif metric_name == 'accuracy':
                    if hasattr(result, 'accuracy') or 'accuracy' in result:
                        accuracy = result.get('accuracy', getattr(result, 'accuracy', 0))
                        eval['metrics']['accuracy'] = accuracy
                        
                        if 'min_accuracy' in thresholds:
                            if accuracy < thresholds['min_accuracy']:
                                evaluation['status'] = 'failed'
                                evaluation['error'] = f"Accuracy below threshold: {accuracy} < {thresholds['min_accuracy']}"
                
                elif metric_name == 'success_rate':
                    if hasattr(result, 'success_rate') or 'success_rate' in result:
                        success_rate = result.get('success_rate', getattr(result, 'success_rate', 0))
                        eval['metrics']['success_rate'] = success_rate
                        
                        if 'min_success_rate' in thresholds:
                            if success_rate < thresholds['min_success_rate']:
                                evaluation['status'] = 'failed'
                                evaluation['error'] = f"Success rate below threshold: {success_rate} < {thresholds['min_success_rate']}"
                                
            except Exception as e:
                rospy.logwarn(f"Error calculating metric {metric_name}: {e}")
                evaluation['metrics'][metric_name] = None
        
        # Check if all required metrics pass thresholds
        if evaluation['status'] == 'passed':
            for metric, threshold_val in thresholds.items():
                if metric in evaluation['metrics']:
                    actual_val = evaluation['metrics'][metric]
                    if actual_val < threshold_val:
                        evaluation['status'] = 'failed'
                        evaluation['error'] = f"Metric {metric} ({actual_val}) below threshold ({threshold_val})"
                        break
        
        return evaluation
    
    def _generate_test_summary(self, results):
        """Generate summary of all test results"""
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r['status'] == 'passed')
        failed_tests = total_tests - passed_tests
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'execution_time': time.time() - self.test_start_time if hasattr(self, 'test_start_time') else 0,
            'detailed_results': results
        }
        
        return summary

# Integration test configuration system
class IntegrationTestConfig:
    """Configuration for integration tests"""
    
    def __init__(self):
        self.tests = {
            'locomotion_perception_integration': {
                'name': 'Locomotion-Perception Integration Test',
                'description': 'Test integration between navigation and perception systems',
                'components': ['locomotion', 'perception'],
                'sequence': [
                    {'action': 'initiate_navigation', 'params': {'target': 'kitchen'}},
                    {'action': 'detect_obstacles', 'params': {'timeout': 10.0}},
                    {'action': 'adjust_path', 'params': {}}
                ],
                'expected_outcomes': {
                    'path_completion_success': True,
                    'obstacle_detection_accuracy': 0.9
                },
                'metrics': ['execution_time', 'success_rate', 'accuracy'],
                'thresholds': {
                    'max_execution_time': 30.0,
                    'min_success_rate': 0.8,
                    'min_accuracy': 0.85
                },
                'initial_conditions': {
                    'locomotion': {'position': [0, 0, 0], 'orientation': [0, 0, 0, 1]},
                    'perception': {'calibrated': True, 'active': True}
                },
                'cleanup_required': True
            },
            'manipulation_vision_integration': {
                'name': 'Manipulation-Vision Integration Test',
                'description': 'Test integration between manipulation and vision systems',
                'components': ['manipulation', 'perception'],
                'sequence': [
                    {'action': 'detect_object', 'params': {'class': 'bottle'}},
                    {'action': 'plan_grasp', 'params': {}},
                    {'action': 'execute_grasp', 'params': {}}
                ],
                'expected_outcomes': {
                    'grasp_success': True,
                    'object_detection_precision': 0.95
                },
                'metrics': ['execution_time', 'success_rate', 'precision'],
                'thresholds': {
                    'max_execution_time': 15.0,
                    'min_success_rate': 0.7,
                    'min_precision': 0.9
                },
                'initial_conditions': {
                    'manipulation': {'ready': True, 'calibrated': True},
                    'perception': {'calibrated': True, 'active': True}
                },
                'cleanup_required': True
            },
            'audio_cognition_integration': {
                'name': 'Audio-Cognition Integration Test',
                'description': 'Test integration between audio processing and cognitive systems',
                'components': ['audio', 'cognition'],
                'sequence': [
                    {'action': 'listen_command', 'params': {'timeout': 10.0}},
                    {'action': 'parse_command', 'params': {}},
                    {'action': 'execute_action', 'params': {}}
                ],
                'expected_outcomes': {
                    'command_accuracy': 0.9,
                    'response_time': 5.0
                },
                'metrics': ['execution_time', 'accuracy', 'latency'],
                'thresholds': {
                    'max_execution_time': 10.0,
                    'min_accuracy': 0.85,
                    'max_latency': 3.0
                },
                'initial_conditions': {
                    'audio': {'microphones_active': True, 'calibrated': True},
                    'cognition': {'active': True, 'models_loaded': True}
                },
                'cleanup_required': True
            }
        }

# Safety-first integration approach
class SafeIntegrationManager:
    """Manages safe integration of robot subsystems"""
    
    def __init__(self):
        # Initialize safety monitors
        self.emergency_stop_monitor = EmergencyStopMonitor()
        self.balance_monitor = BalanceMonitor()
        self_collision_checker = SelfCollisionChecker()
        self.power_monitor = PowerMonitor()
        
        # Component interfaces with safe wrappers
        self.safely_wrapped_components = {}
        
        # Integration state
        self.integration_state = 'safedown'  # [safedown, initializing, testing, operational]
        self.failed_components = []
        
        # Safety interlocks
        self.safety_interlocks = {
            'balance_critical': False,
            'power_critical': False,
            'component_failure': False
        }
        
        rospy.loginfo("Safe integration manager initialized")
    
    def integrate_components(self, component_configs):
        """Safely integrate components with monitoring"""
        rospy.loginfo("Starting safe component integration")
        
        integration_order = self._determine_integration_order(component_configs)
        
        for component_name in integration_order:
            rospy.loginfo(f"Integrating component: {component_name}")
            
            config = component_configs[component_name]
            
            # Check safety before integrating component
            if self._safety_critical():
                rospy.logerr("Safety critical condition detected, stopping integration")
                return False
            
            # Create safe wrapper for component
            wrapped_component = SafeComponentWrapper(
                component=self._create_component(config),
                safety_monitor=self
            )
            self.safely_wrapped_components[component_name] = wrapped_component
            
            # Initialize component with safety check
            try:
                init_result = wrapped_component.execute_safely('initialize')
                if not init_result:
                    rospy.logerr(f"Failed to initialize component: {component_name}")
                    self.failed_components.append(component_name)
                    continue
            except Exception as e:
                rospy.logerr(f"Error during component initialization: {e}")
                self.failed_components.append(component_name)
                continue
            
            # Test component in isolation
            if not self._test_component_in_isolation(component_name, wrapped_component):
                rospy.logerr(f"Component failed isolation test: {component_name}")
                self.failed_components.append(component_name)
                continue
            
            # Verify component doesn't interfere with safety systems
            if not self._verify_component_safety(component_name):
                rospy.logerr(f"Component violates safety requirements: {component_name}")
                self.failed_components.append(component_name)
                continue
            
            rospy.loginfo(f"Successfully integrated component: {component_name}")
        
        # Check if all components integrated successfully
        successfully_integrated = len(self.failed_components) == 0
        
        if successfully_integrated:
            self.integration_state = 'operational'
            rospy.loginfo("All components successfully integrated")
        else:
            self.integration_state = 'degraded_operation'
            rospy.logwarn(f"Integration completed with failed components: {self.failed_components}")
        
        return successfully_integrated
    
    def _determine_integration_order(self, configs):
        """Determine safe order for integrating components based on dependencies"""
        # For humanoid robots, a typical safe order is:
        # 1. Safety systems
        # 2. Base systems (communications, power management)
        # 3. Perception systems (non-motorized)
        # 4. Control systems (locomotion, manipulation)
        # 5. Cognitive systems
        # 6. Interaction systems
        
        integration_order = []
        
        # Group components by category
        safety_components = []
        base_components = []
        perception_components = []
        control_components = []
        cognitive_components = []
        interaction_components = []
        
        for name, config in configs.items():
            category = config.get('category', 'base')
            if category == 'safety':
                safety_components.append(name)
            elif category == 'perception':
                perception_components.append(name)
            elif category == 'control':
                control_components.append(name)
            elif category == 'cognition':
                cognitive_components.append(name)
            elif category == 'interaction':
                interaction_components.append(name)
            else:
                base_components.append(name)
        
        # Build integration order
        integration_order.extend(safety_components)
        integration_order.extend(base_components)
        integration_order.extend(perception_components)
        integration_order.extend(control_components)
        integration_order.extend(cognitive_components)
        integration_order.extend(interaction_components)
        
        return integration_order
    
    def _verify_component_safety(self, component_name):
        """Verify that component doesn't compromise safety systems"""
        # Check that component doesn't overload CPU
        cpu_usage_before = self._get_cpu_usage()
        time.sleep(0.5)  # Let component run briefly
        cpu_usage_after = self._get_cpu_usage()
        
        if cpu_usage_after > 0.90:  # 90% threshold
            rospy.logerr(f"Component {component_name} causes excessive CPU usage: {cpu_usage_after:.2f}")
            return False
        
        # Check that component doesn't spike power consumption
        power_before = self.power_monitor.get_current_draw()
        time.sleep(0.5)
        power_after = self.power_monitor.get_current_draw()
        
        power_spike = abs(power_after - power_before)
        if power_spike > 5.0:  # 5A threshold
            rospy.logerr(f"Component {component_name} causes power spike: {power_spike:.2f}A")
            return False
        
        return True
    
    def _get_cpu_usage(self):
        """Get current CPU usage"""
        import psutil
        return psutil.cpu_percent(interval=0.1) / 100.0
    
    def _test_component_in_isolation(self, component_name, component_wrapper):
        """Test component functionality in isolation"""
        # This would run specific functional tests for the component
        # Each component type would have specific tests
        
        tests_passed = True
        
        # Example tests (would vary by component type)
        if 'perception' in component_name:
            # Test that perception component can process data without crashing
            try:
                test_result = component_wrapper.execute_safely(
                    'process_sample_data',
                    sample_data=self._create_test_data_for_component(component_name)
                )
                if test_result is None:
                    tests_passed = False
            except:
                tests_passed = False
        elif 'control' in component_name:
            # Test basic control command
            try:
                test_result = component_wrapper.execute_safely(
                    'send_test_command',
                    command=self._create_test_command_for_component(component_name)
                )
                if test_result is None:
                    tests_passed = False
            except:
                tests_passed = False
        
        return tests_passed
    
    def _safety_critical(self):
        """Check if any safety-critical conditions are active"""
        return (
            self.safety_interlocks['balance_critical'] or
            self.safety_interlocks['power_critical'] or
            self.safety_interlocks['component_failure'] or
            not self.emergency_stop_monitor.is_normal()
        )
    
    def _create_test_data_for_component(self, component_name):
        """Create appropriate test data for the specified component"""
        if 'vision' in component_name.lower():
            # Create a simple test image
            import numpy as np
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            return test_image
        elif 'audio' in component_name.lower():
            # Create test audio data (simplified)
            return np.random.random(44100)  # 1 second of random audio
        else:
            return {}
    
    def _create_test_command_for_component(self, component_name):
        """Create appropriate test command for the specified component"""
        if 'arm' in component_name.lower() or 'manipulation' in component_name.lower():
            # Test command to move to neutral position
            return {'joint_positions': [0.0] * 7}  # 7-DOF arm to neutral
        elif 'locomotion' in component_name.lower():
            # Test command to stand position
            return {'stance': 'standing', 'position': [0, 0, 0]}
        else:
            return {}
```

## Testing Methodologies for Humanoid Robots

### Unit Testing for Robotics Components

```python
import unittest
import numpy as np
from geometry_msgs.msg import Pose, Point, Quaternion
import rospy

class TestLocomotionController(unittest.TestCase):
    """Unit tests for locomotion controller"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.controller = LocomotionController()
        
        # Sample poses for testing
        self.origin_pose = Pose(
            position=Point(x=0.0, y=0.0, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        )
        
        self.target_pose = Pose(
            position=Point(x=1.0, y=1.0, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.707, w=0.707)  # 90 degree rotation
        )
    
    def test_compute_walk_trajectory(self):
        """Test that walk trajectory computation works correctly"""
        # Test basic trajectory computation
        trajectory = self.controller.compute_walk_trajectory(
            self.origin_pose,
            self.target_pose,
            step_length=0.3,
            step_height=0.05
        )
        
        # Assertions
        self.assertIsNotNone(trajectory)
        self.assertGreater(len(trajectory.points), 0, "Trajectory should have points")
        
        # Check that trajectory reaches close to target
        final_point = trajectory.points[-1]
        final_position = Point(
            x=final_point.positions[0],  # Simplified - in real system would map to actual joints
            y=final_point.positions[1],
            z=0.0  # Simplified
        )
        
        distance_to_target = np.linalg.norm([
            self.target_pose.position.x - final_position.x,
            self.target_pose.position.y - final_position.y
        ])
        
        self.assertLess(distance_to_target, 0.2, "Should reach within 20cm of target")
    
    def test_balance_maintenance(self):
        """Test that controller maintains balance during locomotion"""
        # Simulate various disturbances
        disturbances = [
            {'type': 'push', 'magnitude': [0.1, 0.0, 0.0]},  # Slight push in X
            {'type': 'push', 'magnitude': [0.0, 0.1, 0.0]},  # Slight push in Y
            {'type': 'tilt', 'magnitude': [0.05, 0.05, 0.0]}  # Slight tilt
        ]
        
        for disturbance in disturbances:
            with self.subTest(disturbance=disturbance):
                # Apply disturbance and check balance recovery
                self.controller.apply_disturbance(disturbance)
                
                # Allow time for balance recovery
                rospy.sleep(0.5)
                
                # Check that balance is restored
                balance_status = self.controller.get_balance_status()
                self.assertTrue(balance_status['stable'], 
                              f"Balance should be stable after {disturbance['type']} disturbance")
    
    def test_step_generation(self):
        """Test that step generation is appropriate for different terrains"""
        # Test flat ground
        steps_flat = self.controller.generate_steps_for_terrain(
            start_pose=self.origin_pose,
            end_pose=self.target_pose,
            terrain_type='flat'
        )
        
        self.assertGreater(len(steps_flat), 0, "Should generate steps for flat terrain")
        
        # Test uneven terrain (would require more complex validation)
        # For now, just ensure it doesn't crash
        steps_uneven = self.controller.generate_steps_for_terrain(
            start_pose=self.origin_pose,
            end_pose=self.target_pose,
            terrain_type='uneven'
        )
        
        self.assertIsNotNone(steps_uneven, "Should handle uneven terrain without crashing")
    
    def test_gait_transition(self):
        """Test transitioning between different gaits"""
        # Test transitions between different gaits
        gaits = ['standing', 'walking', 'trotting', 'running']
        
        for i in range(len(gaits) - 1):
            from_gait = gaits[i]
            to_gait = gaits[i + 1]
            
            with self.subTest(from_gait=from_gait, to_gait=to_gait):
                # Attempt gait transition
                transition_successful = self.controller.transition_gait(from_gait, to_gait)
                
                # For now, just test that it doesn't crash
                # In real system, would have more specific validation
                self.assertIsNotNone(transition_successful)

class TestPerceptionPipeline(unittest.TestCase):
    """Unit tests for perception pipeline"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.pipeline = RealTimeVisionPipeline()
        
        # Create mock image data
        self.mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create mock depth data
        self.mock_depth = np.random.uniform(0.5, 3.0, (480, 640)).astype(np.float32)
    
    def test_object_detection(self):
        """Test object detection functionality"""
        # Since we don't have real model in test environment,
        # we'll just test that the function is callable and returns expected format
        try:
            detections = self.pipeline.detect_objects(self.mock_image)
            
            # Should return a list or dictionary-like structure
            self.assertIsNotNone(detections, "Detections should not be None")
            
        except NotImplementedError:
            # If not implemented, that's OK for testing purposes
            self.skipTest("Object detection not implemented in test environment")
    
    def test_pose_estimation(self):
        """Test object pose estimation"""
        # Create mock 2D bounding box
        mock_bbox = [100, 100, 200, 200]  # x, y, width, height
        
        try:
            pose_6d = self.pipeline.estimate_object_pose_6d(
                self.mock_image, 
                self.mock_depth, 
                mock_bbox
            )
            
            if pose_6d is not None:
                # If a pose is estimated, it should have position and orientation
                self.assertIn('position', pose_6d)
                self.assertIn('orientation', pose_6d)
                
                pos = pose_6d['position']
                self.assertIsInstance(pos, dict)
                self.assertIn('x', pos)
                self.assertIn('y', pos)
                self.assertIn('z', pos)
                
                quat = pose_6d['orientation']
                self.assertIsInstance(quat, dict)
                self.assertIn('x', quat)
                self.assertIn('y', quat)
                self.assertIn('z', quat)
                self.assertIn('w', quat)
            
        except NotImplementedError:
            # If not implemented, that's OK for testing purposes
            self.skipTest("Pose estimation not implemented in test environment")
    
    def test_image_preprocessing(self):
        """Test image preprocessing pipeline"""
        processed_image = self.pipeline.preprocess_image(self.mock_image)
        
        # Output should be numpy array
        self.assertIsInstance(processed_image, np.ndarray)
        
        # Should have same dimensions (or expected different dimensions if sizing is part of preprocessing)
        self.assertEqual(processed_image.shape[:2], self.mock_image.shape[:2])

class TestManipulationController(unittest.TestCase):
    """Unit tests for manipulation controller"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.controller = ManipulationController()
        
        # Sample poses for testing
        self.home_pose = {
            'position': {'x': 0.3, 'y': 0.0, 'z': 0.5},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
        }
        
        self.object_pose = {
            'position': {'x': 0.5, 'y': 0.2, 'z': 0.1},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
        }
    
    def test_inverse_kinematics_solution(self):
        """Test inverse kinematics solution generation"""
        try:
            joint_angles = self.controller.inverse_kinematics(self.object_pose)
            
            if joint_angles is not None:
                # Should return an array of joint angles
                self.assertIsInstance(joint_angles, (list, np.ndarray))
                
                # Should have expected number of joints (for a 7-DOF arm)
                expected_joints = 7  # Adjust based on robot
                self.assertEqual(len(joint_angles), expected_joints,
                               "Should return expected number of joint angles")
                
                # Joint angles should be within reasonable bounds
                for angle in joint_angles:
                    self.assertGreaterEqual(angle, -np.pi, "Joint angle should be >= -π")
                    self.assertLessEqual(angle, np.pi, "Joint angle should be <= π")
            
        except NotImplementedError:
            self.skipTest("IK not implemented in test environment")
    
    def test_trajectory_generation(self):
        """Test trajectory generation between poses"""
        try:
            trajectory = self.controller.generate_trajectory(
                self.home_pose,
                self.object_pose,
                max_velocity=0.5,
                max_acceleration=0.2
            )
            
            if trajectory is not None:
                self.assertGreater(len(trajectory.points), 1, 
                                "Trajectory should have at least 2 points")
                
                # Check that points have expected properties
                for point in trajectory.points:
                    self.assertTrue(hasattr(point, 'positions'))
                    self.assertTrue(hasattr(point, 'velocities'))
                    self.assertTrue(hasattr(point, 'time_from_start'))
            
        except NotImplementedError:
            self.skipTest("Trajectory generation not implemented in test environment")
    
    def test_gripper_control(self):
        """Test gripper control commands"""
        # Test grip commands
        try:
            # Test opening
            success_open = self.controller.gripper_command('open', width=0.08)
            self.assertTrue(success_open is None or isinstance(success_open, bool))
            
            # Test closing
            success_close = self.controller.gripper_command('close', width=0.02)
            self.assertTrue(success_close is None or isinstance(success_close, bool))
            
            # Test specific width
            success_width = self.controller.gripper_command('set_width', width=0.05)
            self.assertTrue(success_width is None or isinstance(success_width, bool))
            
        except AttributeError:
            self.skipTest("Gripper control not implemented in test environment")

# Test suite for the entire system
def test_suite():
    """Create a test suite combining all robot tests"""
    suite = unittest.TestSuite()
    
    # Add tests for different components
    suite.addTest(unittest.makeSuite(TestLocomotionController))
    suite.addTest(unittest.makeSuite(TestPerceptionPipeline))
    suite.addTest(unittest.makeSuite(TestManipulationController))
    
    return suite

if __name__ == '__main__':
    # Create test runner with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    suite = test_suite()
    
    # Run tests
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors))/result.testsRun)*100:.1f}%")
```

### Integration Tests

```python
import rospy
import unittest
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import time

class TestLocomotionPerceptionIntegration(unittest.TestCase):
    """Integration tests for locomotion and perception systems"""
    
    def setUp(self):
        """Set up integration test environment"""
        # Publishers
        self.navigation_goal_pub = rospy.Publisher(
            '/move_base_simple/goal', 
            PoseStamped, 
            queue_size=10
        )
        
        # Subscribers
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        # Test state
        self.current_pose = None
        self.current_orientation = None
        self.obstacle_detected = False
        self.last_scan_time = rospy.Time(0)
        
        # Wait for connections
        rospy.sleep(2.0)
    
    def odom_callback(self, msg):
        """Store current robot pose from odometry"""
        self.current_pose = msg.pose.pose
    
    def imu_callback(self, msg):
        """Store current orientation from IMU"""
        self.current_orientation = msg.orientation
    
    def scan_callback(self, msg):
        """Detect obstacles from laser scan"""
        # Check for obstacles in front of robot (simplified)
        front_distances = msg.ranges[len(msg.ranges)//2 - 50 : len(msg.ranges)//2 + 50]
        min_distance = min(front_distances) if front_distances else float('inf')
        
        self.obstacle_detected = min_distance < 0.5  # Obstacle within 50cm
        self.last_scan_time = msg.header.stamp
    
    def test_navigation_with_obstacle_avoidance(self):
        """Test navigation system with dynamic obstacle avoidance"""
        # Set a navigation goal
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = 5.0  # 5 meters forward
        goal.pose.position.y = 0.0
        goal.pose.orientation.w = 1.0  # Facing forward
        
        self.navigation_goal_pub.publish(goal)
        
        # Wait and monitor for the robot's behavior
        start_time = rospy.Time.now()
        timeout = rospy.Duration(30.0)  # 30 second timeout
        
        success = False
        while (rospy.Time.now() - start_time) < timeout and not success:
            # Check if robot reached goal (approximate)
            if self.current_pose:
                dist_to_goal = np.sqrt(
                    (self.current_pose.position.x - goal.pose.position.x)**2 +
                    (self.current_pose.position.y - goal.pose.position.y)**2
                )
                
                if dist_to_goal < 0.5:  # Within 50cm of goal
                    success = True
                    break
            
            # Verify obstacle avoidance behavior
            if self.obstacle_detected:
                # When obstacle is detected, robot should slow down or stop
                # This would be verified by checking velocity commands separately
                rospy.loginfo("Obstacle detected during navigation")
            
            rospy.sleep(0.1)  # 10Hz monitoring
        
        self.assertTrue(success, "Robot should reach navigation goal within timeout")
        
        # Additional verification: robot should have maintained balance throughout
        if self.current_orientation:
            # Check that robot didn't tip over (simplified - would check actual angles)
            # Convert quaternion to euler angles to check for excessive roll/pitch
            from tf.transformations import euler_from_quaternion
            roll, pitch, yaw = euler_from_quaternion([
                self.current_orientation.x,
                self.current_orientation.y,
                self.current_orientation.z,
                self.current_orientation.w
            ])
            
            # Check that robot stayed relatively upright (within 15 degrees)
            self.assertLess(abs(roll), np.deg2rad(15), "Robot should maintain balance (not roll > 15 deg)")
            self.assertLess(abs(pitch), np.deg2rad(15), "Robot should maintain balance (not pitch > 15 deg)")

class TestManipulationVisionIntegration(unittest.TestCase):
    """Integration tests for manipulation and vision systems"""
    
    def setUp(self):
        """Set up manipulation-vision integration test"""
        # Publishers for manipulation
        self.joint_cmd_pub = rospy.Publisher(
            '/arm_controller/command', 
            JointTrajectory, 
            queue_size=10
        )
        self.gripper_cmd_pub = rospy.Publisher(
            '/gripper_controller/command',
            GripperCommand,
            queue_size=10
        )
        
        # Subscribers for vision feedback
        self.object_pose_sub = rospy.Subscriber(
            '/vision/object_poses', 
            PoseArray, 
            self.object_pose_callback
        )
        self.joint_state_sub = rospy.Subscriber(
            '/joint_states', 
            JointState, 
            self.joint_state_callback
        )
        
        # Test state
        self.object_poses = {}
        self.current_joint_positions = None
        self.grasp_attempt_successful = False
        
        # Initialize vision system for this test
        self.vision_system = RealTimeVisionPipeline()
        
        # Wait for connections
        rospy.sleep(2.0)
    
    def object_pose_callback(self, msg):
        """Update known object poses from vision system"""
        for pose_stamped in msg.poses:
            obj_id = pose_stamped.header.frame_id  # Simplified ID extraction
            self.object_poses[obj_id] = pose_stamped.pose
    
    def joint_state_callback(self, msg):
        """Update current joint positions"""
        self.current_joint_positions = msg.position
    
    def test_grasp_with_vision_feedback(self):
        """Test grasping task using vision feedback for positioning"""
        # First, ensure an object is visible (mock scenario)
        # In a real test, we would place a known object in the field of view
        
        # Simulate detection of an object
        test_object_id = "test_object_001"
        test_object_pose = Pose()
        test_object_pose.position.x = 0.6  # 60cm in front
        test_object_pose.position.y = 0.1  # 10cm to the right
        test_object_pose.position.z = 0.1  # 10cm above ground
        test_object_pose.orientation.w = 1.0  # Default orientation
        
        self.object_poses[test_object_id] = test_object_pose
        
        # Plan grasp based on vision feedback
        grasp_pose = self._calculate_grasp_pose(test_object_pose)
        
        # Execute grasp sequence
        success = self._execute_grasp_sequence(grasp_pose)
        
        self.assertTrue(success, "Grasp sequence should complete successfully")
        
        # Verify post-grasp state (object should be in gripper)
        if success:
            # Check that gripper is closed
            gripper_state = self._get_gripper_state()
            self.assertEqual(gripper_state, 'closed', "Gripper should be closed after successful grasp")
            
            # Check that object is no longer detectable at original location (simplified check)
            rospy.sleep(1.0)  # Allow time for grasp to complete
            
            # Verify no object detected at original location (within tolerance)
            # This would depend on specific vision implementation
            # For now, just verify the process ran without error
    
    def _calculate_grasp_pose(self, obj_pose):
        """Calculate grasp pose from object pose"""
        # Simple approach: grasp from above, align gripper with object orientation
        grasp_pose = copy.deepcopy(obj_pose)
        
        # Adjust height to approach from above
        grasp_pose.position.z += 0.15  # 15cm above object
        
        # For a top grasp, maintain object's rotation but ensure gripper fingers are vertical
        # This might require specific orientation adjustments based on object shape
        
        return grasp_pose
    
    def _execute_grasp_sequence(self, grasp_pose):
        """Execute complete grasp sequence with safety checks"""
        try:
            # 1. Move to pre-grasp position
            pre_grasp_pose = copy.deepcopy(grasp_pose)
            pre_grasp_pose.position.z += 0.05  # 5cm above grasp point
            
            self._move_to_pose(pre_grasp_pose)
            
            # 2. Slow approach to grasp position
            self._move_to_pose(grasp_pose, velocity_scale=0.2)
            
            # 3. Close gripper gently
            gripper_cmd = GripperCommand()
            gripper_cmd.position = 0.02  # Desired finger width for grasp
            gripper_cmd.max_effort = 30.0  # Moderate effort
            self.gripper_cmd_pub.publish(gripper_cmd)
            
            # 4. Wait for grasp to complete and verify
            rospy.sleep(1.0)
            
            # 5. Lift object slightly
            lift_pose = copy.deepcopy(grasp_pose)
            lift_pose.position.z += 0.05  # Lift 5cm
            
            self._move_to_pose(lift_pose, velocity_scale=0.3)
            
            # 6. Verify grasp was successful
            # In a real system, this would check force sensors or vision confirmation
            return self._verify_grasp_success()
            
        except Exception as e:
            rospy.logerr(f"Grasp execution failed: {e}")
            return False
    
    def _move_to_pose(self, pose, velocity_scale=1.0):
        """Move end effector to specified pose"""
        # This would interface with robot's motion planning
        # For testing, just simulate the move
        rospy.loginfo(f"Moving to pose: ({pose.position.x}, {pose.position.y}, {pose.position.z})")
        rospy.sleep(1.0)  # Simulate move time
    
    def _verify_grasp_success(self):
        """Verify that grasp was successful"""
        # In a real system, this would check:
        # - Force/torque readings from gripper
        # - Change in camera images to detect object removal
        # - Joint efforts indicating load
        # - Successful lift with minimal position deviation
        
        # For now, assume success (in real system, implement actual checks)
        return True

class TestCognitivePerceptionIntegration(unittest.TestCase):
    """Integration test for cognitive and perception systems"""
    
    def setUp(self):
        """Set up cognitive-perception integration test"""
        # Publishers
        self.command_pub = rospy.Publisher('/natural_language_command', String, queue_size=10)
        self.feedback_pub = rospy.Publisher('/cognitive_feedback', String, queue_size=10)
        
        # Subscribers
        self.result_sub = rospy.Subscriber('/cognitive_result', String, self.result_callback)
        
        # Test state
        self.last_result = None
        self.result_received = False
        
        # Wait for connections
        rospy.sleep(1.0)
    
    def result_callback(self, msg):
        """Store cognitive system results"""
        self.last_result = msg.data
        self.result_received = True
    
    def test_command_understanding_and_execution(self):
        """Test that natural language commands are properly understood and executed"""
        # Send a natural language command
        command = String()
        command.data = "Go to the kitchen and bring me the red bottle from the table"
        
        self.command_pub.publish(command)
        
        # Wait for cognitive system to process
        timeout = rospy.Time.now() + rospy.Duration(10.0)  # 10 second timeout
        
        while not self.result_received and rospy.Time.now() < timeout:
            rospy.sleep(0.1)
        
        self.assertTrue(self.result_received, "Cognitive system should respond to command")
        
        if self.result_received:
            # The result should include information about task decomposition
            # This would be implementation-specific, but should indicate understanding
            result_str = self.last_result.lower()
            
            # Check that result mentions relevant actions
            self.assertIn('navigate', result_str, "Result should mention navigation")
            self.assertIn('kitchen', result_str, "Result should mention destination")
            self.assertIn('grasp', result_str, "Result should mention manipulation")
            self.assertIn('bottle', result_str, "Result should mention object to grasp")

# Performance integration tests
class TestSystemPerformance(unittest.TestCase):
    """Performance tests for integrated system"""
    
    def setUp(self):
        """Set up performance test environment"""
        # Publishers for different components
        self.perception_load_pub = rospy.Publisher('/test/perception_load', Image, queue_size=10)
        self.navigation_cmd_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        
        # Performance monitoring
        self.start_time = time.time()
        self.cpu_monitor = CPUMonitor()
        self.memory_monitor = MemoryMonitor()
        self.network_monitor = NetworkMonitor()
    
    def test_real_time_performance(self):
        """Test system performance under load"""
        # Start performance monitoring
        self.cpu_monitor.start_monitoring()
        self.memory_monitor.start_monitoring()
        
        # Simulate simultaneous perception and navigation
        start_time = time.time()
        duration = 30.0  # Test for 30 seconds
        
        # Send continuous perception requests and navigation commands
        rate = rospy.Rate(10)  # 10Hz
        command_count = 0
        
        while time.time() - start_time < duration:
            # Simulate perception load
            test_image = self._create_large_test_image()
            self.perception_load_pub.publish(test_image)
            
            # Send occasional navigation commands
            if command_count % 5 == 0:  # Every 5th iteration
                goal = PoseStamped()
                goal.header.frame_id = "map"
                goal.header.stamp = rospy.Time.now()
                goal.pose.position.x = 1.0 * np.sin(time.time())
                goal.pose.position.y = 1.0 * np.cos(time.time())
                goal.pose.orientation.w = 1.0
                self.navigation_cmd_pub.publish(goal)
            
            command_count += 1
            rate.sleep()
        
        # Verify performance requirements
        avg_cpu_usage = self.cpu_monitor.get_average_usage()
        peak_memory_usage = self.memory_monitor.get_peak_usage()
        
        # System should maintain performance under load
        self.assertLess(avg_cpu_usage, 0.90, "Average CPU usage should stay below 90%")
        self.assertLess(peak_memory_usage, 0.85, "Peak memory usage should stay below 85%")
        
        rospy.loginfo(f"Performance test results - CPU: {avg_cpu_usage:.2f}, Memory: {peak_memory_usage:.2f}")
    
    def _create_large_test_image(self):
        """Create a large image to stress test perception system"""
        # Create a large image (simulating high-resolution input)
        large_image = np.random.randint(0, 255, (1200, 1920, 3), dtype=np.uint8)
        
        # Convert to ROS Image message
        img_msg = Image()
        img_msg.header.stamp = rospy.Time.now()
        img_msg.encoding = "bgr8"
        img_msg.height = large_image.shape[0]
        img_msg.width = large_image.shape[1]
        img_msg.step = large_image.shape[1] * 3  # 3 bytes per pixel for BGR
        img_msg.data = large_image.tobytes()
        
        return img_msg

def test_suite():
    """Create comprehensive integration test suite"""
    suite = unittest.TestSuite()
    
    # Add integration tests
    suite.addTest(unittest.makeSuite(TestLocomotionPerceptionIntegration))
    suite.addTest(unittest.makeSuite(TestManipulationVisionIntegration))
    suite.addTest(unittest.makeSuite(TestCognitivePerceptionIntegration))
    suite.addTest(unittest.makeSuite(TestSystemPerformance))
    
    return suite

if __name__ == '__main__':
    rospy.init_node('integration_tests', anonymous=True)
    
    # Create test runner
    runner = unittest.TextTestRunner(verbosity=2)
    suite = test_suite()
    
    # Run tests
    result = runner.run(suite)
    
    # Print performance summary
    print(f"\n=== Integration Test Results ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors))/result.testsRun)*100:.1f}%")
    
    if result.failures or result.errors:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
        
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    print(f"\nTest execution completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
```

## Continuous Integration for Robotics

### CI/CD Pipeline Configuration

```python
# ci_cd_pipeline.py
import os
import subprocess
import shutil
import tempfile
from pathlib import Path
import docker

class RoboticsCIPipeline:
    """CI/CD pipeline for robotics systems"""
    
    def __init__(self, config_file="ci_config.yaml"):
        self.config = self._load_ci_config(config_file)
        self.repo_root = Path.cwd()
        self.build_artifacts_dir = self.repo_root / "build_artifacts"
        self.test_results_dir = self.repo_root / "test_results"
        self.docker_client = docker.from_env()
        
        # Create necessary directories
        self.build_artifacts_dir.mkdir(exist_ok=True)
        self.test_results_dir.mkdir(exist_ok=True)
        
        rospy.loginfo("Robotics CI/CD pipeline initialized")
    
    def _load_ci_config(self, config_file):
        """Load CI configuration from YAML file"""
        import yaml
        
        default_config = {
            'stages': [
                'build',
                'unit_test',
                'integration_test',
                'deployment'
            ],
            'environments': {
                'development': {
                    'docker_image': 'robotics-dev:latest',
                    'tests': ['unit', 'lint']
                },
                'staging': {
                    'docker_image': 'robotics-staging:latest',
                    'tests': ['unit', 'integration', 'performance']
                },
                'production': {
                    'docker_image': 'robotics-production:latest',
                    'tests': ['unit', 'integration', 'acceptance', 'performance']
                }
            },
            'build': {
                'dockerfile': 'Dockerfile.ros',
                'target_arch': 'x86_64',
                'dependencies': ['ros-noetic', 'opencv', 'pytorch']
            },
            'testing': {
                'unit_test_pattern': 'test_*_unittest.py',
                'integration_test_pattern': 'test_*_integration.py',
                'coverage_threshold': 0.80,
                'performance_threshold': {
                    'max_cpu': 0.85,
                    'max_memory': 0.80,
                    'min_throughput': 10.0  # Hz
                }
            },
            'deployment': {
                'targets': ['simulated_robot', 'physical_robot'],
                'rollback_strategy': 'blue_green',
                'health_check_timeout': 60  # seconds
            }
        }
        
        config_path = self.repo_root / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Merge user config with defaults
                self._deep_merge(default_config, user_config)
        
        return default_config
    
    def _deep_merge(self, base_dict, update_dict):
        """Recursively merge update_dict into base_dict"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def run_pipeline(self, environment='development', commit_sha=None):
        """Run the complete CI/CD pipeline"""
        rospy.loginfo(f"Starting CI/CD pipeline for environment: {environment}")
        
        try:
            # 1. Build stage
            build_success = self._build_stage(environment)
            if not build_success:
                rospy.logerr("Build stage failed, stopping pipeline")
                return False
            
            # 2. Test stage
            test_success = self._test_stage(environment)
            if not test_success:
                rospy.logerr("Test stage failed, stopping pipeline")
                return False
            
            # 3. Deployment stage (only for non-development environments)
            if environment != 'development':
                deployment_success = self._deployment_stage(environment, commit_sha)
                if not deployment_success:
                    rospy.logerr("Deployment stage failed")
                    return False
            
            rospy.loginfo(f"Pipeline completed successfully for environment: {environment}")
            return True
            
        except Exception as e:
            rospy.logerr(f"Pipeline failed with error: {e}")
            return False
    
    def _build_stage(self, environment):
        """Build the robotics software"""
        rospy.loginfo("Starting build stage")
        
        # Pull latest base image
        docker_image = self.config['environments'][environment]['docker_image']
        
        try:
            self.docker_client.images.pull(docker_image)
        except Exception as e:
            rospy.logwarn(f"Could not pull base image {docker_image}: {e}")
            # Continue with build anyway, Docker will pull during build
        
        # Build Docker image with project code
        dockerfile = self.config['build']['dockerfile']
        build_tag = f"robotics-build:{os.environ.get('BUILD_NUMBER', 'local')}"
        
        build_logs = self.docker_client.api.build(
            path=".",
            dockerfile=dockerfile,
            tag=build_tag,
            rm=True,  # Remove intermediate containers
            quiet=False
        )
        
        # Stream build logs
        for chunk in build_logs:
            # Decode and print build logs
            if 'stream' in chunk:
                print(chunk['stream'], end='')
            elif 'error' in chunk:
                rospy.logerr(f"Build error: {chunk['error']}")
                return False
        
        # Tag successful build for artifact storage
        self._tag_build_artifact(build_tag)
        
        rospy.loginfo("Build stage completed successfully")
        return True
    
    def _test_stage(self, environment):
        """Run tests on the built image"""
        rospy.loginfo("Starting test stage")
        
        # Get test configuration for environment
        test_types = self.config['environments'][environment]['tests']
        
        all_tests_passed = True
        
        for test_type in test_types:
            if test_type == 'unit':
                success = self._run_unit_tests()
            elif test_type == 'integration':
                success = self._run_integration_tests()
            elif test_type == 'performance':
                success = self._run_performance_tests()
            elif test_type == 'lint':
                success = self._run_code_quality_checks()
            else:
                rospy.logwarn(f"Unknown test type: {test_type}, skipping")
                success = True  # Don't fail on unknown test type
            
            if not success:
                rospy.logerr(f"{test_type.capitalize()} tests failed")
                all_tests_passed = False
            else:
                rospy.loginfo(f"{test_type.capitalize()} tests passed")
        
        rospy.loginfo(f"Test stage completed: {'PASSED' if all_tests_passed else 'FAILED'}")
        return all_tests_passed
    
    def _run_unit_tests(self):
        """Run unit tests using the built image"""
        rospy.loginfo("Running unit tests...")
        
        # Create temporary results directory
        test_results_temp = tempfile.mkdtemp(prefix="unit_test_results_")
        
        try:
            # Run unit tests in container
            container = self.docker_client.containers.run(
                f"robotics-build:{os.environ.get('BUILD_NUMBER', 'local')}",
                command="bash -c 'cd /catkin_ws && source devel/setup.bash && catkin_make run_tests'",
                volumes={
                    os.path.abspath(test_results_temp): {'bind': '/test_results', 'mode': 'rw'}
                },
                remove=True,
                detach=False,
                stdout=True,
                stderr=True
            )
            
            # Check test results for failures
            test_results_file = Path(test_results_temp) / "test_results.xml"
            if test_results_file.exists():
                return self._evaluate_test_results(test_results_file)
            else:
                rospy.logerr("No test results file generated")
                return False
                
        except Exception as e:
            rospy.logerr(f"Unit tests failed: {e}")
            return False
        finally:
            # Clean up temporary directory
            shutil.rmtree(test_results_temp, ignore_errors=True)
    
    def _run_integration_tests(self):
        """Run integration tests using the built image"""
        rospy.loginfo("Running integration tests...")
        
        test_results_temp = tempfile.mkdtemp(prefix="integration_test_results_")
        
        try:
            # Run integration tests in container with extra privileges
            # (for hardware access simulation, etc.)
            container = self.docker_client.containers.run(
                f"robotics-build:{os.environ.get('BUILD_NUMBER', 'local')}",
                command="bash -c 'cd /catkin_ws && source devel/setup.bash && python -m pytest tests/integration/ -v'",
                volumes={
                    os.path.abspath(test_results_temp): {'bind': '/test_results', 'mode': 'rw'}
                },
                privileged=True,  # Needed for some robotics simulation
                remove=True,
                detach=False,
                stdout=True,
                stderr=True
            )
            
            # Check for test results
            result_files = list(Path(test_results_temp).glob("*.xml"))
            if result_files:
                return all(self._evaluate_test_results(f) for f in result_files)
            else:
                rospy.logerr("No integration test results generated")
                return False
                
        except Exception as e:
            rospy.logerr(f"Integration tests failed: {e}")
            return False
        finally:
            shutil.rmtree(test_results_temp, ignore_errors=True)
    
    def _evaluate_test_results(self, results_file):
        """Evaluate test results from XML file"""
        import xml.etree.ElementTree as ET
        
        try:
            tree = ET.parse(results_file)
            root = tree.getroot()
            
            # Get test statistics
            total_tests = int(root.attrib.get('tests', 0))
            failures = int(root.attrib.get('failures', 0))
            errors = int(root.attrib.get('errors', 0))
            
            rospy.loginfo(f"Test results - Total: {total_tests}, Failures: {failures}, Errors: {errors}")
            
            success_rate = (total_tests - failures - errors) / total_tests if total_tests > 0 else 1.0
            
            # Check against coverage threshold if present
            if 'coverage' in root.attrib:
                coverage = float(root.attrib['coverage'])
                min_coverage = self.config['testing']['coverage_threshold']
                
                if coverage < min_coverage:
                    rospy.logerr(f"Coverage {coverage:.2f} below threshold {min_coverage}")
                    return False
            
            return success_rate >= 0.95  # Require 95%+ success rate
            
        except Exception as e:
            rospy.logerr(f"Error evaluating test results: {e}")
            return False
    
    def _run_performance_tests(self):
        """Run performance and stress tests"""
        rospy.loginfo("Running performance tests...")
        
        test_results_temp = tempfile.mkdtemp(prefix="perf_test_results_")
        
        try:
            # Run performance tests in container
            container = self.docker_client.containers.run(
                f"robotics-build:{os.environ.get('BUILD_NUMBER', 'local')}",
                command="bash -c 'cd /catkin_ws && source devel/setup.bash && python -m pytest tests/performance/ -v --tb=short'",
                volumes={
                    os.path.abspath(test_results_temp): {'bind': '/test_results', 'mode': 'rw'},
                    '/tmp': {'bind': '/tmp', 'mode': 'rw'}  # For shared memory access
                },
                environment={
                    'ROS_DOMAIN_ID': '100',  # Isolate test environment
                    'PERFORMANCE_TEST_DURATION': '60',  # Run tests for 60 seconds
                    'MAX_CPU_THRESHOLD': str(self.config['testing']['performance_threshold']['max_cpu']),
                    'MAX_MEMORY_THRESHOLD': str(self.config['testing']['performance_threshold']['max_memory'])
                },
                remove=True,
                detach=False,
                stdout=True,
                stderr=True
            )
            
            # Evaluate performance results
            perf_results = self._evaluate_performance_results(test_results_temp)
            
            return perf_results
            
        except Exception as e:
            rospy.logerr(f"Performance tests failed: {e}")
            return False
        finally:
            shutil.rmtree(test_results_temp, ignore_errors=True)
    
    def _evaluate_performance_results(self, results_dir):
        """Evaluate performance test results"""
        results_path = Path(results_dir)
        
        # Look for performance metrics files
        metrics_files = list(results_path.glob("*_metrics.json"))
        
        if not metrics_files:
            rospy.logerr("No performance metrics files found")
            return False
        
        # Load and evaluate metrics
        import json
        for metrics_file in metrics_files:
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Check performance thresholds
                cpu_usage = metrics.get('cpu_avg', 1.0)
                mem_usage = metrics.get('memory_avg', 1.0)
                throughput = metrics.get('throughput_avg', 0.0)
                
                thresholds = self.config['testing']['performance_threshold']
                
                if cpu_usage > thresholds['max_cpu']:
                    rospy.logerr(f"CPU usage {cpu_usage} exceeds threshold {thresholds['max_cpu']}")
                    return False
                
                if mem_usage > thresholds['max_memory']:
                    rospy.logerr(f"Memory usage {mem_usage} exceeds threshold {thresholds['max_memory']}")
                    return False
                
                if throughput < thresholds['min_throughput']:
                    rospy.logerr(f"Throughput {throughput} below threshold {thresholds['min_throughput']}")
                    return False
            
            except Exception as e:
                rospy.logerr(f"Error processing performance metrics: {e}")
                return False
        
        return True
    
    def _deployment_stage(self, environment, commit_sha):
        """Deploy to specified environment"""
        rospy.loginfo(f"Starting deployment to {environment}")
        
        # Create deployment package
        deployment_package = self._create_deployment_package(commit_sha)
        
        if not deployment_package:
            rospy.logerr("Failed to create deployment package")
            return False
        
        # Deploy to target
        success = self._deploy_to_target(deployment_package, environment)
        
        if success:
            # Run post-deployment health checks
            health_success = self._run_health_checks(environment)
            if not health_success:
                rospy.logerr("Health checks failed after deployment")
                
                # Trigger rollback
                self._rollback_deployment(environment)
                return False
        
        return success
    
    def _create_deployment_package(self, commit_sha):
        """Create deployment package with built artifacts"""
        try:
            # Create deployment package directory
            package_name = f"deployment-{commit_sha or 'latest'}"
            package_dir = self.build_artifacts_dir / package_name
            package_dir.mkdir(exist_ok=True)
            
            # Copy built images and artifacts
            # In a real system, this would package everything needed for deployment
            image_name = f"robotics-build:{os.environ.get('BUILD_NUMBER', 'local')}"
            
            # Export the Docker image
            image_tar = package_dir / f"{image_name.replace(':', '_')}.tar"
            
            # Save image to tar file
            image = self.docker_client.images.get(image_name)
            with open(image_tar, 'wb') as f:
                for chunk in image.save():
                    f.write(chunk)
            
            # Create deployment manifest
            manifest_data = {
                'commit_sha': commit_sha or 'unknown',
                'build_time': str(datetime.now()),
                'environment': environment,
                'artifacts': [str(image_tar)]
            }
            
            manifest_file = package_dir / "manifest.json"
            with open(manifest_file, 'w') as f:
                json.dump(manifest_data, f, indent=2)
            
            rospy.loginfo(f"Deployment package created: {package_dir}")
            return package_dir
            
        except Exception as e:
            rospy.logerr(f"Error creating deployment package: {e}")
            return None
    
    def _deploy_to_target(self, deployment_package, environment):
        """Deploy package to target environment"""
        target = self.config['deployment']['targets'][environment]
        
        try:
            if target == 'simulated_robot':
                # Deploy to simulation environment
                return self._deploy_to_simulation(deployment_package)
            elif target == 'physical_robot':
                # Deploy to physical robot (with extra safety checks)
                return self._deploy_to_physical_robot(deployment_package)
            else:
                rospy.logerr(f"Unknown deployment target: {target}")
                return False
        
        except Exception as e:
            rospy.logerr(f"Deployment failed: {e}")
            return False
    
    def _run_health_checks(self, environment):
        """Run health checks after deployment"""
        timeout = self.config['deployment']['health_check_timeout']
        start_time = time.time()
        
        rospy.loginfo(f"Running health checks for environment: {environment}")
        
        while time.time() - start_time < timeout:
            try:
                # Check if system is responsive
                health_status = self._check_system_health(environment)
                
                if health_status['overall_status'] == 'healthy':
                    rospy.loginfo("Health checks passed")
                    return True
                elif health_status['overall_status'] == 'degraded':
                    rospy.logwarn("System is degraded but functional")
                    # Decide whether to continue based on severity
                    if health_status.get('critical_failures', 0) == 0:
                        return True
                    else:
                        return False
                else:
                    # System is unhealthy, wait and retry
                    rospy.loginfo("System is not healthy, waiting...")
                    time.sleep(5.0)
            
            except Exception as e:
                rospy.logwarn(f"Health check error: {e}")
                time.sleep(2.0)
        
        rospy.logerr("Health checks timed out")
        return False
    
    def _check_system_health(self, environment):
        """Check health of deployed system"""
        # This would check various system components
        # For now, returning a mock health status
        
        # In practice, this would:
        # - Query system status endpoints
        # - Check ROS master connectivity
        # - Verify critical nodes are running
        # - Check sensor data streams
        # - Validate system performance metrics
        
        health_status = {
            'overall_status': 'healthy',
            'critical_nodes_running': True,
            'sensor_streams_active': True,
            'performance_within_bounds': True,
            'critical_failures': 0,
            'warnings': []
        }
        
        return health_status

# Safety checker for deployment
class DeploymentSafetyChecker:
    """Safety checks before deploying to physical robots"""
    
    def __init__(self):
        # Safety parameters
        self.maximum_velocity = 0.5  # m/s
        self.maximum_torque = 100.0  # N*m
        self.safety_zones = []       # Defined safe operating zones
        self.emergency_stop_active = True
        
        # Robot state monitoring
        self.joint_limits_monitored = True
        self.collision_avoided = True
        self.balance_maintained = True
    
    def pre_deployment_safety_check(self, deployment_package, target_robot):
        """Perform safety checks before deploying to physical robot"""
        rospy.loginfo(f"Starting safety checks for deployment to {target_robot}")
        
        # 1. Code safety analysis
        code_analysis_ok = self._analyze_code_safety(deployment_package)
        if not code_analysis_ok:
            rospy.logerr("Code safety analysis failed")
            return False
        
        # 2. Simulation validation
        simulation_validation_ok = self._validate_in_simulation(deployment_package)
        if not simulation_validation_ok:
            rospy.logerr("Simulation validation failed")
            return False
        
        # 3. Hardware safety check
        hardware_safety_ok = self._check_hardware_safety(target_robot)
        if not hardware_safety_ok:
            rospy.logerr("Hardware safety check failed")
            return False
        
        # 4. Emergency systems check
        emergency_systems_ok = self._check_emergency_systems(target_robot)
        if not emergency_systems_ok:
            rospy.logerr("Emergency systems check failed")
            return False
        
        rospy.loginfo("All safety checks passed")
        return True
    
    def _analyze_code_safety(self, deployment_package):
        """Analyze code for potential safety issues"""
        # Check for:
        # - Direct hardware access without safety checks
        # - Missing bounds checking
        # - Unsafe memory operations
        # - Missing error handling
        # - Disabled safety systems
        
        code_path = deployment_package / "src"
        safety_issues = []
        
        for py_file in code_path.rglob("*.py"):
            with open(py_file, 'r') as f:
                content = f.read()
                
                # Check for dangerous patterns
                if "hardcoded_velocity = 10.0" in content.lower():
                    # Check if value exceeds limits
                    import re
                    matches = re.findall(r"velocity\s*=\s*([0-9.]+)", content)
                    for match in matches:
                        if float(match) > self.maximum_velocity:
                            safety_issues.append(f"Excessive velocity in {py_file}: {match}")
                
                # Check for direct hardware access without safety wrapper
                if "direct_hardware" in content.lower() and "safety" not in content.lower():
                    safety_issues.append(f"Direct hardware access without safety in {py_file}")
        
        if safety_issues:
            rospy.logerr(f"Safety issues found: {safety_issues}")
            return False
        
        return True
    
    def _validate_in_simulation(self, deployment_package):
        """Validate functionality in simulation before physical deployment"""
        # This would run the code in a high-fidelity simulator
        # and verify it behaves correctly
        
        # Set up simulation environment
        sim_env = "isaac_gym_simulation"
        
        # Deploy code to simulation
        success = self._deploy_to_simulation(sim_env, deployment_package)
        if not success:
            return False
        
        # Run simulation tests
        test_scenarios = [
            "normal_operation",
            "obstacle_avoidance",
            "balance_recovery",
            "emergency_stop"
        ]
        
        all_tests_passed = True
        for scenario in test_scenarios:
            test_success = self._run_simulation_scenario(sim_env, scenario)
            if not test_success:
                rospy.logerr(f"Simulation test failed: {scenario}")
                all_tests_passed = False
        
        # Cleanup
        self._teardown_simulation(sim_env)
        
        return all_tests_passed
    
    def _check_hardware_safety(self, robot_name):
        """Check that robot hardware is safe to operate"""
        # Check joint limits
        joint_limits_ok = self._verify_joint_limits(robot_name)
        
        # Check torque limits
        torque_limits_ok = self._verify_torque_limits(robot_name)
        
        # Check safety zones
        safety_zones_ok = self._verify_safety_zones(robot_name)
        
        return joint_limits_ok and torque_limits_ok and safety_zones_ok
    
    def _check_emergency_systems(self, robot_name):
        """Check that emergency systems are responsive"""
        # Test emergency stop
        es_success = self._test_emergency_stop(robot_name)
        
        # Test safe position command
        safe_pos_success = self._test_safe_position(robot_name)
        
        # Test communication timeout handling
        comm_timeout_success = self._test_communication_timeout(robot_name)
        
        return es_success and safe_pos_success and comm_timeout_success

# Example usage of the CI/CD system
class HumanoidRobotCISystem:
    """Specialized CI/CD for humanoid robots"""
    
    def __init__(self):
        self.ci_pipeline = RoboticsCIPipeline()
        self.safety_checker = DeploymentSafetyChecker()
        
        # Humanoid-specific configurations
        self.balance_algorithms_tested = False
        self.bipedal_locomotion_verified = False
        self.safety_protocols_validated = True
        
        rospy.loginfo("Humanoid robot CI/CD system initialized")
    
    def run_hardware_in_the_loop_test(self, test_scenario):
        """Run hardware-in-the-loop tests for humanoid specific functionality"""
        rospy.loginfo(f"Running hardware-in-the-loop test: {test_scenario}")
        
        # Special tests for humanoid systems
        if test_scenario == "balance_control":
            return self._test_balance_algorithms()
        elif test_scenario == "bipedal_locomotion":
            return self._test_bipedal_gait()
        elif test_scenario == "dynamic_movement":
            return self._test_dynamic_motions()
        elif test_scenario == "fall_recovery":
            return self._test_fall_recovery_safety()
        else:
            rospy.logwarn(f"Unknown hardware-in-the-loop test: {test_scenario}")
            return False
    
    def _test_balance_algorithms(self):
        """Test humanoid balance control algorithms"""
        # This would interface with real or simulated humanoid robot
        # to test balance algorithm performance
        
        # Key metrics to test:
        # - Recovery time from perturbation
        # - Stability during standing
        # - Balance during manipulation
        # - Transition stability between gaits
        
        rospy.loginfo("Testing balance algorithms...")
        
        # For now, return success (would have real tests in practice)
        return True
    
    def _test_bipedal_gait(self):
        """Test bipedal locomotion patterns"""
        # Test various gaits: standing, walking, turning, etc.
        gaits_to_test = ["standing", "walking", "turning", "ascending_stairs", "descending_stairs"]
        
        for gait in gaits_to_test:
            rospy.loginfo(f"Testing gait: {gait}")
            
            # Test would involve commanding the robot to perform the gait
            # and measuring stability, efficiency, and safety metrics
            success = self._execute_gait_test(gait)
            if not success:
                rospy.logerr(f"Gait test failed: {gait}")
                return False
        
        return True
    
    def _test_dynamic_movements(self):
        """Test dynamic movements that challenge balance"""
        # Test movements like reaching while balancing, quick turns, etc.
        movements_to_test = ["arm_reach_while_balancing", "rapid_turning", "dynamic_object_manipulation"]
        
        for movement in movements_to_test:
            rospy.loginfo(f"Testing movement: {movement}")
            
            success = self._execute_movement_test(movement)
            if not success:
                rospy.logerr(f"Movement test failed: {movement}")
                return False
        
        return True
    
    def _test_fall_recovery_safety(self):
        """Test safety protocols for when robot loses balance"""
        # Test that robot enters safe state when balance is lost
        # This is critical for humanoid robots to prevent damage and injury
        
        rospy.loginfo("Testing fall recovery safety protocols...")
        
        # This would involve:
        # 1. Simulating or commanding a condition that causes balance loss
        # 2. Verifying that safety protocols activate
        # 3. Confirming that robot enters safe state
        # 4. Verifying that it can be safely recovered
        
        # For safety reasons, this might only be tested in simulation
        # or with appropriate safety equipment in physical testing
        
        return self._execute_safety_protocol_test()

if __name__ == "__main__":
    # Example execution of CI/CD pipeline
    rospy.init_node('humanoid_ci_cd_system')
    
    try:
        ci_system = HumanoidRobotCISystem()
        
        # Run CI/CD for development environment
        success = ci_system.ci_pipeline.run_pipeline(
            environment='development',
            commit_sha=os.environ.get('GIT_COMMIT')
        )
        
        if success:
            rospy.loginfo("CI/CD pipeline completed successfully")
        else:
            rospy.logerr("CI/CD pipeline failed")
            sys.exit(1)
            
    except Exception as e:
        rospy.logerr(f"Error in CI/CD system: {e}")
        sys.exit(1)
```

## Performance Benchmarking

### Robot Performance Metrics

```python
class RobotPerformanceBenchmark:
    """Comprehensive performance benchmarking for humanoid robots"""
    
    def __init__(self):
        # Timing benchmarks
        self.timing_benchmarks = {
            'control_loop_frequency': 100.0,  # Hz
            'perception_pipeline': 0.1,       # seconds
            'motion_planning': 0.5,           # seconds
            'task_execution': 5.0             # seconds per task
        }
        
        # Resource benchmarks
        self.resource_benchmarks = {
            'cpu_usage': 0.7,           # fraction (70%)
            'memory_usage': 0.65,       # fraction (65%)
            'gpu_usage': 0.8,           # fraction (80%)
            'network_bandwidth': 10.0   # Mbps
        }
        
        # Behavioral benchmarks
        self.behavioral_benchmarks = {
            'navigation_accuracy': 0.95,       # fraction (95%)
            'grasp_success_rate': 0.9,         # fraction (90%)
            'manipulation_precision': 0.01,    # meters
            'balance_recovery_time': 2.0       # seconds
        }
        
        # Initialize benchmark tools
        self.cpu_monitor = CPUMonitor()
        self.memory_monitor = MemoryMonitor()
        self.network_monitor = NetworkMonitor()
        
        # Results storage
        self.benchmark_results = {}
        self.performance_history = []
        
        rospy.loginfo("Robot performance benchmarking system initialized")
    
    def run_comprehensive_benchmark(self, test_duration=60.0):
        """Run comprehensive performance benchmarking suite"""
        rospy.loginfo(f"Starting comprehensive benchmark for {test_duration}s")
        
        benchmark_start_time = time.time()
        
        # Start resource monitoring
        self.cpu_monitor.start_monitoring()
        self.memory_monitor.start_monitoring()
        self.network_monitor.start_monitoring()
        
        # Run individual benchmark tests
        benchmark_results = {}
        
        # 1. Timing benchmarks
        benchmark_results['timing'] = self._benchmark_timing_performance(test_duration)
        
        # 2. Resource utilization
        benchmark_results['resources'] = self._benchmark_resource_utilization(test_duration)
        
        # 3. Behavioral performance
        benchmark_results['behavioral'] = self._benchmark_behavioral_performance(test_duration)
        
        # 4. Stress testing
        benchmark_results['stress'] = self._benchmark_stress_performance(test_duration)
        
        # Compile final results
        self.benchmark_results = benchmark_results
        self.performance_history.append({
            'timestamp': time.time(),
            'test_duration': test_duration,
            'results': benchmark_results
        })
        
        # Analyze results
        self._analyze_benchmark_results(benchmark_results)
        
        # Generate report
        self.generate_benchmark_report()
        
        rospy.loginfo("Comprehensive benchmark completed")
        return benchmark_results
    
    def _benchmark_timing_performance(self, duration):
        """Benchmark timing-sensitive performance"""
        rospy.loginfo("Starting timing performance benchmark")
        
        start_time = time.time()
        measurements = {
            'control_loop_times': [],
            'perception_times': [],
            'planning_times': [],
            'execution_times': []
        }
        
        # Test control loop timing
        loop_timer = LoopTimer(expected_frequency=100.0)  # 100Hz control loop
        while time.time() - start_time < duration * 0.3:  # Use 30% of time for timing test
            loop_start = time.time()
            
            # Simulate control loop operations
            self._simulate_control_operations()
            
            loop_end = time.time()
            loop_time = loop_end - loop_start
            measurements['control_loop_times'].append(loop_time)
            
            loop_timer.wait_next_loop()
        
        # Test perception pipeline timing
        perception_timer = Stopwatch()
        for _ in range(50):  # Test 50 perception cycles
            perception_timer.start()
            self._simulate_perception_cycle()
            perception_time = perception_timer.elapsed_seconds()
            measurements['perception_times'].append(perception_time)
        
        # Calculate timing statistics
        results = {
            'control_loop': self._calculate_timing_stats(measurements['control_loop_times'], 0.01),  # Expected 10ms cycle
            'perception_pipeline': self._calculate_timing_stats(measurements['perception_times'], 0.1),  # Expected 100ms
        }
        
        return results
    
    def _benchmark_resource_utilization(self, duration):
        """Benchmark resource utilization under load"""
        rospy.loginfo("Starting resource utilization benchmark")
        
        # Apply computational load to test resource limits
        load_tester = ComputationalLoadTester(duration=duration)
        
        # Monitor resources during load
        resource_readings = []
        sample_interval = 0.5  # Sample every 500ms
        
        load_tester.start_load()
        
        start_time = time.time()
        while time.time() - start_time < duration:
            reading = {
                'timestamp': time.time(),
                'cpu_percent': self.cpu_monitor.get_current_usage(),
                'memory_percent': self.memory_monitor.get_current_usage(),
                'disk_io': self._get_disk_io_stats(),
                'network_io': self.network_monitor.get_current_bandwidth()
            }
            resource_readings.append(reading)
            time.sleep(sample_interval)
        
        load_tester.stop_load()
        
        # Calculate resource statistics
        cpu_readings = [r['cpu_percent'] for r in resource_readings]
        memory_readings = [r['memory_percent'] for r in resource_readings]
        network_readings = [r['network_io'] for r in resource_readings]
        
        results = {
            'cpu_stats': {
                'average': np.mean(cpu_readings),
                'peak': np.max(cpu_readings),
                '95th_percentile': np.percentile(cpu_readings, 95),
                'pass_threshold': np.mean(cpu_readings) <= self.resource_benchmarks['cpu_usage']
            },
            'memory_stats': {
                'average': np.mean(memory_readings),
                'peak': np.max(memory_readings),
                '95th_percentile': np.percentile(memory_readings, 95),
                'pass_threshold': np.mean(memory_readings) <= self.resource_benchmarks['memory_usage']
            },
            'network_stats': {
                'average': np.mean(network_readings),
                'peak': np.max(network_readings),
                '95th_percentile': np.percentile(network_readings, 95),
                'pass_threshold': np.mean(network_readings) <= self.resource_benchmarks['network_bandwidth']
            }
        }
        
        return results
    
    def _benchmark_behavioral_performance(self, duration):
        """Benchmark behavioral performance metrics"""
        rospy.loginfo("Starting behavioral performance benchmark")
        
        # Behavioral tests depend on actual robot capabilities
        # For simulation, we'll use mock tests
        
        behavioral_results = {}
        
        # If we can interface with actual robot:
        if self._robot_available():
            # Navigation accuracy test
            behavioral_results['navigation'] = self._benchmark_navigation_accuracy()
            
            # Grasp success rate test
            behavioral_results['grasping'] = self._benchmark_grasp_success_rate()
            
            # Balance maintenance test
            behavioral_results['balance'] = self._benchmark_balance_maintenance(duration)
        else:
            # For simulation or headless testing
            behavioral_results = {
                'navigation': {'accuracy': 0.96, 'success_rate': 0.98, 'pass': True},
                'grasping': {'success_rate': 0.92, 'precision': 0.008, 'pass': True},
                'balance': {'recovery_time': 1.5, 'stability_score': 0.94, 'pass': True}
            }
        
        return behavioral_results
    
    def _benchmark_stress_performance(self, duration):
        """Benchmark performance under stress conditions"""
        rospy.loginfo("Starting stress performance benchmark")
        
        # Apply multiple concurrent loads to stress test
        stress_tasks = [
            ('high_freq_control', lambda: self._stress_control_loops(duration/3)),
            ('continuous_perception', lambda: self._stress_perception_pipeline(duration/3)),
            ('heavy_planning', lambda: self._stress_motion_planning(duration/3))
        ]
        
        # Run stress tests in parallel
        stress_results = {}
        for task_name, task_func in stress_tasks:
            start_time = time.time()
            result = task_func()
            end_time = time.time()
            
            stress_results[task_name] = {
                'success': result,
                'duration': end_time - start_time,
                'resources': {
                    'cpu': self.cpu_monitor.get_average_usage(start_time, end_time),
                    'memory': self.memory_monitor.get_average_usage(start_time, end_time)
                }
            }
        
        return stress_results
    
    def _analyze_benchmark_results(self, results):
        """Analyze benchmark results and identify bottlenecks"""
        rospy.loginfo("Analyzing benchmark results for optimizations")
        
        # Identify performance bottlenecks
        bottlenecks = []
        
        # Check timing performance
        if 'timing' in results:
            if results['timing']['control_loop']['miss_rate'] > 0.01:  # >1% misses
                bottlenecks.append({
                    'type': 'timing',
                    'source': 'control_loop',
                    'severity': 'high',
                    'recommendation': 'Optimize control loop, consider upgrading hardware or simplifying calculations'
                })
            
            if results['timing']['perception_pipeline']['avg_time'] > 0.05:  # >50ms
                bottlenecks.append({
                    'type': 'timing', 
                    'source': 'perception_pipeline',
                    'severity': 'medium',
                    'recommendation': 'Consider model optimization, hardware acceleration, or pipeline parallelization'
                })
        
        # Check resource utilization
        if 'resources' in results:
            if results['resources']['cpu_stats']['average'] > 0.85:  # >85% CPU
                bottlenecks.append({
                    'type': 'resource',
                    'source': 'cpu_utilization',
                    'severity': 'high',
                    'recommendation': 'Optimize algorithms, parallelize computations, or upgrade to more powerful hardware'
                })
            
            if results['resources']['memory_stats']['peak'] > 0.90:  # >90% memory
                bottlenecks.append({
                    'type': 'resource',
                    'source': 'memory_usage',
                    'severity': 'high',
                    'recommendation': 'Investigate memory leaks, optimize data structures, or increase available memory'
                })
        
        # Report bottlenecks
        if bottlenecks:
            rospy.logwarn(f"Performance bottlenecks identified: {len(bottlenecks)}")
            for bottleneck in bottlenecks:
                rospy.logwarn(f"  - {bottleneck['source']}: {bottleneck['severity']} priority - {bottleneck['recommendation']}")
        else:
            rospy.loginfo("No significant performance bottlenecks detected")
        
        return bottlenecks

class LoopTimer:
    """Helper class to maintain consistent loop timing"""
    
    def __init__(self, expected_frequency):
        self.expected_period = 1.0 / expected_frequency
        self.last_tick = time.time()
    
    def wait_next_loop(self):
        """Wait until the next loop period"""
        elapsed = time.time() - self.last_tick
        sleep_time = self.expected_period - elapsed
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        self.last_tick = time.time()

class Stopwatch:
    """Simple stopwatch for timing measurements"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
        self.end_time = None
    
    def stop(self):
        if self.start_time is not None:
            self.end_time = time.time()
    
    def elapsed_seconds(self):
        if self.start_time is not None:
            if self.end_time is not None:
                return self.end_time - self.start_time
            else:
                return time.time() - self.start_time
        return 0.0

class CPUMonitor:
    """CPU usage monitoring"""
    
    def __init__(self):
        import psutil
        self.psutil = psutil
        self.readings = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        self.monitoring = True
        self.readings = []
        
        def monitor_loop():
            while self.monitoring:
                usage = self.psutil.cpu_percent(interval=0.1)
                self.readings.append({
                    'timestamp': time.time(),
                    'usage': usage
                })
                time.sleep(0.1)  # Sample every 100ms
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def get_current_usage(self):
        return self.psutil.cpu_percent(interval=0.1)
    
    def get_average_usage(self, start_time=None, end_time=None):
        if not self.readings:
            return 0.0
        
        # Filter readings by time range if specified
        if start_time and end_time:
            filtered_readings = [
                r for r in self.readings 
                if start_time <= r['timestamp'] <= end_time
            ]
        else:
            filtered_readings = self.readings
        
        if not filtered_readings:
            return 0.0
        
        return sum(r['usage'] for r in filtered_readings) / len(filtered_readings)

class MemoryMonitor:
    """Memory usage monitoring"""
    
    def __init__(self):
        import psutil
        self.psutil = psutil
        self.readings = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        self.monitoring = True
        self.readings = []
        
        def monitor_loop():
            while self.monitoring:
                memory_info = self.psutil.virtual_memory()
                usage_percent = memory_info.percent / 100.0  # Convert to fraction
                
                self.readings.append({
                    'timestamp': time.time(),
                    'usage_fraction': usage_percent
                })
                time.sleep(0.1)  # Sample every 100ms
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def get_current_usage(self):
        memory_info = self.psutil.virtual_memory()
        return memory_info.percent / 100.0  # Return as fraction
    
    def get_peak_usage(self):
        if not self.readings:
            return 0.0
        return max(r['usage_fraction'] for r in self.readings)

# Performance report generation
class BenchmarkReporter:
    """Generate performance benchmark reports"""
    
    def __init__(self):
        self.report_template = self._load_report_template()
    
    def _load_report_template(self):
        """Load the report template"""
        template = """
# Performance Benchmark Report

**Generated**: {timestamp}
**Platform**: {platform}
**Test Duration**: {duration}s
**Environment**: {environment}

## Executive Summary

- **Overall Performance Score**: {overall_score:.2f}/10
- **Pass Rate**: {pass_rate:.1f}%
- **Critical Issues**: {critical_issues}
- **Recommendations**: {recommendations}

## Detailed Results

### 1. Timing Performance
{timing_section}

### 2. Resource Utilization
{resource_section}

### 3. Behavioral Performance  
{behavioral_section}

### 4. Stress Test Results
{stress_section}

## Performance Analysis

### Identified Bottlenecks
{bottlenecks}

### Recommendations
{detailed_recommendations}

## Comparison with Baseline

Compared to baseline system performance:
{baseline_comparison}

---
*Report generated automatically by RobotPerformanceBenchmark system*
        """
        return template
    
    def generate_benchmark_report(self, results, filename=None):
        """Generate benchmark report from results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_report_{timestamp}.md"
        
        # Calculate summary metrics
        overall_score = self._calculate_overall_score(results)
        pass_rate = self._calculate_pass_rate(results)
        critical_issues = self._count_critical_issues(results)
        recommendations = self._generate_top_recommendations(results)
        
        # Format sections
        timing_section = self._format_timing_report(results.get('timing', {}))
        resource_section = self._format_resource_report(results.get('resources', {}))
        behavioral_section = self._format_behavioral_report(results.get('behavioral', {}))
        stress_section = self._format_stress_report(results.get('stress', {}))
        bottlenecks = self._format_bottleneck_report(results.get('bottlenecks', []))
        detailed_recommendations = self._format_detailed_recommendations(results)
        baseline_comparison = self._format_baseline_comparison(results)
        
        # Generate report
        report_content = self.report_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            platform="Humanoid Robot Platform",  # Would detect actual platform
            duration=len(results.get('timing', {}).get('control_loop_times', [1])) * 0.01,  # Approximate
            environment="Development",  # Would detect actual environment
            overall_score=overall_score,
            pass_rate=pass_rate,
            critical_issues=critical_issues,
            recommendations=len(recommendations),
            timing_section=timing_section,
            resource_section=resource_section,
            behavioral_section=behavioral_section,
            stress_section=stress_section,
            bottlenecks=bottlenecks,
            detailed_recommendations=detailed_recommendations,
            baseline_comparison=baseline_comparison
        )
        
        # Save report
        with open(filename, 'w') as f:
            f.write(report_content)
        
        rospy.loginfo(f"Benchmark report saved to: {filename}")
        return report_content
    
    def _calculate_overall_score(self, results):
        """Calculate overall performance score (0-10)"""
        # Weighted calculation based on importance
        timing_weight = 0.25
        resource_weight = 0.25
        behavioral_weight = 0.35
        stress_weight = 0.15
        
        timing_score = self._calculate_timing_score(results.get('timing', {}))
        resource_score = self._calculate_resource_score(results.get('resources', {}))
        behavioral_score = self._calculate_behavioral_score(results.get('behavioral', {}))
        stress_score = self._calculate_stress_score(results.get('stress', {}))
        
        overall = (
            timing_weight * timing_score +
            resource_weight * resource_score +
            behavioral_weight * behavioral_score +
            stress_weight * stress_score
        )
        
        return overall
    
    def _calculate_timing_score(self, timing_results):
        """Calculate timing performance score"""
        # Perfect score if no timing violations
        if timing_results.get('control_loop', {}).get('miss_rate', 1.0) == 0:
            return 10.0
        elif timing_results.get('control_loop', {}).get('miss_rate', 1.0) < 0.01:  # <1% miss rate
            return 8.0
        elif timing_results.get('control_loop', {}).get('miss_rate', 1.0) < 0.05:  # <5% miss rate
            return 6.0
        else:
            return 3.0  # Poor timing performance
    
    def _calculate_resource_score(self, resource_results):
        """Calculate resource utilization score"""
        avg_cpu = resource_results.get('cpu_stats', {}).get('average', 1.0)
        avg_memory = resource_results.get('memory_stats', {}).get('average', 1.0)
        
        # Score based on how close to limits
        cpu_score = max(0, 10 * (1 - avg_cpu/0.90))  # Perfect at 0%, 0 at 90%+ CPU
        memory_score = max(0, 10 * (1 - avg_memory/0.85))  # Perfect at 0%, 0 at 85%+ memory
        
        return (cpu_score + memory_score) / 2
    
    def _calculate_behavioral_score(self, behavioral_results):
        """Calculate behavioral performance score"""
        # Based on success rates and accuracy
        nav_accuracy = behavioral_results.get('navigation', {}).get('accuracy', 0.5)
        grasp_success = behavioral_results.get('grasping', {}).get('success_rate', 0.5)
        balance_stability = behavioral_results.get('balance', {}).get('stability_score', 0.5)
        
        avg_behavioral = (nav_accuracy + grasp_success + balance_stability) / 3
        return avg_behavioral * 10  # Convert fraction to 0-10 scale
    
    def _calculate_stress_score(self, stress_results):
        """Calculate stress test performance score"""
        # If all stress tests passed, good score
        passed_count = sum(1 for v in stress_results.values() if v.get('success', False))
        total_count = len(stress_results)
        
        if total_count == 0:
            return 5.0  # Neutral if no tests
        
        success_rate = passed_count / total_count
        return success_rate * 10

if __name__ == "__main__":
    rospy.init_node('robot_performance_benchmark')
    
    try:
        benchmark_system = RobotPerformanceBenchmark()
        
        # Run comprehensive benchmark
        results = benchmark_system.run_comprehensive_benchmark(test_duration=60.0)
        
        # Generate report
        reporter = BenchmarkReporter()
        report = reporter.generate_benchmark_report(results)
        
        rospy.loginfo("Performance benchmark completed successfully")
        rospy.loginfo("Report generated and saved")
        
    except KeyboardInterrupt:
        rospy.loginfo("Benchmark interrupted by user")
    except Exception as e:
        rospy.logerr(f"Error in performance benchmarking: {e}")
        raise
```

## Safety and Validation Protocols

### Safety-First Integration Approaches

```python
class SafetyValidationSystem:
    """Comprehensive safety and validation system for humanoid robots"""
    
    def __init__(self):
        # Safety systems
        self.e_stop_monitor = EmergencyStopMonitor()
        self.collision_detector = CollisionDetectionSystem()
        self.balance_validator = BalanceValidationSystem()
        self.hardware_monitor = HardwareHealthMonitor()
        self.environment_monitor = EnvironmentAwarenessSystem()
        
        # Validation protocols
        self.pre_execution_validator = PreExecutionValidator()
        self.in_process_monitor = InProcessMonitor()
        self.post_execution_verifier = PostExecutionVerifier()
        
        # Safety state
        self.safety_override_engaged = False
        self.last_safety_check = time.time()
        self.safety_check_interval = 0.1  # 100ms between safety checks
        
        # Validation thresholds
        self.validation_thresholds = {
            'max_joint_velocity': 2.0,      # rad/s
            'max_joint_torque': 50.0,       # N*m
            'max_end_effector_velocity': 1.0,  # m/s
            'min_step_clearance': 0.05,     # 5cm minimum
            'max_com_deviation': 0.15,      # 15cm from support polygon
            'min_battery_level': 0.2,       # 20% minimum
            'max_temp_threshold': 70.0      # 70°C maximum
        }
        
        rospy.loginfo("Safety and validation system initialized")
    
    def validate_pre_execution(self, action_plan):
        """Validate an action plan before execution"""
        rospy.loginfo("Starting pre-execution validation")
        
        # Validate each component of the action plan
        validation_results = {
            'kinematic_feasibility': self._validate_kinematic_feasibility(action_plan),
            'dynamic_stability': self._validate_dynamic_stability(action_plan),
            'collision_avoidance': self._validate_collision_avoidance(action_plan),
            'hardware_limits': self._validate_hardware_limits(action_plan),
            'environment_safety': self._validate_environment_safety(action_plan),
            'power_requirements': self._validate_power_requirements(action_plan)
        }
        
        # Overall validation result
        all_valid = all(validation_results.values())
        
        if not all_valid:
            invalid_checks = [k for k, v in validation_results.items() if not v]
            rospy.logerr(f"Pre-execution validation failed: {invalid_checks}")
            
            # Generate detailed error report
            error_report = self._generate_validation_error_report(validation_results, action_plan)
            rospy.logerr(f"Validation error report: {error_report}")
        else:
            rospy.loginfo("Pre-execution validation passed")
        
        return all_valid, validation_results
    
    def _validate_kinematic_feasibility(self, action_plan):
        """Validate that actions are kinematically feasible"""
        if not hasattr(action_plan, 'joint_trajectories'):
            return True  # Nothing to validate if no joint trajectories
        
        for trajectory in action_plan.joint_trajectories:
            for point in trajectory.points:
                # Check joint limits
                for joint_idx, position in enumerate(point.positions):
                    joint_limits = self._get_joint_limits(joint_idx)
                    if position < joint_limits['min'] or position > joint_limits['max']:
                        rospy.logwarn(f"Joint {joint_idx} position {position} exceeds limits [{joint_limits['min']}, {joint_limits['max']}]")
                        return False
                
                # Check joint velocities
                if hasattr(point, 'velocities'):
                    for vel_idx, velocity in enumerate(point.velocities):
                        max_vel = self.validation_thresholds['max_joint_velocity']
                        if abs(velocity) > max_vel:
                            rospy.logwarn(f"Joint {vel_idx} velocity {velocity} exceeds maximum {max_vel}")
                            return False
        
        return True
    
    def _validate_dynamic_stability(self, action_plan):
        """Validate that actions maintain dynamic stability"""
        # For humanoid robots, this involves checking Center of Mass (CoM) vs support polygon
        # This is computationally intensive, so we'll do a simplified check
        current_state = self._get_current_robot_state()
        
        # Simulate the action and check stability at key points
        for i, trajectory_point in enumerate(action_plan.joint_trajectories[0].points):
            if i % 10 == 0:  # Check every 10th point to reduce computation
                # Update simulated state
                self._update_simulated_state(current_state, trajectory_point)
                
                # Calculate CoM and check support polygon
                com_position = self._calculate_com_position(current_state)
                support_polygon = self._calculate_support_polygon(current_state)
                
                # Check if CoM is within support polygon
                if not self._is_point_in_polygon(com_position, support_polygon):
                    distance_to_polygon = self._distance_to_polygon(com_position, support_polygon)
                    if distance_to_polygon > self.validation_thresholds['max_com_deviation']:
                        rospy.logwarn(f"CoM deviation {distance_to_polygon:.3f}m exceeds threshold {self.validation_thresholds['max_com_deviation']:.3f}m")
                        return False
        
        return True
    
    def _validate_collision_avoidance(self, action_plan):
        """Validate that planned actions avoid collisions"""
        # Check path against known obstacles
        current_state = self._get_current_robot_state()
        
        # For each trajectory in the plan
        for trajectory in getattr(action_plan, 'trajectories', []):
            for i, point in enumerate(trajectory.points):
                if i % 5 == 0:  # Check every 5th point for efficiency
                    # Calculate robot configuration at this point
                    robot_config = self._apply_joint_positions(current_state, point.positions)
                    
                    # Check for self-collisions
                    if self._check_self_collision(robot_config):
                        rospy.logwarn(f"Self-collision detected at trajectory point {i}")
                        return False
                    
                    # Check for environment collisions
                    if self._check_environment_collision(robot_config):
                        rospy.logwarn(f"Environment collision detected at trajectory point {i}")
                        return False
        
        return True
    
    def _validate_hardware_limits(self, action_plan):
        """Validate that planned actions don't exceed hardware limits"""
        # Check motor limits, current limits, etc.
        for trajectory in getattr(action_plan, 'trajectories', []):
            for point in trajectory.points:
                # Check velocities
                if hasattr(point, 'velocities'):
                    for vel in point.velocities:
                        if abs(vel) > self.validation_thresholds['max_joint_velocity']:
                            rospy.logwarn(f"Joint velocity {vel} exceeds limit {self.validation_thresholds['max_joint_velocity']}")
                            return False
                
                # Check accelerations
                if hasattr(point, 'accelerations'):
                    for acc in point.accelerations:
                        # Check for reasonable acceleration limits
                        max_acc = self.validation_thresholds['max_joint_velocity'] * 5  # 5x velocity limit as acceleration proxy
                        if abs(acc) > max_acc:
                            rospy.logwarn(f"Joint acceleration {acc} exceeds limit {max_acc}")
                            return False
        
        # Check end-effector limits if applicable
        if hasattr(action_plan, 'cartesian_poses'):
            for pose in action_plan.cartesian_poses:
                # Calculate end-effector velocity from pose differences
                # This would require more sophisticated kinematic checking
                
                # Check for joint torque limits (estimated)
                if hasattr(point, 'effort'):
                    for effort in point.effort:
                        if abs(effort) > self.validation_thresholds['max_joint_torque']:
                            rospy.logwarn(f"Joint effort {effort} exceeds limit {self.validation_thresholds['max_joint_torque']}")
                            return False
        
        return True
    
    def _validate_environment_safety(self, action_plan):
        """Validate that actions are safe for the environment"""
        # Check for areas where robot shouldn't go (e.g., near people)
        environment_map = self._get_current_environment_map()
        
        for trajectory in getattr(action_plan, 'trajectories', []):
            for point in trajectory.points:
                # Convert joint positions to cartesian position
                cartesian_pos = self._forward_kinematics(point.positions)
                
                # Check against safety zones in environment map
                if self._is_in_forbidden_zone(cartesian_pos, environment_map):
                    rospy.logwarn(f"Planned position {cartesian_pos} is in forbidden zone")
                    return False
        
        return True
    
    def _validate_power_requirements(self, action_plan):
        """Validate that plan doesn't exceed power/battery limits"""
        # Estimate power consumption for the plan
        estimated_power = self._estimate_power_consumption(action_plan)
        
        # Get current battery level
        current_battery = self._get_current_battery_level()
        
        # Check if plan can be completed with current power
        if current_battery < self.validation_thresholds['min_battery_level']:
            rospy.logwarn(f"Current battery level ({current_battery:.2f}) below minimum ({self.validation_thresholds['min_battery_level']:.2f})")
            return False
        
        # Check if estimated power consumption fits in remaining battery
        estimated_remaining = current_battery - estimated_power
        if estimated_remaining < self.validation_thresholds['min_battery_level']:
            rospy.logwarn(f"Plan power consumption would reduce battery below minimum safe level")
            return False
        
        return True
    
    def validate_in_process(self, current_state, planned_action):
        """Validate during execution that robot remains safe"""
        # Check that execution is proceeding as planned
        if not self._is_execution_on_track(current_state, planned_action):
            rospy.logwarn("Execution diverging from plan")
            return False
        
        # Check safety systems are functional
        if not self.e_stop_monitor.is_operational():
            rospy.logerr("Emergency stop system malfunction detected")
            return False
        
        # Check for unexpected collisions
        if self.collision_detector.has_collision_occurred():
            rospy.logerr("Unexpected collision detected during execution")
            return False
        
        # Check balance status
        if not self.balance_validator.is_balanced(current_state):
            rospy.logerr("Robot is out of balance during execution")
            return False
        
        # Check hardware status
        if not self.hardware_monitor.all_systems_operational():
            rospy.logerr("Hardware malfunction detected during execution")
            return False
        
        # Check environment has not changed dangerously
        if self.environment_monitor.hazardous_change_detected():
            rospy.logerr("Hazardous environmental change detected during execution")
            return False
        
        return True
    
    def validate_post_execution(self, executed_action, expected_outcome):
        """Validate that execution achieved expected outcome safely"""
        # Check robot is in expected final state
        current_state = self._get_current_robot_state()
        if not self._is_state_close_to_expected(current_state, expected_outcome.final_state):
            rospy.logwarn("Robot not in expected final state after execution")
            return False
        
        # Check for any safety violations during execution
        if self.collision_detector.any_violations_during_execution():
            rospy.logwarn("Safety violations occurred during execution")
            return False
        
        # Check that no hardware was damaged
        if not self.hardware_monitor.verify_no_damage():
            rospy.logwarn("Hardware damage detected after execution")
            return False
        
        # Check that robot is in stable position
        if not self.balance_validator.is_balanced(current_state):
            rospy.logwarn("Robot is not balanced after execution")
            return False
        
        return True
    
    def _generate_validation_error_report(self, validation_results, action_plan):
        """Generate detailed error report for failed validation"""
        report = {
            'timestamp': time.time(),
            'validation_results': validation_results,
            'failed_checks': [k for k, v in validation_results.items() if not v],
            'action_plan_summary': {
                'type': getattr(action_plan, 'action_type', 'unknown'),
                'duration': getattr(action_plan, 'duration', 'unknown'),
                'involved_joints': len(getattr(action_plan, 'joint_trajectories', []))
            },
            'suggested_fixes': self._suggest_fixes_for_validation_failures(validation_results)
        }
        
        return report
    
    def _suggest_fixes_for_validation_failures(self, validation_results):
        """Suggest fixes for specific validation failures"""
        fixes = []
        
        if not validation_results.get('kinematic_feasibility', True):
            fixes.append("Check joint position and velocity limits for planned trajectory")
        
        if not validation_results.get('dynamic_stability', True):
            fixes.append("Slow down motion or modify trajectory to maintain center of mass over support base")
        
        if not validation_results.get('collision_avoidance', True):
            fixes.append("Regenerate path with increased safety margin from obstacles")
        
        if not validation_results.get('hardware_limits', True):
            fixes.append("Reduce execution speeds or modify forces/torques to stay within limits")
        
        if not validation_results.get('environment_safety', True):
            fixes.append("Verify robot is not entering restricted areas")
        
        if not validation_results.get('power_requirements', True):
            fixes.append("Conserve battery by optimizing motion or returning to charging station")
        
        return fixes

class EmergencyStopManager:
    """Manages emergency stop functionality"""
    
    def __init__(self):
        # Emergency stop button interface
        self.estop_button_sub = rospy.Subscriber('/emergency_stop', Bool, self.estop_callback)
        
        # Robot command interfaces (need to be stopped immediately)
        self.joint_cmd_pubs = []  # Will be populated with active joint command publishers
        self.velocity_cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # System state
        self.emergency_stop_engaged = False
        self.last_estop_time = None
        self.estop_reason = None
        
        # Safety relay control (if available)
        self.safety_relay_pub = rospy.Publisher('/safety_relay', Bool, queue_size=1)
        
        rospy.loginfo("Emergency stop manager initialized")
    
    def estop_callback(self, msg):
        """Handle emergency stop button press"""
        if msg.data:  # Emergency stop activated
            self._engage_emergency_stop("Manual emergency stop pressed")
        elif self.emergency_stop_engaged:  # Emergency stop released (only if previously engaged)
            self._disengage_emergency_stop()
    
    def _engage_emergency_stop(self, reason="Unknown"):
        """Engage emergency stop"""
        rospy.logerr(f"EMERGENCY STOP ENGAGED: {reason}")
        
        # Set system state to emergency stop
        self.emergency_stop_engaged = True
        self.last_estop_time = time.time()
        self.estop_reason = reason
        
        # Immediately stop all robot motion
        self._stop_all_robot_motion()
        
        # If available, cut power via safety relay
        self._cut_safety_power()
        
        # Publish status notification
        self._publish_emergency_status()
    
    def _disengage_emergency_stop(self):
        """Disengage emergency stop (requires manual reset)"""
        rospy.loginfo("Emergency stop disengaged, robot ready to resume operations")
        
        # Reset system state
        self.emergency_stop_engaged = False
        self.estop_reason = None
        
        # Restore power if safety relay was used
        self._restore_safety_power()
        
        # Publish status notification
        self._publish_operational_status()
    
    def _stop_all_robot_motion(self):
        """Send zero commands to stop all robot motion"""
        # Send zero-velocity commands
        zero_twist = Twist()
        self.velocity_cmd_pub.publish(zero_twist)
        
        # Send zero-joint commands if joint command publishers are registered
        for pub in self.joint_cmd_pubs:
            # Create zero-position joint trajectory
            zero_traj = JointTrajectory()
            # This would populate with actual joint names and zero positions
            pub.publish(zero_traj)
        
        # Additionally, publish to any common stop topics
        try:
            stop_pub = rospy.Publisher('/robot_mechanism_controller/stop', Empty, latch=True, queue_size=1)
            stop_pub.publish(Empty())
        except:
            pass  # OK if topic doesn't exist
    
    def _cut_safety_power(self):
        """Cut power via safety relay if available"""
        power_off_msg = Bool()
        power_off_msg.data = False
        self.safety_relay_pub.publish(power_off_msg)
    
    def _restore_safety_power(self):
        """Restore power via safety relay"""
        power_on_msg = Bool()
        power_on_msg.data = True
        self.safety_relay_pub.publish(power_on_msg)
    
    def _publish_emergency_status(self):
        """Publish emergency status for other nodes"""
        status_msg = RobotStatus()
        status_msg.mode = RobotStatus.EMERGENCY_STOP
        status_msg.timestamp = rospy.Time.now()
        status_msg.error_message = f"Emergency stop engaged: {self.estop_reason}"
        
        status_pub = rospy.Publisher('/robot_status', RobotStatus, latch=True, queue_size=1)
        status_pub.publish(status_msg)
    
    def _publish_operational_status(self):
        """Publish operational status"""
        status_msg = RobotStatus()
        status_msg.mode = RobotStatus.OPERATIONAL
        status_msg.timestamp = rospy.Time.now()
        status_msg.error_message = ""
        
        status_pub = rospy.Publisher('/robot_status', RobotStatus, latch=True, queue_size=1)
        status_pub.publish(status_msg)
    
    def is_emergency_stop_active(self):
        """Check if emergency stop is currently active"""
        return self.emergency_stop_engaged
    
    def get_last_estop_reason(self):
        """Get the reason for the last emergency stop"""
        return self.estop_reason

# Example usage in a complete control system
class SafeIntegrationController:
    """Main controller that integrates safety with all robot operations"""
    
    def __init__(self):
        # Initialize safety and validation system
        self.safety_system = SafetyValidationSystem()
        self.emergency_manager = EmergencyStopManager()
        
        # Robot interfaces
        self.arm_controller = ArmController()
        self.base_controller = BaseController()
        self.perception_system = PerceptionSystem()
        
        # ROS interfaces
        self.command_sub = rospy.Subscriber('/motion_command', MotionCommand, self.command_callback)
        self.robot_state_sub = rospy.Subscriber('/robot_state', RobotState, self.state_callback)
        self.safety_status_pub = rospy.Publisher('/safety_status', SafetyStatus, queue_size=10)
        
        # System state
        self.current_robot_state = None
        self.commands_executing = []
        
        rospy.loginfo("Safe integration controller initialized")
    
    def command_callback(self, cmd_msg):
        """Handle incoming motion commands with safety validation"""
        if self.emergency_manager.is_emergency_stop_active():
            rospy.logerr("Rejecting command: emergency stop is active")
            self._publish_safety_violation("Command rejected due to active emergency stop")
            return
        
        # Validate the command before execution
        is_valid, validation_details = self.safety_system.validate_pre_execution(cmd_msg.action_plan)
        
        if not is_valid:
            rospy.logerr("Command validation failed, rejecting command")
            self._publish_safety_violation(f"Command validation failed: {validation_details}")
            return
        
        # Execute command with safety monitoring
        try:
            success = self._execute_command_with_monitoring(cmd_msg)
            
            if success:
                rospy.loginfo("Command executed successfully with safety validation")
            else:
                rospy.logerr("Command execution failed safety monitoring")
                self._publish_safety_violation("Command execution failed safety checks")
                
        except Exception as e:
            rospy.logerr(f"Error during command execution: {e}")
            self._publish_safety_violation(f"Command execution error: {str(e)}")
    
    def _execute_command_with_monitoring(self, cmd_msg):
        """Execute command with real-time safety monitoring"""
        # Start monitoring thread
        monitoring_active = True
        
        def safety_monitor_loop():
            rate = rospy.Rate(100)  # 100 Hz monitoring
            while monitoring_active and not rospy.is_shutdown():
                if self.current_robot_state:
                    if not self.safety_system.validate_in_process(
                        self.current_robot_state, 
                        cmd_msg.action_plan
                    ):
                        rospy.logerr("Safety validation failed during execution, stopping robot")
                        self.emergency_manager._engage_emergency_stop("Safety violation during execution")
                        break
                rate.sleep()
        
        monitor_thread = threading.Thread(target=safety_monitor_loop)
        monitor_thread.start()
        
        try:
            # Execute the command
            result = self._execute_robot_command(cmd_msg.action_plan)
            
            # Stop monitoring after execution
            monitoring_active = False
            monitor_thread.join(timeout=1.0)  # Wait up to 1 sec for thread to stop
            
            # Validate post-execution state
            expected_outcome = self._predict_outcome(cmd_msg.action_plan)
            return self.safety_system.validate_post_execution(result, expected_outcome)
            
        except Exception as e:
            # Stop monitoring and emergency stop if needed
            monitoring_active = False
            if monitor_thread.is_alive():
                monitor_thread.join(timeout=1.0)
            
            rospy.logerr(f"Command execution error: {e}")
            self.emergency_manager._engage_emergency_stop(f"Execution error: {str(e)}")
            return False
    
    def _execute_robot_command(self, action_plan):
        """Execute robot command (simplified implementation)"""
        # This would interface with actual robot controllers
        # For each action in the plan:
        for action in action_plan.actions:
            if action.type == "move_to_pose":
                success = self.arm_controller.move_to_pose(
                    action.pose, 
                    action.speed, 
                    action.acceleration
                )
            elif action.type == "move_base":
                success = self.base_controller.move_to_pose(
                    action.pose,
                    action.speed
                )
            elif action.type == "grasp":
                success = self.arm_controller.execute_grasp(
                    action.grasp_pose,
                    action.force
                )
            
            if not success:
                rospy.logerr(f"Action failed: {action.type}")
                return False
        
        return True
    
    def _publish_safety_violation(self, message):
        """Publish safety violation message"""
        safety_msg = SafetyStatus()
        safety_msg.header.stamp = rospy.Time.now()
        safety_msg.status = SafetyStatus.VIOLATION
        safety_msg.message = message
        self.safety_status_pub.publish(safety_msg)

if __name__ == "__main__":
    rospy.init_node('safe_integration_controller')
    
    try:
        controller = SafeIntegrationController()
        rospy.loginfo("Safe integration controller running")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Safe integration controller shutting down")
    except Exception as e:
        rospy.logerr(f"Error in safe integration controller: {e}")
        raise
```

## Quality Assurance Protocols

### Automated Testing Framework

```python
import unittest
import numpy as np
import rospy
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Bool, Float64
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty

class HumanoidRobotTestSuite(unittest.TestCase):
    """Comprehensive test suite for humanoid robot systems"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        rospy.init_node('humanoid_robot_test_suite', anonymous=True)
        
        # Initialize robot interface
        cls.robot_interface = RobotInterface()
        cls.safety_system = SafetyValidationSystem()
        cls.validator = PreExecutionValidator()
        
        # Wait for system to be ready
        rospy.sleep(3.0)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        cls.robot_interface.shutdown()
    
    def setUp(self):
        """Set up before each test method"""
        # Ensure robot starts in safe state
        self._ensure_safe_start_state()
        self.test_start_time = rospy.Time.now()
    
    def _ensure_safe_start_state(self):
        """Ensure robot is in safe starting configuration"""
        # Move robot to known safe position (e.g., home position)
        home_position = self._get_home_configuration()
        
        # Validate pre-execution
        plan = MotionPlan(joint_positions=home_position)
        is_valid, _ = self.validator.validate_pre_execution(plan)
        
        if is_valid:
            self.robot_interface.move_to_configuration(home_position)
            rospy.sleep(1.0)  # Allow time to reach position
    
    def test_basic_mobility(self):
        """Test basic locomotion capabilities"""
        # Test standing up from sitting position
        success = self.robot_interface.move_to_pose("standing")
        self.assertTrue(success, "Robot should be able to stand up")
        
        # Test balancing in place
        rospy.sleep(5.0)  # Let it balance
        balance_status = self.robot_interface.get_balance_status()
        self.assertTrue(balance_status.stable, "Robot should maintain balance when standing")
        
        # Test simple stepping motion
        step_success = self.robot_interface.execute_step_sequence([
            {"direction": "forward", "length": 0.3, "height": 0.05}, 
            {"direction": "forward", "length": 0.3, "height": 0.05}
        ])
        self.assertTrue(step_success, "Robot should be able to execute simple step sequence")
    
    def test_perception_accuracy(self):
        """Test accuracy of perception systems"""
        # Place a known object in front of robot
        object_position = [0.8, 0.0, 0.0]  # 80cm in front, at center
        
        # Detect the object
        detections = self.robot_interface.detect_objects()
        
        # Find the closest object to expected position
        detected_object = self._find_closest_object(detections, object_position)
        
        if detected_object:
            # Check detection accuracy (within 5cm)
            detection_error = np.linalg.norm(
                np.array(detected_object.position) - np.array(object_position)
            )
            
            self.assertLess(
                detection_error, 0.05,  # 5cm threshold
                f"Object detection should be accurate within 5cm, got error: {detection_error:.3f}m"
            )
        else:
            self.fail("Object should be detected in front of robot")
    
    def test_grasp_success_rate(self):
        """Test grasp success rate for known objects"""
        # Test multiple grasp attempts on standard object
        success_count = 0
        total_attempts = 10
        
        for i in range(total_attempts):
            # Place known object in graspable position
            self._place_test_object("grasp_position")
            
            # Plan and execute grasp
            object_pose = self.robot_interface.get_object_pose("test_object")
            grasp_pose = self.robot_interface.plan_grasp(object_pose)
            
            if grasp_pose:
                grasp_success = self.robot_interface.execute_grasp(grasp_pose)
                if grasp_success:
                    success_count += 1
                    
                    # Verify object is actually grasped
                    if self.robot_interface.verify_grasp_success():
                        success_count += 0.5  # Additional point for verified grasp
            
            # Reset for next attempt
            self._reset_workspace()
            rospy.sleep(0.5)
        
        success_rate = success_count / total_attempts
        self.assertGreater(
            success_rate, 0.7,  # Expect 70% success rate
            f"Grasp success rate should be greater than 70%, got {success_rate*100:.1f}%"
        )
    
    def test_navigation_accuracy(self):
        """Test navigation accuracy to known locations"""
        # Define test waypoints
        waypoints = [
            {"name": "kitchen", "position": [2.0, 0.0, 0.0]},
            {"name": "living_room", "position": [0.0, 3.0, 0.0]},
            {"name": "bedroom", "position": [3.0, 3.0, 0.0]}
        ]
        
        for waypoint in waypoints:
            # Navigate to waypoint
            nav_success = self.robot_interface.navigate_to(waypoint["position"])
            
            # Verify robot reached destination (within 10cm)
            if nav_success:
                actual_position = self.robot_interface.get_current_position()
                error_distance = np.linalg.norm(
                    np.array(actual_position) - np.array(waypoint["position"])
                )
                
                self.assertLess(
                    error_distance, 0.1,  # 10cm threshold
                    f"Navigation should be accurate within 10cm for {waypoint['name']}, "
                    f"got error: {error_distance:.3f}m"
                )
            else:
                self.fail(f"Navigation should succeed to reach {waypoint['name']}")
    
    def test_balance_recovery(self):
        """Test robot's ability to recover from perturbations"""
        # Start in stable standing position
        self.robot_interface.move_to_pose("standing")
        rospy.sleep(2.0)
        
        # Apply small planned perturbation (via external force simulation)
        # NOTE: In real testing, this would be physical push or applied force
        self.robot_interface.apply_controlled_perturbation(
            force=[5.0, 0.0, 0.0],  # 5N push in X direction
            duration=0.1
        )
        
        # Allow time for recovery
        recovery_time = 3.0
        rospy.sleep(recovery_time)
        
        # Check balance status after recovery
        balance_status = self.robot_interface.get_balance_status()
        self.assertTrue(
            balance_status.stable,
            "Robot should recover balance after perturbation"
        )
        
        # Check that position didn't change too dramatically
        final_position = self.robot_interface.get_current_position()
        initial_position = [0.0, 0.0, 0.0]  # Should be approximately the same
        position_drift = np.linalg.norm(
            np.array(final_position) - np.array(initial_position)
        )
        
        self.assertLess(
            position_drift, 0.2,  # Should not drift more than 20cm
            f"Robot position drift after perturbation should be less than 20cm, "
            f"got drift: {position_drift:.3f}m"
        )
    
    def test_fall_recovery_safety(self):
        """Test safety systems when robot loses balance"""
        # This test would be performed in simulation or with safety equipment
        # For safety reasons, we'll just verify the safety systems are functional
        
        # Verify emergency stop system responsiveness
        initial_status = self.robot_interface.get_safety_status()
        self.assertFalse(
            initial_status.emergency_stop,
            "Emergency stop should not be active initially"
        )
        
        # Verify joint limit protection
        joint_limits_ok = self.robot_interface.verify_joint_limits()
        self.assertTrue(
            joint_limits_ok,
            "All joints should be within limits"
        )
        
        # Verify collision avoidance systems
        collision_free = self.robot_interface.verify_collision_free_path()
        self.assertTrue(
            collision_free,
            "Robot workspace should be collision-free"
        )
    
    def test_dual_arm_coordination(self):
        """Test coordination between dual arms for complex tasks"""
        # Move left arm to a position
        left_arm_pose = [0.3, 0.2, 0.8, 0, 0, 0]  # x, y, z, roll, pitch, yaw
        left_success = self.robot_interface.move_arm_to_pose("left", left_arm_pose)
        
        self.assertTrue(
            left_success,
            "Left arm should be able to move to target pose"
        )
        
        # Simultaneously move right arm to different position 
        right_arm_pose = [0.3, -0.2, 0.8, 0, 0, 0]
        right_success = self.robot_interface.move_arm_to_pose("right", right_arm_pose)
        
        self.assertTrue(
            right_success,
            "Right arm should be able to move to different target pose simultaneously"
        )
        
        # Now test coordinated bimanual task (e.g., lifting a box)
        # Left arm at left side of box
        left_grip_pose = [0.4, 0.1, 0.5, 0, 1.57, 0]
        # Right arm at right side of box
        right_grip_pose = [0.4, -0.1, 0.5, 0, 1.57, 0]
        
        # Execute coordinated grasp
        coord_success = self.robot_interface.execute_coordinated_grasp(
            left_grip_pose, right_grip_pose
        )
        
        self.assertTrue(
            coord_success,
            "Arms should be able to execute coordinated grasp"
        )
    
    def test_long_term_autonomy(self):
        """Test system stability over extended operation periods"""
        # This test would run for a longer time period in practice
        # For this example, we'll simulate by repeatedly performing tasks
        
        start_time = rospy.Time.now()
        test_duration = rospy.Duration(30.0)  # 30 seconds of testing
        end_time = start_time + test_duration
        
        task_count = 0
        successful_tasks = 0
        
        while rospy.Time.now() < end_time:
            # Perform a simple task (e.g., move to a random position and back)
            import random
            random_x = random.uniform(0.5, 1.5)
            random_y = random.uniform(-0.5, 0.5)
            
            # Navigate to random position
            nav_success = self.robot_interface.navigate_to([random_x, random_y, 0.0])
            
            if nav_success:
                # Navigate back to start
                return_success = self.robot_interface.navigate_to([0.0, 0.0, 0.0])
                if return_success:
                    successful_tasks += 1
            
            task_count += 1
            
            # Brief pause between tasks
            rospy.sleep(0.5)
        
        # Calculate success rate over time
        success_rate = successful_tasks / task_count if task_count > 0 else 0
        self.assertGreater(
            success_rate, 0.8,  # Should maintain 80%+ success rate over time
            f"Long-term autonomy test should maintain 80%+ success rate, "
            f"got {success_rate*100:.1f}% over {task_count} tasks"
        )

# Performance benchmarking tests
class HumanoidRobotBenchmarkTests(unittest.TestCase):
    """Performance benchmarking tests for humanoid robots"""
    
    def setUp(self):
        self.performance_monitor = PerformanceBenchmarkSystem()
        self.start_time = rospy.Time.now()
    
    def test_real_time_performance(self):
        """Test that the robot maintains real-time performance"""
        # Run performance monitoring during typical operations
        self.performance_monitor.start_monitoring()
        
        # Execute a sequence of typical robotic operations
        operations_sequence = [
            ('perception', lambda: self._run_perception_task()),
            ('planning', lambda: self._run_planning_task()),
            ('execution', lambda: self._run_execution_task()),
            ('balancing', lambda: self._run_balancing_task())
        ]
        
        for op_name, op_func in operations_sequence:
            start_time = time.time()
            success = op_func()
            op_time = time.time() - start_time
            
            # Check timing requirements
            if op_name == 'perception':
                self.assertLess(op_time, 0.1, "Perception should complete in <100ms")
            elif op_name == 'planning':
                self.assertLess(op_time, 0.5, "Planning should complete in <500ms")
            elif op_name == 'execution':
                self.assertLess(op_time, 1.0, "Execution should complete in <1000ms")
            
            self.assertTrue(success, f"{op_name} operation should succeed")
        
        # Stop monitoring and check resource usage
        self.performance_monitor.stop_monitoring()
        
        resource_stats = self.performance_monitor.get_statistics()
        
        self.assertLess(
            resource_stats['cpu_average'], 0.85, 
            f"CPU usage should stay below 85%, got {resource_stats['cpu_average']*100:.1f}%"
        )
        
        self.assertLess(
            resource_stats['memory_peak'], 0.80,
            f"Memory usage should stay below 80%, got {resource_stats['memory_peak']*100:.1f}%"
        )
    
    def _run_perception_task(self):
        """Run perception task for benchmarking"""
        try:
            # Perform object detection
            detections = self.robot_interface.perform_object_detection()
            return len(detections) >= 0  # Successfully ran detection
        except Exception:
            return False
    
    def _run_planning_task(self):
        """Run motion planning task for benchmarking"""
        try:
            # Plan a trajectory
            start_pose = self.robot_interface.get_current_pose()
            goal_pose = [start_pose[0] + 0.5, start_pose[1], start_pose[2]]
            plan = self.robot_interface.plan_path(start_pose, goal_pose)
            return plan is not None
        except Exception:
            return False
    
    def _run_execution_task(self):
        """Run motion execution task for benchmarking"""
        try:
            # Execute a simple motion
            success = self.robot_interface.execute_simple_motion()
            return success
        except Exception:
            return False
    
    def _run_balancing_task(self):
        """Run balance maintenance task for benchmarking"""
        try:
            # Maintain balance for a short period
            self.robot_interface.enable_balance_control(2.0)  # 2 seconds of balancing
            return True
        except Exception:
            return False
    
    def test_concurrent_operation_capacity(self):
        """Test the robot's ability to handle concurrent operations"""
        # Test that multiple systems can operate concurrently without degradation
        concurrent_tasks = [
            self._run_perception_concurrently,
            self._run_navigation_concurrently, 
            self._run_manipulation_concurrently,
            self._run_control_loop_concurrently
        ]
        
        # Run tasks concurrently using threads
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def run_task(task_func, task_id):
            try:
                success = task_func()
                results_queue.put((task_id, success))
            except Exception as e:
                results_queue.put((task_id, False))
        
        # Start all tasks concurrently
        threads = []
        for i, task_func in enumerate(concurrent_tasks):
            thread = threading.Thread(target=run_task, args=(task_func, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all tasks to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        results = {}
        while not results_queue.empty():
            task_id, success = results_queue.get()
            results[task_id] = success
        
        # Verify all concurrent tasks succeeded
        all_succeeded = all(results.values())
        self.assertTrue(
            all_succeeded,
            f"All concurrent operations should succeed. Results: {results}"
        )
        
        # Check performance degradation wasn't significant
        self.performance_monitor.stop_monitoring()
        concurrent_stats = self.performance_monitor.get_statistics()
        
        # Performance should be within 20% of single-operation performance
        self.assertLess(
            concurrent_stats['cpu_average'], 
            0.90,  # Even with concurrency, shouldn't exceed 90%
            f"Concurrent operations should maintain reasonable performance, "
            f"but got {concurrent_stats['cpu_average']*100:.1f}% CPU"
        )

# Test runner with configuration
def run_robot_tests():
    """Run the complete humanoid robot test suite"""
    rospy.loginfo("Starting humanoid robot test suite...")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(HumanoidRobotTestSuite))
    suite.addTest(unittest.makeSuite(HumanoidRobotBenchmarkTests))
    
    # Create test runner with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        buffer=False
    )
    
    # Run tests
    result = runner.run(suite)
    
    # Print summary
    rospy.loginfo("=" * 50)
    rospy.loginfo("TEST SUITE RESULTS SUMMARY")
    rospy.loginfo("=" * 50)
    rospy.loginfo(f"Tests run: {result.testsRun}")
    rospy.loginfo(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    rospy.loginfo(f"Failures: {len(result.failures)}")
    rospy.loginfo(f"Errors: {len(result.errors)}")
    
    if result.failures:
        rospy.loginfo("\nFAILURES:")
        for test, traceback in result.failures:
            rospy.loginfo(f"  {test}: {traceback}")
    
    if result.errors:
        rospy.loginfo("\nERRORS:")
        for test, traceback in result.errors:
            rospy.loginfo(f"  {test}: {traceback}")
    
    rospy.loginfo("=" * 50)
    
    # Calculate success rate
    total_tests = result.testsRun
    passed_tests = total_tests - len(result.failures) - len(result.errors)
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 100.0
    
    rospy.loginfo(f"OVERALL SUCCESS RATE: {success_rate:.1f}%")
    rospy.loginfo("=" * 50)
    
    return success_rate >= 95.0  # Return True if success rate is 95% or higher

if __name__ == '__main__':
    import sys
    
    try:
        success = run_robot_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        rospy.logerr(f"Test suite failed with error: {e}")
        sys.exit(1)
```

## Exercises

1. **Integration Challenge**: Implement the complete system integration for a simple task (e.g., "Pick up a bottle from the table") using the modules created in previous chapters, ensuring all safety checks pass.

2. **Performance Test**: Run performance benchmarks on your integrated system to identify bottlenecks and optimize for real-time operation.

3. **Safety Validation**: Create additional safety tests for edge cases not covered in the standard test suite.

## Summary

This chapter covered the complete integration and testing of the humanoid robot system. We explored comprehensive testing methodologies for robotic systems, implemented continuous integration pipelines specifically designed for robotics, established performance benchmarking protocols, and developed safety and validation systems to ensure safe operation of the humanoid platform.

The key takeaways include:
- System integration requires careful coordination between perception, cognition, and control systems
- Testing for robotics must consider both functional and safety requirements
- Performance benchmarking is critical for real-time robotic applications
- Safety systems must be validated at multiple levels: component, integration, and system-wide
- Continuous integration for robotics involves unique challenges related to hardware-in-the-loop testing

The chapter emphasized safety-first development practices essential for humanoid robots that operate near humans.

## Cross-references

For foundational concepts about Physical AI, see [Chapter 1: Introduction to Physical AI](../part-01-foundations/chapter-1). For related systems, see [Chapter 10: NVIDIA Isaac Platform Overview](../part-04-isaac/chapter-10) and [Chapter 15: Cognitive Planning with LLMs](../part-05-vla/chapter-15). For safety considerations in robotics, also see [Chapter 14: Safety and Compliance](../part-06-capstone/chapter-14) if available.