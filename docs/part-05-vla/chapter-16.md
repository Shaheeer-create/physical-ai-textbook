---
title: "Chapter 16: Computer Vision Integration"
description: "Implementing real-time object recognition and visual servoing for robotics"
sidebar_position: 3
---

# Chapter 16: Computer Vision Integration

## Learning Objectives

After completing this chapter, you should be able to:

- Implement real-time object recognition for robotics applications
- Design visual servoing systems for robot control
- Integrate hand-eye coordination for manipulation tasks
- Develop grasping and manipulation vision systems

## Introduction to Computer Vision in Robotics

### The Role of Vision in Robot Perception

Computer vision is fundamental to robotics, providing robots with the ability to perceive and understand their environment. Unlike traditional computer vision applications, robotics vision systems must:

1. **Operate in Real-Time**: Processing must be fast enough to support robot control
2. **Handle Dynamic Scenes**: Deal with robot movement, changing lighting, and dynamic objects
3. **Provide Spatial Information**: Estimate object positions and orientations for manipulation
4. **Integrate with Control Systems**: Feed vision data to robot controllers
5. **Handle Uncertainty**: Function reliably despite imperfect detection and segmentation

### Types of Vision in Robotics

1. **Scene Understanding**: Recognizing objects and structures in the environment
2. **Localization**: Determining the robot's position from visual cues
3. **Mapping**: Creating environmental representations from visual data
4. **Manipulation Guidance**: Providing precise positioning for manipulation
5. **Human Interaction**: Recognizing human gestures, faces, and intentions

## Real-time Object Recognition

### Deep Learning Approaches

Modern object recognition in robotics relies heavily on deep learning, particularly convolutional neural networks (CNNs). For real-time performance on robotic platforms, specialized architectures are used:

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np

class RealTimeObjectDetector(nn.Module):
    """Real-time object detection optimized for robotics platforms"""
    
    def __init__(self, num_classes=80, width_multiple=1.0, depth_multiple=1.0):
        super(RealTimeObjectDetector, self).__init__()
        
        # Backbones optimized for mobile/edge deployment
        self.backbone = MobileNetV3Backbone(width_multiple)
        
        # Detection heads for different scales
        self.heads = nn.ModuleList([
            DetectionHead(96, num_classes, width_multiple) if width_multiple >= 0.75 else nn.Identity(),  # Small objects
            DetectionHead(128, num_classes, width_multiple),  # Medium objects  
            DetectionHead(160, num_classes, width_multiple),  # Large objects
        ])
        
        # Feature pyramid network for multi-scale detection
        self.fpn = FeaturePyramidNetwork([96, 128, 160], 128)
        
        # Anchor boxes for different aspect ratios and scales
        self.anchor_generator = AnchorGenerator(
            sizes=[[32, 64, 128], [256, 512], [1024]],
            aspect_ratios=[[0.5, 1.0, 2.0]]
        )
        
        # Loss function for training
        self.criterion = YOLOLoss(num_classes)
    
    def forward(self, images, targets=None):
        """
        Forward pass for object detection
        
        Args:
            images: Batch of input images (B, C, H, W)
            targets: Optional ground truth for training (list of dicts)
        
        Returns:
            During training: loss dictionary
            During inference: list of detections per image
        """
        # Extract features from backbone
        features = self.backbone(images)  # List of feature maps at different scales
        
        # Apply FPN for multi-scale feature fusion
        fpn_features = self.fpn(features)
        
        # Apply detection heads
        predictions = []
        for i, head in enumerate(self.heads):
            if not isinstance(head, nn.Identity):  # Skip identity layers
                pred = head(fpn_features[i])
                predictions.append(pred)
        
        if self.training and targets is not None:
            # Calculate losses during training
            return self.criterion(predictions, targets)
        else:
            # Perform inference and return detections
            return self.post_process_predictions(predictions, images.shape[-2:])
    
    def post_process_predictions(self, predictions, image_shape):
        """Post-process predictions to get final detections"""
        all_detections = []
        
        for batch_idx in range(len(predictions[0])):
            batch_detections = []
            
            for scale_idx, scale_preds in enumerate(predictions):
                # Get predictions for this batch
                batch_scale_preds = scale_preds[batch_idx]
                
                # Decode anchor boxes and apply NMS
                detections = self.decode_predictions(batch_scale_preds, scale_idx, image_shape)
                batch_detections.extend(detections)
            
            # Apply non-maximum suppression across all scales
            final_detections = self.nms(batch_detections, iou_threshold=0.5)
            all_detections.append(final_detections)
        
        return all_detections
    
    def decode_predictions(self, predictions, scale_idx, image_shape):
        """Decode predictions to bounding boxes and confidence scores"""
        # Separate into boxes, objectness, and class scores
        boxes = predictions[..., :4]
        obj_scores = predictions[..., 4]
        cls_scores = predictions[..., 5:]
        
        # Apply sigmoid to get confidence scores
        obj_scores = torch.sigmoid(obj_scores)
        cls_scores = torch.sigmoid(cls_scores)
        
        # Convert box coordinates (usually in grid format) to image coordinates
        decoded_boxes = self.convert_to_image_coords(boxes, scale_idx, image_shape)
        
        # Get class predictions (top classes with sufficient confidence)
        cls_max_scores, cls_indices = torch.max(cls_scores, dim=-1)
        final_scores = obj_scores * cls_max_scores
        
        # Filter based on confidence threshold (0.3 for example)
        valid_mask = final_scores > 0.3
        
        detections = []
        for i in range(valid_mask.shape[0]):
            for j in range(valid_mask.shape[1]):
                if valid_mask[i, j]:  # High confidence detection
                    x1, y1, x2, y2 = decoded_boxes[i, j]
                    conf = final_scores[i, j]
                    cls_id = cls_indices[i, j].item()
                    
                    detection = {
                        'bbox': [x1.item(), y1.item(), x2.item(), y2.item()],
                        'confidence': conf.item(),
                        'class_id': cls_id,
                        'class_name': COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f'unknown_{cls_id}'
                    }
                    detections.append(detection)
        
        return detections
    
    def nms(self, detections, iou_threshold=0.5):
        """Non-Maximum Suppression to remove duplicate detections"""
        if len(detections) == 0:
            return []
        
        # Sort by confidence score
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            # Take the detection with highest confidence
            current = detections.pop(0)
            keep.append(current)
            
            # Remove detections that overlap too much with current
            detections = [det for det in detections 
                         if self.iou(current['bbox'], det['bbox']) < iou_threshold]
        
        return keep
    
    def iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area

class MobileNetV3Backbone(nn.Module):
    """Lightweight backbone optimized for real-time robotics"""
    
    def __init__(self, width_mult=1.0):
        super(MobileNetV3Backbone, self).__init__()
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, int(16 * width_mult), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(16 * width_mult))
        self.hs1 = HardSwish()
        
        # MobileNetV3 blocks (simplified)
        self.blocks = nn.ModuleList([
            InvertedResidual(16, 16, 3, 2, False, "RE", 1, width_mult),  # 1/4 resolution
            InvertedResidual(16, 24, 3, 2, False, "RE", 1, width_mult),  # 1/8 resolution
            InvertedResidual(24, 24, 3, 1, False, "RE", 1, width_mult),  # 1/8 resolution
            InvertedResidual(24, 40, 5, 2, True, "RE", 2, width_mult),   # 1/16 resolution
            InvertedResidual(40, 40, 5, 1, True, "RE", 2, width_mult),   # 1/16 resolution
            InvertedResidual(40, 40, 5, 1, True, "RE", 2, width_mult),   # 1/16 resolution
            InvertedResidual(40, 80, 3, 2, False, "HS", 1, width_mult),  # 1/32 resolution
            InvertedResidual(80, 80, 3, 1, False, "HS", 1, width_mult),  # 1/32 resolution
            InvertedResidual(80, 80, 3, 1, False, "HS", 1, width_mult),  # 1/32 resolution
            InvertedResidual(80, 112, 3, 1, True, "HS", 1, width_mult),  # 1/32 resolution
            InvertedResidual(112, 112, 3, 1, True, "HS", 1, width_mult), # 1/32 resolution
            InvertedResidual(112, 160, 5, 2, True, "HS", 2, width_mult), # 1/64 resolution
            InvertedResidual(160, 160, 5, 1, True, "HS", 2, width_mult), # 1/64 resolution
        ])
        
        self.out_channels = [24, 40, 80]  # Channels at different resolutions for FPN
    
    def forward(self, x):
        features = []
        
        x = self.hs1(self.bn1(self.conv1(x)))
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            # Extract features at different resolutions for detection heads
            if i in [2, 4, 7, 12]:  # indices where we want feature maps
                features.append(x)
        
        return features

class DetectionHead(nn.Module):
    """Detection head for a specific scale"""
    
    def __init__(self, in_channels, num_classes, width_mult=1.0):
        super(DetectionHead, self).__init__()
        
        # Prediction layers
        self.conv1 = nn.Conv2d(in_channels, int(256 * width_mult), 3, padding=1)
        self.bn1 = nn.BatchNorm2d(int(256 * width_mult))
        self.act = nn.LeakyReLU(0.1)
        
        # Number of anchors at each location (usually 3: different aspect ratios)
        self.num_anchors = 3
        self.num_classes = num_classes
        
        # Final prediction head
        # For each anchor: [tx, ty, tw, th, objectness, class1, class2, ...]
        self.pred = nn.Conv2d(int(256 * width_mult), 
                             self.num_anchors * (4 + 1 + self.num_classes), 
                             1)
    
    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pred(x)
        
        # Reshape to [batch, num_anchors, height, width, 4 + 1 + num_classes]
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, self.num_anchors, 4 + 1 + self.num_classes, height, width)
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, num_anchors, 5 + num_classes]
        
        return x

class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature integration"""
    
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(channels, out_channels, 1) for channels in in_channels_list
        ])
        
        # Top-down pathway
        self.top_down_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])
    
    def forward(self, features):
        """Features should be in increasing spatial resolution order (smallest last)"""
        results = []
        
        # Start with highest resolution
        prev_feature = self.lateral_convs[-1](features[-1])
        results.insert(0, self.top_down_convs[-1](prev_feature))
        
        # Process in reverse order (top-down)
        for i in range(len(features) - 2, -1, -1):
            lateral_feature = self.lateral_convs[i](features[i])
            
            # Upsample previous feature map
            prev_feature = F.interpolate(prev_feature, size=lateral_feature.shape[-2:], 
                                         mode='nearest')
            
            # Add lateral feature
            prev_feature = lateral_feature + prev_feature
            
            # Apply top-down conv
            results.insert(0, self.top_down_convs[i](prev_feature))
        
        return results

# HardSwish activation function (from MobileNetV3)
class HardSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3) / 6

# Inverted residual block (from MobileNetV3)
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel, stride, use_se, activation, n_div=4, width_mult=1.0):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        
        hidden_dim = int(inp * 4 * width_mult)
        self.identity = stride == 1 and inp == oup
        self.use_res_connect = stride == 1 and inp == oup
        
        layers = []
        if stride == 1 and inp == oup:
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(HardSwish() if activation == "HS" else nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(oup))
        else:
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(HardSwish() if activation == "HS" else nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel, stride, kernel//2, 
                                   groups=hidden_dim, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(HardSwish() if activation == "HS" else nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(oup))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# Anchor generator for object detection
class AnchorGenerator(nn.Module):
    def __init__(self, sizes, aspect_ratios):
        super(AnchorGenerator, self).__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        
    def forward(self, image_shape, feature_map_shapes):
        """Generate anchors for different feature map scales"""
        anchors = []
        for feat_h, feat_w in feature_map_shapes:
            # Generate anchors for this feature map size
            grid_anchors = self._create_grid_anchors(feat_h, feat_w, image_shape)
            anchors.append(grid_anchors)
        
        return anchors
    
    def _create_grid_anchors(self, feat_h, feat_w, image_shape):
        """Create anchor boxes on a grid"""
        # This is a simplified version - in practice, this would be more complex
        return torch.zeros(feat_h, feat_w, len(self.sizes) * len(self.aspect_ratios), 4)
```

### TensorRT Optimization for Robotics

For deployment on NVIDIA robotics platforms:

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TensorRTOptimizer:
    """Optimize neural networks for real-time execution on robotics platforms"""
    
    def __init__(self, model_path, engine_path=None):
        self.model_path = model_path
        self.engine_path = engine_path or model_path.replace('.pt', '.engine')
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None
    
    def build_engine(self, input_shape, precision='fp16'):
        """
        Build TensorRT engine from PyTorch model
        Args:
            input_shape: Shape of input tensor (e.g., [1, 3, 416, 416])
            precision: 'fp16', 'fp32', or 'int8'
        """
        # Create builder and network
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        
        # Set precision and optimization flags
        if precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        
        # Set memory limit (in MB)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        # Parse the ONNX model
        parser = trt.OnnxParser(network, self.logger)
        
        # Load the ONNX model
        with open(self.model_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Set optimization profile
        profile = builder.create_optimization_profile()
        
        # Set input and output dimensions
        input_name = network.get_input(0).name
        min_shape = [1] + [input_shape[i] for i in range(1, len(input_shape))]
        opt_shape = input_shape  # Optimal batch size
        max_shape = [4] + [input_shape[i] for i in range(1, len(input_shape))]  # Max batch size
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        # Build engine
        self.engine = builder.build_serialized_network(network, config)
        
        if self.engine is None:
            print("Failed to build engine")
            return None
        
        # Save engine to file
        with open(self.engine_path, 'wb') as f:
            f.write(self.engine)
        
        print(f"TensorRT engine saved to {self.engine_path}")
        
        # Create execution context
        self.context = trt.Runtime(self.logger).deserialize_cuda_engine(self.engine)
        
        # Allocate buffers
        self._allocate_buffers()
        
        return self.engine
    
    def _allocate_buffers(self):
        """Allocate CUDA buffers for input/output"""
        # Create CUDA stream
        self.stream = cuda.Stream()
        
        # Get bindings info
        for binding in self.context:
            size = trt.volume(self.context.get_binding_shape(binding))
            dtype = trt.nptype(self.context.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.context.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def load_engine(self):
        """Load existing TensorRT engine from file"""
        with open(self.engine_path, 'rb') as f:
            self.engine = f.read()
        
        self.context = trt.Runtime(self.logger).deserialize_cuda_engine(self.engine)
        
        # Allocate buffers
        self._allocate_buffers()
    
    def infer(self, input_data):
        """
        Run inference on input data
        Args:
            input_data: NumPy array of input data
        Returns:
            Output data as NumPy array
        """
        # Copy input data to host buffer
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # Transfer input data to device
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        
        # Synchronize
        self.stream.synchronize()
        
        # Reshape output
        output = self.outputs[0]['host'].reshape(self.context.get_binding_shape(1))
        
        return output

class RealTimeVisionPipeline:
    """Real-time computer vision pipeline for robotics applications"""
    
    def __init__(self, model_path, camera_topic='/camera/rgb/image_raw'):
        self.model_path = model_path
        self.camera_topic = camera_topic
        
        # Initialize TensorRT optimizer
        self.trt_optimizer = TensorRTOptimizer(model_path)
        
        # Load optimized model
        if not self.load_optimized_model():
            print("Building TensorRT engine...")
            self.build_optimized_model()
        
        # Initialize camera interface
        self.rgb_sub = rospy.Subscriber(camera_topic, Image, self.image_callback)
        self.bridge = CvBridge()
        
        # Detection results publisher
        self.detection_pub = rospy.Publisher('/vision/detections', DetectionArray, queue_size=10)
        
        # Performance monitoring
        self.fps_counter = FPSCounter(window_size=10)
        self.last_inference_time = 0
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.image_queue = []  # For handling multiple images in flight
        
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_sub = rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, self.camera_info_callback)
        
        rospy.loginfo("Real-time vision pipeline initialized")
    
    def load_optimized_model(self):
        """Load pre-built TensorRT engine if available"""
        try:
            self.trt_optimizer.load_engine()
            rospy.loginfo("Loaded TensorRT engine")
            return True
        except FileNotFoundError:
            rospy.logwarn(f"TensorRT engine not found at {self.trt_optimizer.engine_path}")
            return False
    
    def build_optimized_model(self):
        """Build TensorRT engine from PyTorch model"""
        # Set a reasonable input shape for object detection
        # This would typically be the model's expected input size
        input_shape = [1, 3, 416, 416]  # Example: YOLOv4-tiny input size
        self.trt_optimizer.build_engine(input_shape, precision='fp16')
    
    def camera_info_callback(self, msg):
        """Receive camera calibration parameters"""
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.dist_coeffs = np.array(msg.D)
    
    def image_callback(self, msg):
        """Process incoming image"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Add to processing queue
            self.image_queue.append({
                'image': cv_image,
                'header': msg.header,
                'timestamp': time.time()
            })
            
            # Process if queue is ready
            if len(self.image_queue) >= 1:  # Process every image
                self.process_images()
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def process_images(self):
        """Process images in the queue"""
        if not self.image_queue:
            return
        
        # Get the most recent image (skip older ones if too many queued)
        if len(self.image_queue) > 2:
            # Keep only the most recent image if too far behind
            self.image_queue = [self.image_queue[-1]]
        
        for item in self.image_queue:
            self.process_single_image(item['image'], item['header'])
        
        self.image_queue.clear()
    
    def process_single_image(self, image, header):
        """Process a single image for real-time object detection"""
        start_time = time.time()
        
        # Preprocess image for the model
        input_tensor = self.preprocess_image(image)
        
        # Run inference with TensorRT
        try:
            output = self.trt_optimizer.infer(input_tensor)
            
            # Post-process detection results
            detections = self.post_process_detections(output, image.shape[:2], header)
            
            # Calculate 3D positions if camera calibration available
            if self.camera_matrix is not None:
                detections = self.calculate_3d_positions(detections, image)
            
            # Publish results
            self.publish_detections(detections, header)
            
            self.last_inference_time = time.time() - start_time
            self.fps_counter.update()
            
            rospy.loginfo_throttle(
                1.0,  # Log once per second
                f"Processed image at {(1.0/self.last_inference_time):.2f} FPS, "
                f"inference time: {self.last_inference_time*1000:.1f} ms"
            )
            
        except Exception as e:
            rospy.logerr(f"Error during inference: {e}")
    
    def preprocess_image(self, image):
        """Preprocess image for network input"""
        # Resize image to model input size (e.g., 416x416 for YOLO)
        input_h, input_w = 416, 416  # Model input dimensions
        
        # Preserve aspect ratio with padding
        h, w = image.shape[:2]
        scale = min(input_w / w, input_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create canvas and center image
        canvas = np.full((input_h, input_w, 3), 128, dtype=np.uint8)  # Gray canvas
        start_x = (input_w - new_w) // 2
        start_y = (input_h - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        # Convert BGR to RGB and normalize to [0, 1]
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        canvas = canvas.astype(np.float32) / 255.0
        
        # Transpose to CHW format (channels first)
        canvas = canvas.transpose(2, 0, 1)
        
        # Add batch dimension
        batch_canvas = np.expand_dims(canvas, axis=0)
        
        return batch_canvas
    
    def post_process_detections(self, output, img_shape, header):
        """Post-process neural network output into detection format"""
        # This is a simplified example - in practice, this would depend 
        # on the specific network architecture (YOLO, SSD, etc.)
        
        height, width = img_shape
        detections = []
        
        # Example for YOLO-style output (this is pseudocode)
        # In practice, you'd decode anchors, apply sigmoid, etc.
        for detection in output:
            x, y, w, h, conf = detection[:5]  # Bounding box and confidence
            
            # Convert from model coordinates to image coordinates
            x *= width
            y *= height
            w *= width
            h *= height
            
            # Convert to corner coordinates
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
            
            # Apply confidence threshold
            if conf > self.confidence_threshold:
                # TODO: Add class name using argmax of class scores
                detection_dict = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class_id': 0,  # TODO: Get from output
                    'class_name': 'unknown',  # TODO: Get from output
                    'header': header
                }
                
                # Add 3D position if available (will be computed later)
                detection_dict['position_3d'] = None
                
                detections.append(detection_dict)
        
        # Apply Non-Maximum Suppression
        detections = self.apply_nms(detections, self.nms_threshold)
        
        return detections
    
    def calculate_3d_positions(self, detections, image):
        """Calculate 3D world positions from 2D detections using camera calibration"""
        if self.camera_matrix is None:
            rospy.logwarn("Camera calibration not available, skipping 3D position calculation")
            return detections
        
        for detection in detections:
            bbox = detection['bbox']
            x, y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2  # Center point
            
            # Undistort point
            point_2d = np.array([[x, y]], dtype=np.float32)
            undistorted = cv2.undistortPoints(
                point_2d,
                self.camera_matrix,
                self.dist_coeffs,
                P=self.camera_matrix
            )[0][0]
            
            # Convert to 3D ray from camera center
            # In practice, you'd use depth from stereo or depth camera
            # Here, we'll just store the 2D coordinates
            detection['position_3d'] = {
                'x': float(undistorted[0]),
                'y': float(undistorted[1]),
                'z': 1.0  # Placeholder depth
            }
        
        return detections
    
    def apply_nms(self, detections, threshold):
        """Apply Non-Maximum Suppression to eliminate duplicate detections"""
        if len(detections) == 0:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            # Take highest confidence detection
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            detections = [det for det in detections 
                         if self.calculate_iou(current['bbox'], det['bbox']) < threshold]
        
        return keep
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def publish_detections(self, detections, header):
        """Publish detection results to ROS topic"""
        detection_array = DetectionArray()
        detection_array.header = header
        
        for det in detections:
            detection_msg = Detection2D()
            detection_msg.header = det['header']
            
            # Bounding box
            bbox = det['bbox']
            detection_msg.bbox.center.x = (bbox[0] + bbox[2]) / 2.0
            detection_msg.bbox.center.y = (bbox[1] + bbox[3]) / 2.0
            detection_msg.bbox.size_x = bbox[2] - bbox[0]
            detection_msg.bbox.size_y = bbox[3] - bbox[1]
            
            # Confidence
            detection_msg.results.append(
                ObjectHypothesisWithPose(
                    hypothesis=ObjectHypothesis(
                        id=det['class_id'],
                        score=det['confidence']
                    )
                )
            )
            
            # Add to array
            detection_array.detections.append(detection_msg)
        
        self.detection_pub.publish(detection_array)

class FPSCounter:
    """Simple FPS counter for performance monitoring"""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.times = []
        self.last_time = time.time()
    
    def update(self):
        """Update counter with current frame"""
        current_time = time.time()
        self.times.append(current_time - self.last_time)
        self.last_time = current_time
        
        # Keep only last N times
        if len(self.times) > self.window_size:
            self.times.pop(0)
    
    def get_fps(self):
        """Get current FPS"""
        if not self.times:
            return 0.0
        
        avg_time = sum(self.times) / len(self.times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
```

## Visual Servoing Systems

### Position-Based vs. Image-Based Visual Servoing

```python
import geometry_msgs.msg
import tf2_ros
from scipy.spatial.transform import Rotation as R

class VisualServoController:
    """Controller for visual servoing - controlling robot motion based on visual feedback"""
    
    def __init__(self, servo_type='position_based'):
        """
        Args:
            servo_type: 'position_based' or 'image_based'
        """
        self.servo_type = servo_type
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # PID controllers for visual servoing
        self.pid_x = PIDController(kp=1.0, ki=0.01, kd=0.1)
        self.pid_y = PIDController(kp=1.0, ki=0.01, kd=0.1)
        self.pid_z = PIDController(kp=1.0, ki=0.01, kd=0.1)
        self.pid_yaw = PIDController(kp=0.5, ki=0.005, kd=0.05)
        
        # Goal state
        self.goal_3d_pos = None
        self.goal_image_pos = None
        self.goal_rotation = None
        
        # Current state
        self.current_3d_pos = None
        self.current_image_pos = None
        self.current_rotation = None
        
        # Robot state publisher
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Camera parameters
        self.fx = 525.0  # Focal length x
        self.fy = 525.0  # Focal length y
        self.cx = 319.5  # Principal point x
        self.cy = 239.5  # Principal point y
        
        rospy.loginfo(f"Visual servo controller initialized: {servo_type}")
    
    def set_goal_3d(self, x, y, z, rotation=None):
        """Set 3D position goal for position-based servoing"""
        self.goal_3d_pos = np.array([x, y, z])
        self.goal_rotation = rotation or [0, 0, 0, 1]  # Default: no rotation (quaternion)
    
    def set_goal_image(self, x, y):
        """Set 2D image coordinates goal for image-based servoing"""
        self.goal_image_pos = np.array([x, y])
    
    def update_visual_feedback(self, detection_3d=None, detection_2d=None):
        """Update controller with latest visual feedback"""
        if self.servo_type == 'position_based':
            if detection_3d is not None:
                self.current_3d_pos = np.array([detection_3d.x, detection_3d.y, detection_3d.z])
                
                # If rotation is available
                if hasattr(detection_3d, 'rotation'):
                    self.current_rotation = detection_3d.rotation
        
        elif self.servo_type == 'image_based':
            if detection_2d is not None:
                self.current_image_pos = np.array([detection_2d.x, detection_2d.y])
    
    def compute_control_command(self):
        """Compute control command based on visual error"""
        if self.servo_type == 'position_based':
            return self._compute_position_based_control()
        elif self.servo_type == 'image_based':
            return self._compute_image_based_control()
        else:
            raise ValueError(f"Unknown servo type: {self.servo_type}")
    
    def _compute_position_based_control(self):
        """Compute control command for position-based visual servoing"""
        if self.goal_3d_pos is None or self.current_3d_pos is None:
            return Twist()  # No goal or current position available
        
        # Calculate position error
        pos_error = self.goal_3d_pos - self.current_3d_pos
        
        # Calculate rotational error (simplified)
        rot_error = np.array([0, 0, 0])  # To be computed based on quaternions
        if self.goal_rotation and self.current_rotation:
            goal_rot = R.from_quat(self.goal_rotation)
            curr_rot = R.from_quat(self.current_rotation)
            rel_rot = goal_rot.inv() * curr_rot
            rot_error = rel_rot.as_rotvec()  # Convert to angle-axis representation
        
        # Apply PID control to each axis
        cmd = Twist()
        cmd.linear.x = self.pid_x.update(pos_error[0])
        cmd.linear.y = self.pid_y.update(pos_error[1])
        cmd.linear.z = self.pid_z.update(pos_error[2])
        
        cmd.angular.z = self.pid_yaw.update(rot_error[2])  # Simplified: only yaw control
        
        return cmd
    
    def _compute_image_based_control(self):
        """Compute control command for image-based visual servoing"""
        if self.goal_image_pos is None or self.current_image_pos is None:
            return Twist()  # No goal or current position available
        
        # Calculate pixel error
        pixel_error = self.goal_image_pos - self.current_image_pos
        
        # Convert pixel errors to camera-frame errors using pinhole camera model
        # dx = (pixel_x - cx) * depth / fx
        # dy = (pixel_y - cy) * depth / fy
        # We need an estimate of depth to the target
        
        # For now, assume constant depth (or use depth from detection)
        assumed_depth = 1.0  # meters
        
        # Convert pixel errors to camera frame displacements
        cam_dx = (pixel_error[0]) * assumed_depth / self.fx
        cam_dy = (pixel_error[1]) * assumed_depth / self.fy
        
        # Calculate camera-frame velocity
        cam_vel_x = self.pid_x.update(cam_dx)
        cam_vel_y = self.pid_y.update(cam_dy)
        
        # Convert from camera frame to world/base frame (assuming level camera)
        current_yaw = self._get_current_robot_yaw()
        cmd = Twist()
        cmd.linear.x = cam_vel_x * np.cos(current_yaw) - cam_vel_y * np.sin(current_yaw)
        cmd.linear.y = cam_vel_x * np.sin(current_yaw) + cam_vel_y * np.cos(current_yaw)
        
        # Add angular component to correct for lateral error
        cmd.angular.z = self.pid_yaw.update(-pixel_error[0] / 100.0)  # Normalize pixel error
        
        return cmd
    
    def _get_current_robot_yaw(self):
        """Get current robot yaw angle from TF"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rospy.Time(0), rospy.Duration(1.0)
            )
            quat = transform.transform.rotation
            euler = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_euler('xyz')
            return euler[2]  # Yaw angle
        except:
            # If TF not available, return 0
            return 0.0

class PIDController:
    """Simple PID controller for visual servoing"""
    
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, integral_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
    
    def update(self, error):
        """Update controller with new error and return control output"""
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            return 0.0
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative
        
        # Store values for next iteration
        self.prev_error = error
        self.last_time = current_time
        
        # Calculate control output
        output = p_term + i_term + d_term
        
        # Limit output range
        output = np.clip(output, -5.0, 5.0)
        
        return output

# Example usage of visual servoing
class ObjectTrackingServo:
    """Example of using visual servoing to track an object"""
    
    def __init__(self):
        self.visual_servo = VisualServoController(servo_type='image_based')
        self.detection_sub = rospy.Subscriber('/vision/detections', DetectionArray, self.detection_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Track the first detected object
        self.tracked_object = None
        self.tracking_enabled = False
        
        # Timer for control loop
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        rospy.loginfo("Object tracking servo initialized")
    
    def detection_callback(self, msg):
        """Process detection messages"""
        if not msg.detections:
            self.tracked_object = None
            self.tracking_enabled = False
            return
        
        # For simplicity, track the first detection
        # In practice, you'd have logic to select which object to track
        detection = msg.detections[0]
        
        # Update position in image
        image_pos = np.array([
            detection.bbox.center.x, 
            detection.bbox.center.y
        ])
        
        # Set goal to center of image (desired position)
        goal_pos = np.array([320, 240])  # Assuming 640x480 image
        
        if not self.tracking_enabled:
            self.visual_servo.set_goal_image(goal_pos[0], goal_pos[1])
            self.tracking_enabled = True
        
        # Update visual feedback
        self.visual_servo.update_visual_feedback(detection_2d=image_pos)
    
    def control_loop(self, event):
        """Control loop that runs periodically"""
        if not self.tracking_enabled:
            # Publish zero velocity when not tracking
            cmd = Twist()
            self.cmd_pub.publish(cmd)
            return
        
        # Compute control command
        cmd = self.visual_servo.compute_control_command()
        
        # Publish command
        self.cmd_pub.publish(cmd)

if __name__ == "__main__":
    rospy.init_node('object_tracking_servo')
    tracker = ObjectTrackingServo()
    rospy.spin()
```

## Hand-Eye Coordination

### Eye-in-Hand vs. Eye-to-Hand Configurations

```python
class HandEyeCalibration:
    """Handle hand-eye calibration for coordinated robot-vision systems"""
    
    def __init__(self, calibration_method='ax=xb'):
        """
        Args:
            calibration_method: 'ax=xb' (eye-in-hand) or 'ax=xb_fixed' (eye-to-hand)
        """
        self.calibration_method = calibration_method
        self.extrinsics = None  # Transformation from camera to end-effector (or world)
        self.intrinsics = None  # Camera intrinsic parameters
        
        # Calibration data storage
        self.calibration_poses = []  # Robot poses during calibration
        self.calibration_points = []  # Image points during calibration
        
        # Chessboard pattern for calibration
        self.pattern_size = (9, 6)  # 9x6 internal corners
        self.square_size = 0.025  # 2.5cm squares
    
    def detect_calibration_pattern(self, image):
        """Detect chessboard pattern in image for calibration"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(
            gray, 
            self.pattern_size, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # Improve corner accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(
                gray, 
                corners, 
                (11, 11), 
                (-1, -1), 
                criteria
            )
        
        return ret, corners
    
    def add_calibration_sample(self, robot_pose, image):
        """Add a sample to the calibration dataset"""
        # Get robot pose transformation
        # This would convert from robot pose representation to homogeneous transformation
        T_robot = self._pose_to_transform(robot_pose)
        
        # Detect calibration pattern
        success, image_points = self.detect_calibration_pattern(image)
        
        if success:
            # Generate corresponding object points (chessboard corners in 3D)
            object_points = self._generate_object_points()
            
            self.calibration_poses.append(T_robot)
            self.calibration_points.append({
                'image_points': image_points.reshape(-1, 2),
                'object_points': object_points,
                'robot_pose': robot_pose
            })
            
            rospy.loginfo(f"Added calibration sample #{len(self.calibration_poses)}")
            return True
        else:
            rospy.logwarn("Failed to detect calibration pattern in image")
            return False
    
    def _generate_object_points(self):
        """Generate chessboard object points"""
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def _pose_to_transform(self, pose):
        """Convert ROS pose to 4x4 homogeneous transformation matrix"""
        # Convert position
        t = np.array([pose.position.x, pose.position.y, pose.position.z])
        
        # Convert orientation (quaternion to rotation matrix)
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        rotation = R.from_quat(quat).as_matrix()
        
        # Create homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = t
        
        return T
    
    def calibrate_hand_eye(self):
        """Perform hand-eye calibration"""
        if len(self.calibration_poses) < 6:
            rospy.logerr("Need at least 6 calibration samples for hand-eye calibration")
            return False
        
        try:
            if self.calibration_method == 'ax=xb':
                # Eye-in-hand calibration
                T_calib = self._calibrate_eye_in_hand()
            elif self.calibration_method == 'ax=xb_fixed':
                # Eye-to-hand calibration  
                T_calib = self._calibrate_eye_to_hand()
            else:
                raise ValueError(f"Unknown calibration method: {self.calibration_method}")
            
            self.extrinsics = T_calib
            rospy.loginfo("Hand-eye calibration completed successfully")
            rospy.loginfo(f"Extrinsic transformation:\n{T_calib}")
            return True
            
        except Exception as e:
            rospy.logerr(f"Hand-eye calibration failed: {e}")
            return False
    
    def _calibrate_eye_in_hand(self):
        """Calibrate eye-in-hand system (camera mounted on robot end-effector)"""
        # The hand-eye calibration problem: AX = XB
        # A represents robot motion, X is unknown camera-to-end-effector transform, 
        # B represents observed motion in camera frame
        
        # Convert poses to relative transformations
        A_list = []  # Motion of robot (end-effector)
        B_list = []  # Observed motion in camera frame (from checkerboard detection)
        
        for i in range(1, len(self.calibration_poses)):
            # Robot motion (from pose i-1 to pose i)
            T_prev = self.calibration_poses[i-1]
            T_curr = self.calibration_poses[i]
            A = np.linalg.inv(T_prev) @ T_curr  # Motion of robot: T_prev_world_to_curr_world
            A_list.append(A)
            
            # Camera motion (from image i-1 to image i) - calculate from point correspondences
            prev_points = self.calibration_points[i-1]['image_points']
            curr_points = self.calibration_points[i]['image_points']
            
            # Estimate camera motion using Essential matrix or Homography
            # This is a simplified version - in practice, you'd use proper methods
            B = self._estimate_camera_motion(
                prev_points, 
                curr_points, 
                self.intrinsics
            )
            B_list.append(B)
        
        # Solve AX = XB using Tsai's method or other robust algorithms
        # For simplicity, using OpenCV's implementation
        success, R_cam, t_cam = cv2.calibrateHandEye(
            A_list, B_list,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        
        if success:
            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = R_cam
            T[:3, 3] = t_cam.flatten()
            return T
        else:
            raise RuntimeError("Failed to solve hand-eye calibration")
    
    def _calibrate_eye_to_hand(self):
        """Calibrate eye-to-hand system (camera fixed in world, observing robot)"""
        # For eye-to-hand calibration, we have different mathematical formulation
        # Camera is fixed in world frame, robot moves relative to camera
        
        # This would involve different mathematical approach
        # Simplified: Average of all camera-to-object transformations corrected by robot poses
        pass  # Implementation depends on specific eye-to-hand approach
    
    def _estimate_camera_motion(self, prev_points, curr_points, K):
        """Estimate camera motion from corresponding points"""
        # Use Essential matrix to estimate rotation and translation
        E, mask = cv2.findEssentialMat(
            prev_points, curr_points, 
            K, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=1.0
        )
        
        if E is not None:
            # Recover rotation and translation
            _, R, t, _ = cv2.recoverPose(E, prev_points, curr_points, K)
            
            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()
            
            return T
        else:
            # If essential matrix fails, return identity (no motion)
            return np.eye(4)

class ManipulationVisionController:
    """Controller for vision-guided manipulation tasks"""
    
    def __init__(self):
        # Initialize hand-eye calibration
        self.hand_eye_calib = HandEyeCalibration(calibration_method='ax=xb')
        
        # Initialize visual servoing for object approach
        self.approach_servo = VisualServoController(servo_type='image_based')
        
        # Initialize grasping vision system
        self.grasp_planner = GraspPoseEstimator()
        
        # Robot interfaces
        self.joint_pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=10)
        self.gripper_pub = rospy.Publisher('/gripper_controller/command', GripperCommand, queue_size=10)
        
        # Vision interfaces
        self.detection_sub = rospy.Subscriber('/vision/detections', DetectionArray, self.detection_callback)
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        
        # Object pose estimation
        self.object_poses = {}  # Track detected objects by ID
        
        # Manipulation states
        self.manipulation_state = 'idle'  # idle, tracking, approaching, grasping, retracting
        self.currentTarget = None
        
        rospy.loginfo("Manipulation vision controller initialized")
    
    def detection_callback(self, msg):
        """Process object detections for manipulation"""
        for detection in msg.detections:
            # Get 3D position from depth if available
            if hasattr(detection, 'position_3d') and detection.position_3d:
                # Store object pose in world frame
                obj_pose = self._convert_2d_to_3d_pose(
                    detection.bbox.center,
                    detection.position_3d,
                    msg.header  # Frame information
                )
                
                self.object_poses[detection.results[0].hypothesis.id] = obj_pose
    
    def _convert_2d_to_3d_pose(self, center_2d, depth_3d, header):
        """Convert 2D image center with depth to 3D pose in robot frame"""
        # This would use camera calibration and hand-eye transform
        # Convert from image coordinates to camera frame
        camera_pose = self._image_to_camera_frame(center_2d, depth_3d)
        
        # Transform from camera frame to robot base frame using hand-eye calibration
        if self.hand_eye_calib.extrinsics is not None:
            T_robot_camera = self.hand_eye_calib.extrinsics
            T_camera_object = self._pose_to_transform(camera_pose)
            T_robot_object = T_robot_camera @ T_camera_object
            return self._transform_to_pose_msg(T_robot_object)
        else:
            # If no calibration, return camera frame pose
            return camera_pose
    
    def _image_to_camera_frame(self, center_2d, depth_3d):
        """Convert image coordinates + depth to 3D coordinates in camera frame"""
        # Use camera intrinsic parameters
        # [x, y, z]_camera = [fx, 0, cx; 0, fy, cy; 0, 0, 1]^-1 * [u, v, 1]^T * depth
        u, v = center_2d.x, center_2d.y
        z = depth_3d.z  # Use depth value
        
        x = (u - self.hand_eye_calib.intrinsics[0, 2]) * z / self.hand_eye_calib.intrinsics[0, 0]
        y = (v - self.hand_eye_calib.intrinsics[1, 2]) * z / self.hand_eye_calib.intrinsics[1, 1]
        
        return [x, y, z]
    
    def execute_pick_and_place(self, target_object_id, place_location):
        """Execute pick and place task using vision guidance"""
        if target_object_id not in self.object_poses:
            rospy.logerr(f"Target object {target_object_id} not visible")
            return False
        
        target_pose = self.object_poses[target_object_id]
        
        # State machine for pick and place
        if self.manipulation_state == 'idle':
            # Approach object using visual servoing
            self._approach_object(target_pose)
            self.manipulation_state = 'approaching'
        
        elif self.manipulation_state == 'approaching':
            # Check if approach is complete
            if self._is_approach_complete():
                # Plan grasp using vision
                grasp_pose = self.grasp_planner.plan_grasp(target_pose, target_object_id)
                self._execute_grasp(grasp_pose)
                self.manipulation_state = 'grasping'
        
        elif self.manipulation_state == 'grasping':
            # Check if grasp is complete
            if self._is_grasp_complete():
                # Move to place location
                self._move_to_place_location(place_location)
                self.manipulation_state = 'retracting'
        
        elif self.manipulation_state == 'retracting':
            # Release object at place location
            self._release_object()
            self.manipulation_state = 'idle'
            return True  # Task complete
    
    def _approach_object(self, object_pose):
        """Approach object using visual servoing before grasp"""
        # Move robot end-effector to position above object
        approach_pose = copy.deepcopy(object_pose)
        approach_pose.position.z += 0.1  # 10 cm above object
        
        # Use visual servoing to move toward this pose
        # This would involve sending joint commands or Cartesian commands
        self._move_to_pose(approach_pose)
    
    def _execute_grasp(self, grasp_pose):
        """Execute grasp maneuver with vision feedback"""
        # Move to pre-grasp position (slightly above grasp point)
        pre_grasp_pose = copy.deepcopy(grasp_pose)
        pre_grasp_pose.position.z += 0.02  # 2 cm above grasp point
        
        self._move_to_pose(pre_grasp_pose)
        
        # Execute grasp motion (downward)
        self._execute_grasp_motion(grasp_pose)
        
        # Close gripper
        self._close_gripper()
    
    def _move_to_pose(self, pose):
        """Move robot to specified pose"""
        # This would interface with robot's motion planning system
        # For now, simplified example
        rospy.loginfo(f"Moving to pose: {pose}")
        # Implementation would send trajectory to robot controller
    
    def _execute_grasp_motion(self, grasp_pose):
        """Execute precise grasp motion"""
        # Move downward slowly to grasp location
        # This might involve force control or vision feedback
        rospy.loginfo(f"Executing grasp motion to: {grasp_pose}")
    
    def _close_gripper(self):
        """Close robot gripper"""
        gripper_cmd = GripperCommand()
        gripper_cmd.position = 0.0  # Fully closed
        gripper_cmd.max_effort = 50.0  # Moderate effort
        self.gripper_pub.publish(gripper_cmd)
    
    def _move_to_place_location(self, place_pose):
        """Move object to place location"""
        # Move to place location with object
        self._move_to_pose(place_pose)
    
    def _release_object(self):
        """Release object at destination"""
        gripper_cmd = GripperCommand()
        gripper_cmd.position = 1.0  # Fully open
        gripper_cmd.max_effort = 10.0  # Low effort to avoid throwing
        self.gripper_pub.publish(gripper_cmd)

class GraspPoseEstimator:
    """Estimate optimal grasp poses using vision"""
    
    def __init__(self):
        # Use deep learning-based grasp detection
        # This would typically be a model trained on grasp data
        self.grasp_model = self._load_grasp_model()
        
        # Parameters for grasp planning
        self.min_grasp_quality = 0.7
        self.max_grasp_attempts = 5
        self.approach_distance = 0.05  # 5cm approach distance
    
    def plan_grasp(self, object_pose, object_class_id):
        """Plan optimal grasp pose for the given object"""
        # In a real implementation, this would use the grasp model
        # For this example, return a simple grasp pose
        
        # Generate potential grasp points around the object
        potential_grasps = self._generate_potential_grasps(object_pose, object_class_id)
        
        # Score each potential grasp
        best_grasp = None
        best_score = 0.0
        
        for grasp in potential_grasps:
            score = self._evaluate_grasp_quality(grasp, object_pose)
            if score > best_score and score > self.min_grasp_quality:
                best_score = score
                best_grasp = grasp
        
        if best_grasp is not None:
            rospy.loginfo(f"Selected grasp with quality: {best_score:.3f}")
            return best_grasp
        else:
            rospy.logwarn("No suitable grasp found")
            return None
    
    def _generate_potential_grasps(self, object_pose, object_class_id):
        """Generate potential grasp poses for the object"""
        grasps = []
        
        # For different object classes, generate appropriate grasp types
        if object_class_id in [0, 1, 2]:  # Bottles, cans, etc. (cylindrical)
            # Generate side grasps
            for angle in np.linspace(0, 2*np.pi, 8):  # 8 grasp angles
                grasp = self._create_side_grasp(object_pose, angle)
                grasps.append(grasp)
        
        elif object_class_id in [3, 4]:  # Cuboids
            # Generate corner or edge grasps
            grasp = self._create_top_grasp(object_pose)  # Top grasp for cuboids
            grasps.append(grasp)
        
        else:  # Generic approach
            # Generate multiple grasp candidates
            for i in range(5):
                grasp = self._create_generic_grasp(object_pose, i)
                grasps.append(grasp)
        
        return grasps
    
    def _create_side_grasp(self, object_pose, angle):
        """Create a side grasp for cylindrical objects"""
        # Position gripper perpendicular to the cylinder axis
        grasp_pose = copy.deepcopy(object_pose)
        
        # Calculate grasp approach direction
        approach_x = np.cos(angle)
        approach_y = np.sin(angle)
        
        # Adjust position to be at the object surface
        grasp_pose.position.x += approach_x * 0.05  # Offset for object radius
        grasp_pose.position.y += approach_y * 0.05
        
        # Set orientation to grasp along the cylinder axis
        q = R.from_euler('z', angle).as_quat()  # Grasp direction
        q = [q[0], q[1], q[2], q[3]]  # [x, y, z, w]
        grasp_pose.orientation = Quaternion(*q)
        
        return grasp_pose
    
    def _create_top_grasp(self, object_pose):
        """Create a top-down grasp for cuboidal objects"""
        grasp_pose = copy.deepcopy(object_pose)
        
        # Approach from above
        grasp_pose.position.z += 0.05  # 5cm above object
        
        # Set orientation for top-down grasp (typically aligned with world z)
        grasp_pose.orientation = Quaternion(0, 0, 0, 1)  # World-aligned
        
        return grasp_pose
    
    def _evaluate_grasp_quality(self, grasp_pose, object_pose):
        """Evaluate the quality of a grasp pose"""
        # This would involve complex computation in a real system
        # For this example, return a simple quality score
        
        # Simple scoring based on:
        # - Distance from object center
        # - Grasp orientation appropriateness
        # - Clearance from obstacles (not implemented)
        
        dist_from_center = np.linalg.norm([
            grasp_pose.position.x - object_pose.position.x,
            grasp_pose.position.y - object_pose.position.y
        ])
        
        # Prefer grasps close to object center
        dist_score = max(0, 1.0 - dist_from_center / 0.1)  # Score drops beyond 10cm
        
        # Prefer certain orientations based on object type
        orient_score = 0.8  # Simplified
        
        quality = 0.6 * dist_score + 0.4 * orient_score
        return quality
```

## Grasping and Manipulation Vision

### 3D Object Pose Estimation

```python
class ObjectPoseEstimator:
    """Estimate 6D object pose (position and orientation) using vision"""
    
    def __init__(self):
        # Pre-trained 6D pose estimation network
        # In practice, you might use networks like PVNet, Pix2Pose, etc.
        self.pose_model = self._load_pose_estimation_model()
        
        # Object models for template matching
        self.object_models = {}  # 3D models of objects for ICP refinement
        
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Pose refinement parameters
        self.icp_iterations = 10
        self.icp_tolerance = 1e-4
    
    def estimate_object_poses(self, image, detections, depth_image=None):
        """Estimate 6D poses for detected objects in the image"""
        results = []
        
        for detection in detections:
            bbox = detection.bbox
            obj_class = detection.results[0].hypothesis.id
            confidence = detection.results[0].hypothesis.score
            
            if confidence < 0.5:  # Skip low-confidence detections
                continue
            
            # Crop image around detection
            x1, y1, x2, y2 = int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)
            cropped_img = image[y1:y2, x1:x2]
            
            # Estimate pose using the model
            pose_6d = self._estimate_pose_6d(cropped_img, obj_class, bbox)
            
            if pose_6d is not None:
                # Refine pose using depth information if available
                if depth_image is not None:
                    pose_6d = self._refine_pose_with_depth(
                        pose_6d, cropped_img, depth_image, bbox
                    )
                
                results.append({
                    'object_class': obj_class,
                    'pose_6d': pose_6d,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        return results
    
    def _estimate_pose_6d(self, cropped_image, object_class, bbox):
        """Estimate 6D pose using deep learning model"""
        # This would use a pre-trained model like PVNet or similar
        # For this implementation, we'll use a simplified approach
        
        # Get keypoints or features from the image
        keypoints = self._extract_keypoints(cropped_image)
        
        # Match to known object model to get pose
        pose = self._match_to_object_model(keypoints, object_class)
        
        # Convert to 6D format [x, y, z, qx, qy, qz, qw]
        if pose is not None:
            return {
                'position': [pose['x'], pose['y'], pose['z']],
                'orientation': [pose['qx'], pose['qy'], pose['qz'], pose['qw']],
                'confidence': pose['confidence']
            }
        
        return None
    
    def _extract_keypoints(self, image):
        """Extract 2D keypoints from image"""
        # This would use a CNN-based keypoint detector
        # For now, use classical feature detection as placeholder
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use classic detectors as fallback
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        return {
            'keypoints': keypoints,
            'descriptors': descriptors
        }
    
    def _match_to_object_model(self, features, obj_class):
        """Match extracted features to object model for pose estimation"""
        # This would use learned object models
        # For now, return a dummy pose
        # In practice, this would use PnP, ICP, or other pose estimation algorithms
        if obj_class in self.object_models:
            # Use stored object model
            model = self.object_models[obj_class]
            
            # Use Perspective-n-Point for pose estimation
            # 3D model points
            model_points = model['points']
            # 2D image points (from features)
            img_points = np.array([kp.pt for kp in features['keypoints']], dtype=np.float32)
            
            if len(img_points) >= 4:  # Need at least 4 points for PnP
                success, rvec, tvec = cv2.solvePnP(
                    model_points, 
                    img_points, 
                    self.camera_matrix, 
                    self.dist_coeffs
                )
                
                if success:
                    # Convert rotation vector to quaternion
                    rotation_matrix = cv2.Rodrigues(rvec)[0]
                    quaternion = R.from_matrix(rotation_matrix).as_quat()
                    
                    return {
                        'x': tvec[0, 0],
                        'y': tvec[1, 0], 
                        'z': tvec[2, 0],
                        'qx': quaternion[0],
                        'qy': quaternion[1], 
                        'qz': quaternion[2],
                        'qw': quaternion[3],
                        'confidence': 0.8  # Dummy confidence
                    }
        
        # If no match found, return None
        return None
    
    def _refine_pose_with_depth(self, initial_pose, rgb_image, depth_image, bbox):
        """Refine initial pose using depth information"""
        # Convert initial pose to transformation matrix
        pos = initial_pose['position']
        quat = initial_pose['orientation']
        R_mat = R.from_quat(quat).as_matrix()
        
        T_init = np.eye(4)
        T_init[:3, :3] = R_mat
        T_init[:3, 3] = pos
        
        # Get depth points in the bounding box
        x1, y1, x2, y2 = int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)
        depth_roi = depth_image[y1:y2, x1:x2]
        
        # Convert depth to 3D points
        points_3d = self._depth_to_3d(depth_roi, bbox)
        
        if len(points_3d) == 0:
            return initial_pose
        
        # Get corresponding model points
        if initial_pose['object_class'] in self.object_models:
            model_points = self.object_models[initial_pose['object_class']]['points']
            
            # ICP refinement
            T_refined = self._icp_refinement(T_init, model_points, points_3d)
            
            # Convert back to position/orientation format
            refined_pos = T_refined[:3, 3]
            refined_rot = R.from_matrix(T_refined[:3, :3]).as_quat()
            
            return {
                'position': refined_pos.tolist(),
                'orientation': refined_rot.tolist(),
                'confidence': initial_pose['confidence'] + 0.1  # Increase confidence after refinement
            }
        
        return initial_pose
    
    def _depth_to_3d(self, depth_image, bbox):
        """Convert depth image ROI to 3D points"""
        h, w = depth_image.shape
        points_3d = []
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Convert to camera frame
        x_3d = (x_coords - self.camera_matrix[0, 2]) * depth_image / self.camera_matrix[0, 0]
        y_3d = (y_coords - self.camera_matrix[1, 2]) * depth_image / self.camera_matrix[1, 1]
        z_3d = depth_image
        
        # Flatten and combine
        valid_points = z_3d > 0  # Filter invalid depth values (0)
        points_3d = np.stack([
            x_3d[valid_points],
            y_3d[valid_points], 
            z_3d[valid_points]
        ], axis=1)
        
        return points_3d
    
    def _icp_refinement(self, T_initial, model_points, observed_points):
        """Iterative Closest Point for pose refinement"""
        T = T_initial.copy()
        prev_error = float('inf')
        
        for i in range(self.icp_iterations):
            # Transform model points to current estimate
            transformed_model = self._transform_points(model_points, T)
            
            # Find closest points between transformed model and observations
            distances, indices = self._find_closest_points(transformed_model, observed_points)
            
            # Calculate pose update using SVD
            T_update = self._calculate_pose_update(transformed_model[indices], observed_points)
            
            # Apply update
            T = T_update @ T
            
            # Check convergence
            current_error = np.mean(distances)
            if abs(prev_error - current_error) < self.icp_tolerance:
                break
            prev_error = current_error
        
        return T
    
    def _transform_points(self, points, transformation):
        """Transform 3D points using 4x4 transformation matrix"""
        # Add homogeneous coordinate
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        # Apply transformation
        transformed_homo = (transformation @ points_homo.T).T
        # Remove homogeneous coordinate
        return transformed_homo[:, :3]
    
    def _find_closest_points(self, points1, points2):
        """For each point in points1, find closest point in points2"""
        from scipy.spatial.distance import cdist
        
        # Compute distance matrix
        dist_matrix = cdist(points1, points2)
        
        # Find closest points
        closest_indices = np.argmin(dist_matrix, axis=1)
        closest_distances = np.min(dist_matrix, axis=1)
        
        return closest_distances, closest_indices
    
    def _calculate_pose_update(self, model_points, obs_points):
        """Calculate pose update using SVD-based method"""
        # Centroid alignment
        centroid_model = np.mean(model_points, axis=0)
        centroid_obs = np.mean(obs_points, axis=0)
        
        # Center the points
        centered_model = model_points - centroid_model
        centered_obs = obs_points - centroid_obs
        
        # SVD to find rotation
        H = centered_model.T @ centered_obs
        U, _, Vt = np.linalg.svd(H)
        
        # Compute rotation matrix
        R = Vt.T @ U.T
        # Ensure it's a proper rotation matrix (not reflection)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = centroid_obs - R @ centroid_model
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T

# Grasping system that integrates all vision components
class VisionGuidedGraspingSystem:
    """Complete system for vision-guided grasping and manipulation"""
    
    def __init__(self):
        # Initialize all vision components
        self.detector = RealTimeVisionPipeline(
            model_path="models/yolo_6d_pose.pt"
        )
        self.pose_estimator = ObjectPoseEstimator()
        self.grasp_planner = GraspPoseEstimator()
        self.controller = ManipulationVisionController()
        
        # Initialize hand-eye calibration
        self.calibrator = HandEyeCalibration()
        
        # System states
        self.system_state = 'idle'
        self.tracked_objects = {}
        
        rospy.loginfo("Vision-guided grasping system initialized")
    
    def run_grasp_pipeline(self, target_object_class):
        """Run complete grasping pipeline"""
        rospy.loginfo(f"Starting grasp pipeline for object class: {target_object_class}")
        
        # 1. Detect objects in scene
        detections = self._wait_for_detections(target_object_class)
        if not detections:
            rospy.logerr("No target object detected")
            return False
        
        # 2. Estimate object poses
        object_poses = self.pose_estimator.estimate_object_poses(
            self.current_image, detections, self.current_depth
        )
        
        # 3. Check if target object is present
        target_pose = None
        for obj_pose in object_poses:
            if obj_pose['object_class'] == target_object_class:
                target_pose = obj_pose['pose_6d']
                break
        
        if target_pose is None:
            rospy.logerr(f"Target object class {target_object_class} not found with good pose")
            return False
        
        # 4. Plan grasp pose
        grasp_pose = self.grasp_planner.plan_grasp(
            self._pose_6d_to_ros(target_pose), target_object_class
        )
        
        if grasp_pose is None:
            rospy.logerr("No suitable grasp pose found")
            return False
        
        # 5. Execute manipulation
        success = self.controller.execute_grasp_task(grasp_pose)
        
        return success
    
    def _wait_for_detections(self, target_class):
        """Wait for target object to be detected"""
        timeout = rospy.Time.now() + rospy.Duration(10.0)  # 10 second timeout
        
        while rospy.Time.now() < timeout:
            # Check if we have recent detections with the target class
            if hasattr(self, 'last_detections'):
                for det in self.last_detections:
                    if det.results[0].hypothesis.id == target_class:
                        return [det]  # Return first detection of target class
            
            rospy.sleep(0.1)  # Wait 100ms
        
        return []  # Timeout - no detections found
    
    def _pose_6d_to_ros(self, pose_6d):
        """Convert 6D pose format to ROS Pose message"""
        ros_pose = Pose()
        pos = pose_6d['position']
        quat = pose_6d['orientation']
        
        ros_pose.position.x = pos[0]
        ros_pose.position.y = pos[1]
        ros_pose.position.z = pos[2]
        
        ros_pose.orientation.x = quat[0]
        ros_pose.orientation.y = quat[1]
        ros_pose.orientation.z = quat[2]
        ros_pose.orientation.w = quat[3]
        
        return ros_pose

if __name__ == "__main__":
    rospy.init_node('vision_guided_grasping_system')
    
    # Example usage
    grasping_system = VisionGuidedGraspingSystem()
    
    # Example: Grasp a bottle (class ID 0)
    success = grasping_system.run_grasp_pipeline(target_object_class=0)
    
    if success:
        rospy.loginfo("Grasping task completed successfully!")
    else:
        rospy.logerr("Grasping task failed.")
    
    rospy.spin()
```

## Exercises

1. **Object Recognition Exercise**: Implement a real-time object detection system using YOLO or similar architecture that identifies and tracks objects relevant to a pick-and-place task.

2. **Visual Servoing Exercise**: Create a visual servoing controller that uses image-based feedback to position a robot end-effector over a target object.

3. **Grasping Vision Exercise**: Develop a system that combines 2D object detection with 3D pose estimation to generate appropriate grasp poses for different object shapes.

## Summary

This chapter covered computer vision integration in robotics, focusing on real-time object recognition, visual servoing systems, hand-eye coordination, and grasping vision. We explored how modern deep learning techniques enable robots to perceive and understand their environment, how visual feedback can guide robot motion through visual servoing, and how coordinated vision and manipulation systems enable complex tasks.

The key takeaways include:
- Real-time object detection requires optimized architectures like MobileNet or efficient detection networks
- Visual servoing provides feedback control for precise robot positioning
- Hand-eye calibration is essential for coordinate transformations between vision and manipulation
- 3D pose estimation enables full 6D object localization for grasping
- Successful integration requires careful consideration of hardware constraints and real-time performance

## Cross-references

For foundational concepts about Physical AI, see [Chapter 1: Introduction to Physical AI](../part-01-foundations/chapter-1). For sensor systems, see [Chapter 2: The Robotic Sensorium](../part-01-foundations/chapter-2). For related perception systems, see [Chapter 11: Advanced Perception with Isaac](../part-04-isaac/chapter-11). For navigation aspects using vision, see [Chapter 13: Navigation and Path Planning](../part-04-isaac/chapter-13).