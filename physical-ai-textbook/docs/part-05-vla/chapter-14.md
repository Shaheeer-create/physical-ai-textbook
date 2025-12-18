---
title: "Chapter 14: Voice-to-Action Systems"
description: "Integrating speech recognition and voice command processing for robotics"
sidebar_position: 1
---

# Chapter 14: Voice-to-Action Systems

## Learning Objectives

After completing this chapter, you should be able to:

- Integrate OpenAI Whisper for speech recognition in robotics applications
- Design voice command processing pipelines
- Implement natural language understanding for robotics
- Create multi-language support for voice commands

## Introduction to Voice Command Systems in Robotics

### The Importance of Voice Interfaces

Voice interfaces offer several advantages for robotics applications:

1. **Natural Interaction**: Voice is a familiar and intuitive communication method
2. **Hands-Free Operation**: Particularly valuable when users are occupied with other tasks
3. **Accessibility**: Provides access to users with mobility limitations
4. **Social Engagement**: Makes robots more approachable and engaging
5. **Efficiency**: Allows rapid command input without physical interface navigation

### Voice Command Architecture

A typical voice-command system in robotics includes:

1. **Audio Reception**: Microphone array for capturing speech
2. **Speech Recognition**: Converting audio to text
3. **Natural Language Understanding**: Interpreting the meaning of text
4. **Action Mapping**: Converting understood commands to robot actions
5. **Response Generation**: Providing feedback to the user

## OpenAI Whisper Integration

### Overview of Whisper

Whisper is OpenAI's automatic speech recognition (ASR) system trained on a large dataset of diverse audio. Its key advantages for robotics include:

1. **Multilingual Support**: Supports multiple languages in a single model
2. **Robustness**: Performs well in various acoustic conditions
3. **Timestamping**: Provides timing information for speech segments
4. **Transcription Quality**: High accuracy across different accents and speaking styles
5. **Open Source**: Can be deployed on edge devices

### Whisper Architecture for Robotics

```python
import whisper
import torch
import numpy as np
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class WhisperRobotInterface:
    def __init__(self):
        # Initialize Whisper model
        # For edge deployment, use smaller models like "small" or "medium"
        self.model = whisper.load_model("small.en")  # English model for efficiency
        
        # Initialize ROS components
        self.voice_command_publisher = rospy.Publisher('/voice_commands', String, queue_size=10)
        self.robot_command_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.chunk_size = 1024  # Process audio in chunks
        self.silence_threshold = 0.01  # Threshold for detecting speech
        self.min_speech_duration = 500  # Minimum speech duration in milliseconds
        
        # Command recognition parameters
        self.confidence_threshold = 0.7  # Minimum confidence for accepting commands
        self.language = "en"  # Default language
        
        # Robot command mappings
        self.command_mappings = {
            "move forward": self.move_forward,
            "go forward": self.move_forward,
            "move back": self.move_backward,
            "go back": self.move_backward,
            "turn left": self.turn_left,
            "turn right": self.turn_right,
            "stop": self.stop_robot,
            "halt": self.stop_robot,
            "go faster": self.increase_speed,
            "speed up": self.increase_speed,
            "slow down": self.decrease_speed,
            "rotate": self.rotate_robot
        }
        
        # Initialize audio recording
        self.setup_audio_recording()
        
    def setup_audio_recording(self):
        """Set up audio recording using PyAudio or similar library"""
        import pyaudio
        
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        rospy.loginfo("Audio recording initialized")
    
    def detect_speech_activity(self, audio_chunk):
        """Detect speech activity in audio chunk"""
        # Convert to numpy array if needed
        if not isinstance(audio_chunk, np.ndarray):
            audio_chunk = np.frombuffer(audio_chunk, dtype=np.float32)
        
        # Calculate energy of the chunk
        energy = np.mean(np.abs(audio_chunk) ** 2)
        
        # Determine if chunk contains speech
        is_speech = energy > self.silence_threshold
        
        return is_speech, energy
    
    def collect_speech_segment(self):
        """Collect a complete speech segment"""
        rospy.loginfo("Listening for voice command...")
        
        # Buffer to store audio chunks
        audio_buffer = []
        speaking = False
        silence_count = 0
        max_silence_chunks = int(0.5 * self.sample_rate / self.chunk_size)  # 0.5 seconds of silence
        
        start_time = rospy.Time.now()
        
        while not rospy.is_shutdown():
            # Read audio chunk
            chunk_data = self.stream.read(self.chunk_size)
            is_speech, energy = self.detect_speech_activity(chunk_data)
            
            if is_speech:
                speaking = True
                audio_buffer.extend(np.frombuffer(chunk_data, dtype=np.float32))
                silence_count = 0
            elif speaking:
                # Still collecting after speech stops
                audio_buffer.extend(np.frombuffer(chunk_data, dtype=np.float32))
                silence_count += 1
                
                if silence_count >= max_silence_chunks:
                    # Speech segment is complete
                    break
            else:
                # Waiting for speech to start
                continue
        
        duration = (rospy.Time.now() - start_time).to_sec()
        
        if len(audio_buffer) > 0 and duration >= self.min_speech_duration / 1000.0:
            rospy.loginfo(f"Collected speech segment of {duration:.2f}s")
            return np.array(audio_buffer, dtype=np.float32)
        else:
            rospy.loginfo("No valid speech segment collected")
            return None
    
    def transcribe_audio(self, audio_array):
        """Transcribe audio using Whisper"""
        try:
            # Process audio with Whisper
            result = self.model.transcribe(
                audio_array,
                language=self.language,
                temperature=0.0,  # Deterministic output
                compression_ratio_threshold=2.4,  # Filter low-quality outputs
                logprob_threshold=-1.0,  # Filter low probability outputs
                no_speech_threshold=0.6,  # Threshold for no-speech detection
            )
            
            transcription = result["text"].strip()
            confidence = result.get("avg_logprob", -2.0)  # Default confidence value
            
            rospy.loginfo(f"Transcribed: '{transcription}' (confidence: {confidence:.3f})")
            
            return transcription, confidence
        except Exception as e:
            rospy.logerr(f"Error in Whisper transcription: {e}")
            return "", -2.0
    
    def process_voice_command(self, text, confidence):
        """Process the transcribed text and map to robot commands"""
        if confidence < self.confidence_threshold:
            rospy.logwarn(f"Low confidence transcription ignored: {confidence}")
            return
        
        # Normalize the text
        normalized_text = text.lower().strip()
        
        # Find the best matching command
        best_match = None
        best_score = 0.0
        
        for command_phrase, command_func in self.command_mappings.items():
            similarity = self.calculate_similarity(normalized_text, command_phrase)
            if similarity > best_score:
                best_score = similarity
                best_match = command_func
        
        # Execute command if match is strong enough
        if best_score > 0.6:  # Threshold for command matching
            rospy.loginfo(f"Executing command: {list(self.command_mappings.keys())[list(self.command_mappings.values()).index(best_match)]}")
            best_match()
        else:
            # Try to use LLM for more complex command interpretation
            self.process_complex_command(normalized_text)
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two text strings"""
        # Simple word overlap similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    def process_complex_command(self, text):
        """Use LLM for complex command interpretation"""
        # For complex commands, we can use an LLM to interpret more nuanced instructions
        # This is simplified; in practice, you'd use a more sophisticated NLP system
        
        rospy.loginfo(f"Processing complex command: {text}")
        
        # Example: Look for keywords that indicate complex commands
        if "to the" in text or "navigate" in text:
            # This might be a navigation command
            self.handle_navigation_command(text)
        elif "find" in text or "locate" in text:
            # This might be an object finding command
            self.handle_object_finding_command(text)
        elif "bring" in text or "give" in text:
            # This might be an object manipulation command
            self.handle_manipulation_command(text)
        else:
            rospy.loginfo(f"Unknown command: {text}")
            self.say_response("I don't understand that command.")
    
    def run_voice_interface(self):
        """Main loop for voice command processing"""
        rospy.loginfo("Starting voice command interface")
        
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown():
            try:
                # Collect speech segment
                audio_data = self.collect_speech_segment()
                
                if audio_data is not None:
                    # Transcribe audio
                    text, confidence = self.transcribe_audio(audio_data)
                    
                    # Publish raw transcription
                    transcription_msg = String()
                    transcription_msg.data = f"{text} (conf: {confidence:.3f})"
                    self.voice_command_publisher.publish(transcription_msg)
                    
                    # Process voice command
                    self.process_voice_command(text, confidence)
                
            except Exception as e:
                rospy.logerr(f"Error in voice interface loop: {e}")
            
            rate.sleep()
    
    # Robot command methods
    def move_forward(self):
        cmd = Twist()
        cmd.linear.x = 0.5  # Forward velocity
        self.robot_command_publisher.publish(cmd)
    
    def move_backward(self):
        cmd = Twist()
        cmd.linear.x = -0.5  # Backward velocity
        self.robot_command_publisher.publish(cmd)
    
    def turn_left(self):
        cmd = Twist()
        cmd.angular.z = 0.5  # Left turn
        self.robot_command_publisher.publish(cmd)
    
    def turn_right(self):
        cmd = Twist()
        cmd.angular.z = -0.5  # Right turn
        self.robot_command_publisher.publish(cmd)
    
    def stop_robot(self):
        cmd = Twist()
        # Zero velocities for stopping
        self.robot_command_publisher.publish(cmd)
    
    def increase_speed(self):
        # In a real implementation, this would increase the robot's speed
        rospy.loginfo("Speed increased")
    
    def decrease_speed(self):
        # In a real implementation, this would decrease the robot's speed
        rospy.loginfo("Speed decreased")
    
    def rotate_robot(self):
        cmd = Twist()
        cmd.angular.z = 0.3  # Rotate at 0.3 rad/s
        self.robot_command_publisher.publish(cmd)
    
    def say_response(self, response_text):
        """Method to provide verbal response (would integrate with text-to-speech)"""
        rospy.loginfo(f"Robot says: {response_text}")
    
    def handle_navigation_command(self, text):
        """Handle navigation-related commands"""
        rospy.loginfo(f"Processing navigation command: {text}")
        # Implementation would involve parsing destination and commanding navigation
        self.say_response("Navigating to destination.")
    
    def handle_object_finding_command(self, text):
        """Handle object finding commands"""
        rospy.loginfo(f"Processing object finding command: {text}")
        # Implementation would involve object detection and reporting
        self.say_response("Looking for the object.")
    
    def handle_manipulation_command(self, text):
        """Handle object manipulation commands"""
        rospy.loginfo(f"Processing manipulation command: {text}")
        # Implementation would involve grasp planning and execution
        self.say_response("Attempting to manipulate the object.")

if __name__ == "__main__":
    rospy.init_node("whisper_robot_interface")
    
    interface = WhisperRobotInterface()
    
    try:
        interface.run_voice_interface()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down voice interface")
    finally:
        # Clean up audio resources
        if hasattr(interface, 'stream'):
            interface.stream.stop_stream()
            interface.stream.close()
        
        if hasattr(interface, 'audio'):
            interface.audio.terminate()
```

### Whisper Model Optimization for Robotics

For edge deployment on robotics platforms:

```python
import whisper
import torch
import numpy as np

class OptimizedWhisperInterface:
    def __init__(self, model_size="small", device="cuda", compute_type="float16"):
        """
        Initialize Whisper with optimization for robotics
        
        Args:
            model_size: Size of Whisper model (tiny, base, small, medium, large)
            device: Device for inference ('cuda', 'cpu')
            compute_type: Precision for inference ('float16', 'float32')
        """
        # Determine device automatically if not specified
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"  # Use half precision on GPU
        else:
            self.device = "cpu"
            self.compute_type = "float32"  # Use full precision on CPU
        
        # Load model with specific optimizations
        self.model = whisper.load_model(model_size, device=self.device, in_memory=True)
        
        # Set model to evaluation mode to disable dropout
        self.model.eval()
        
        # If using CUDA, enable mixed precision for efficiency
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            self.model = self.model.half() if compute_type == "float16" else self.model
        
        # Audio parameters
        self.sample_rate = 16000
        self.model_sample_rate = 16000  # Whisper expects 16kHz
        self.max_input_length = 30  # Maximum audio length in seconds for processing
        self.chunk_size = 16000 * 5  # Process in 5-second chunks for long audio
        
        # Transcription parameters for robotics use
        self.transcription_opts = {
            "language": "en",
            "temperature": 0.0,  # Deterministic output
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "beam_size": 5,  # Use beam search for better accuracy
            "best_of": 5,    # Consider multiple hypotheses
        }
    
    def preprocess_audio(self, audio_data):
        """
        Preprocess audio for Whisper
        
        Args:
            audio_data: Raw audio data as numpy array
        
        Returns:
            Normalized audio tensor ready for Whisper
        """
        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize audio to [-1, 1] range
        audio_data = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) != 0 else audio_data
        
        # Pad/truncate to appropriate length for Whisper
        target_samples = self.max_input_length * self.model_sample_rate
        if len(audio_data) > target_samples:
            # Truncate if too long
            rospy.logwarn(f"Audio too long ({len(audio_data) / self.sample_rate:.2f}s), truncating to {self.max_input_length}s")
            audio_data = audio_data[:target_samples]
        elif len(audio_data) < self.model_sample_rate:  # At least 1 second
            # Pad if too short
            padding = np.zeros(target_samples - len(audio_data), dtype=np.float32)
            audio_data = np.concatenate([audio_data, padding])
        
        # Convert to PyTorch tensor and move to appropriate device
        audio_tensor = torch.from_numpy(audio_data).to(self.device)
        
        return audio_tensor
    
    def transcribe_audio_efficiently(self, audio_data, language=None):
        """
        Efficiently transcribe audio with error handling and optimization
        
        Args:
            audio_data: Audio data as numpy array
            language: Language code override (optional)
        
        Returns:
            Tuple of (text, confidence)
        """
        try:
            # Preprocess audio
            audio_tensor = self.preprocess_audio(audio_data)
            
            # Encode audio using Whisper encoder
            mel = whisper.log_mel_spectrogram(audio_tensor).to(self.device)
            
            # Perform transcription
            options = dict(self.transcription_opts)
            if language:
                options["language"] = language
            
            result = self.model.transcribe(
                audio_tensor,
                **options
            )
            
            transcription = result["text"].strip()
            
            # Calculate confidence from log probabilities
            confidence = result.get("avg_logprob", -2.0)
            
            # Convert log probability to a more interpretable confidence score
            # Log prob of 0 is perfect, very negative is poor
            confidence_score = max(0.0, min(1.0, 1.0 + confidence))  # Normalize to [0, 1]
            
            return transcription, confidence_score
            
        except Exception as e:
            rospy.logerr(f"Error in efficient transcription: {e}")
            return "", 0.0
    
    def transcribe_long_audio(self, audio_data, language=None):
        """
        Transcribe long audio by processing in chunks
        
        Args:
            audio_data: Long audio data as numpy array
            language: Language code (optional)
        
        Returns:
            Full transcription of the audio
        """
        if len(audio_data) <= self.max_input_length * self.sample_rate:
            # Audio is short enough, process normally
            return self.transcribe_audio_efficiently(audio_data, language)
        
        # Process in chunks
        total_samples = len(audio_data)
        chunk_duration = self.max_input_length
        chunk_samples = chunk_duration * self.sample_rate
        
        full_transcription = ""
        confidence_scores = []
        
        for i in range(0, total_samples, chunk_samples):
            chunk = audio_data[i:i+chunk_samples]
            
            # Add some overlap to maintain context
            if i > 0 and len(chunk) < chunk_samples:
                # Include some overlap from previous chunk if this is the last chunk
                overlap_start = max(0, i - int(0.5 * self.sample_rate))
                chunk = audio_data[overlap_start:i+chunk_samples]
            
            # Transcribe chunk
            text, conf = self.transcribe_audio_efficiently(chunk, language)
            
            # Only add to full transcription if it has content
            if text.strip():
                if full_transcription:
                    full_transcription += " " + text
                else:
                    full_transcription = text
            
            if conf > 0.0:  # Only include valid confidence scores
                confidence_scores.append(conf)
        
        # Calculate average confidence for the whole transcription
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return full_transcription, avg_confidence
```

## Voice Command Processing Pipeline

### Natural Language Understanding (NLU)

For more sophisticated command interpretation:

```python
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class CommandType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    INFORMATION = "information"
    SYSTEM = "system"

@dataclass
class CommandIntent:
    command_type: CommandType
    action: str
    parameters: Dict[str, any]
    confidence: float

class RobotNLU:
    def __init__(self):
        # Define command patterns with regex
        self.patterns = {
            CommandType.NAVIGATION: [
                (r"go\s+(?P<direction>forward|backward|left|right)", "move"),
                (r"move\s+(?P<direction>forward|backward|left|right)", "move"),
                (r"turn\s+(?P<direction>left|right)", "turn"),
                (r"stop", "stop"),
                (r"go\s+to\s+(?P<location>\w+)", "navigate_to"),
                (r"bring\s+me\s+(?P<object>\w+)", "fetch_object"),
            ],
            CommandType.MANIPULATION: [
                (r"pick\s+up(?P<object>\s+\w+)", "pick_up"),
                (r"grasp(?P<object>\s+\w+)", "grasp"),
                (r"place\s+(?P<object>\s+\w+)\s+on\s+(?P<surface>\w+)", "place_on"),
                (r"put(?P<object>\s+\w+)\s+down", "put_down"),
            ],
            CommandType.INFORMATION: [
                (r"tell\s+me\s+about\s+(?P<topic>\w+)", "describe"),
                (r"what\s+is\s+(?P<object>\w+)", "identify"),
                (r"how\s+many\s+(?P<item>\w+)", "count"),
                (r"where\s+is\s+(?P<object>\w+)", "locate_object"),
            ],
            CommandType.SYSTEM: [
                (r"power\s+off", "shutdown"),
                (r"power\s+down", "shutdown"),
                (r"restart", "restart"),
                (r"reset", "reset"),
                (r"status", "report_status"),
                (r"battery\s+level", "report_battery"),
            ]
        }
        
        # Location mappings
        self.location_synonyms = {
            "kitchen": ["kitchen", "cooking area", "food prep"],
            "living room": ["living room", "lounge", "sitting area"],
            "bedroom": ["bedroom", "sleeping area", "bed area"],
            "office": ["office", "study", "workspace"],
            "entrance": ["entrance", "door", "entryway", "hallway"]
        }
        
        # Object mappings
        self.object_synonyms = {
            "water": ["water", "bottle of water", "water bottle"],
            "apple": ["apple", "red fruit", "fruit"],
            "book": ["book", "reading material", "novel"],
            "phone": ["phone", "smartphone", "mobile device"]
        }
    
    def parse_command(self, text: str) -> Optional[CommandIntent]:
        """Parse natural language command into structured intent"""
        text = text.lower().strip()
        
        best_intent = None
        best_confidence = 0.0
        
        # Try each pattern
        for command_type, patterns in self.patterns.items():
            for pattern, action in patterns:
                match = re.search(pattern, text)
                if match:
                    # Calculate confidence based on pattern match quality
                    confidence = self.calculate_pattern_confidence(pattern, text, match)
                    
                    # Extract parameters
                    params = {}
                    for param_name, param_value in match.groupdict().items():
                        if param_value:
                            # Handle synonyms
                            if param_name == "location":
                                params[param_name] = self.resolve_location_synonym(param_value.strip())
                            elif param_name == "object":
                                params[param_name] = self.resolve_object_synonym(param_value.strip())
                            else:
                                params[param_name] = param_value.strip()
                    
                    intent = CommandIntent(
                        command_type=command_type,
                        action=action,
                        parameters=params,
                        confidence=confidence
                    )
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = intent
        
        # If no pattern matched, try semantic analysis with LLM
        if not best_intent:
            best_intent = self.semantic_analysis(text)
        
        return best_intent
    
    def calculate_pattern_confidence(self, pattern: str, text: str, match) -> float:
        """Calculate confidence score for pattern match"""
        # Base confidence on pattern complexity and text match coverage
        pattern_tokens = len(pattern.split())
        text_tokens = len(text.split())
        
        # More specific patterns get higher confidence
        base_conf = 0.7
        
        # Adjust confidence based on how much of the text is matched
        matched_text = match.group(0)
        match_coverage = len(matched_text) / len(text)
        coverage_adjustment = min(1.0, match_coverage * 1.2)  # Slightly favor longer matches
        
        confidence = base_conf * coverage_adjustment
        return min(1.0, confidence)
    
    def resolve_location_synonym(self, location_text: str) -> str:
        """Resolve location synonyms to canonical form"""
        for canonical, synonyms in self.location_synonyms.items():
            if location_text in synonyms:
                return canonical
        return location_text
    
    def resolve_object_synonym(self, object_text: str) -> str:
        """Resolve object synonyms to canonical form"""
        for canonical, synonyms in self.object_synonyms.items():
            if object_text in synonyms:
                return canonical
        return object_text
    
    def semantic_analysis(self, text: str) -> Optional[CommandIntent]:
        """Use semantic analysis for complex commands (simplified version)"""
        # This would normally use an LLM to understand complex commands
        # For now, implement basic semantic analysis
        
        text_lower = text.lower()
        
        # Look for keywords that indicate command types
        if any(word in text_lower for word in ["go", "move", "navigate", "to"]):
            return CommandIntent(
                CommandType.NAVIGATION,
                "navigate",
                {"destination": self.extract_destination(text)},
                0.6
            )
        elif any(word in text_lower for word in ["pick", "grasp", "take", "get"]):
            return CommandIntent(
                CommandType.MANIPULATION,
                "manipulate",
                {"action": "grasp", "object": self.extract_object(text)},
                0.6
            )
        elif any(word in text_lower for word in ["tell", "what", "how", "where"]):
            return CommandIntent(
                CommandType.INFORMATION,
                "inform",
                {"query": text},
                0.6
            )
        
        return None
    
    def extract_destination(self, text: str) -> str:
        """Extract destination from navigation command"""
        # Simple extraction - look for location words
        for loc_synonyms in self.location_synonyms.values():
            for synonym in loc_synonyms:
                if synonym in text.lower():
                    # Find the canonical form
                    for canonical, syns in self.location_synonyms.items():
                        if synonym in syns:
                            return canonical
        return "unknown location"
    
    def extract_object(self, text: str) -> str:
        """Extract object from manipulation command"""
        # Simple extraction - look for object words
        for obj_synonyms in self.object_synonyms.values():
            for synonym in obj_synonyms:
                if synonym in text.lower():
                    # Find the canonical form
                    for canonical, syns in self.object_synonyms.items():
                        if synonym in syns:
                            return canonical
        return "unknown object"

# Voice command processor that integrates Whisper and NLU
class VoiceCommandProcessor:
    def __init__(self):
        self.whisper_interface = OptimizedWhisperInterface()
        self.nlu = RobotNLU()
        self.command_threshold = 0.6  # Minimum confidence for executing commands
    
    def process_speech_to_action(self, audio_data):
        """Process speech input to robot action"""
        # Transcribe audio
        transcription, confidence = self.whisper_interface.transcribe_audio_efficiently(audio_data)
        
        if not transcription.strip():
            return None, 0.0
        
        # Parse command intent
        intent = self.nlu.parse_command(transcription)
        
        if intent and intent.confidence >= self.command_threshold:
            rospy.loginfo(f"Recognized command: {intent.command_type.value} - {intent.action} "
                         f"with parameters: {intent.parameters} (confidence: {intent.confidence:.3f})")
            
            # Execute the command
            action_result = self.execute_command(intent)
            return action_result, intent.confidence
        else:
            rospy.loginfo(f"Command not recognized or low confidence: '{transcription}' "
                         f"(confidence: {confidence:.3f})")
            return None, confidence
    
    def execute_command(self, intent: CommandIntent):
        """Execute the parsed command intent"""
        # This would connect to the robot's action execution system
        # For now, just log what would be executed
        
        if intent.command_type == CommandType.NAVIGATION:
            return self.execute_navigation_command(intent)
        elif intent.command_type == CommandType.MANIPULATION:
            return self.execute_manipulation_command(intent)
        elif intent.command_type == CommandType.INFORMATION:
            return self.execute_information_command(intent)
        elif intent.command_type == CommandType.SYSTEM:
            return self.execute_system_command(intent)
    
    def execute_navigation_command(self, intent: CommandIntent):
        """Execute navigation commands"""
        if intent.action == "move":
            direction = intent.parameters.get("direction", "unknown")
            rospy.loginfo(f"Moving in direction: {direction}")
            # Implementation would send navigation commands to robot
        elif intent.action == "navigate_to":
            location = intent.parameters.get("location", "unknown")
            rospy.loginfo(f"Navigating to location: {location}")
            # Implementation would initiate navigation to location
        elif intent.action == "fetch_object":
            obj = intent.parameters.get("object", "unknown")
            location = intent.parameters.get("location", "current")
            rospy.loginfo(f"Fetching {obj} from {location}")
            # Implementation would navigate to location and fetch object
        
        return f"Navigation command executed: {intent.action}"
    
    def execute_manipulation_command(self, intent: CommandIntent):
        """Execute manipulation commands"""
        action = intent.action
        obj = intent.parameters.get("object", "unknown")
        
        rospy.loginfo(f"Manipulation command: {action} {obj}")
        
        # Implementation would control robot manipulator
        return f"Manipulation command executed: {action} {obj}"
    
    def execute_information_command(self, intent: CommandIntent):
        """Execute information commands"""
        query = intent.parameters.get("query", "unknown")
        
        rospy.loginfo(f"Information query: {query}")
        
        # Implementation would query robot sensors or knowledge base
        return f"Information provided about: {query}"
    
    def execute_system_command(self, intent: CommandIntent):
        """Execute system commands"""
        action = intent.action
        
        rospy.loginfo(f"System command: {action}")
        
        # Implementation would execute system-level commands
        return f"System command executed: {action}"

if __name__ == "__main__":
    # Example usage
    processor = VoiceCommandProcessor()
    
    # This would be called with actual audio data
    # result, conf = processor.process_speech_to_action(audio_data)
```

## Multi-Language Support Considerations

### Handling Multiple Languages

```python
class MultiLanguageVoiceInterface:
    def __init__(self):
        # Dictionary to hold models for different languages
        self.models = {}
        
        # Supported languages mapping
        self.language_codes = {
            "english": "en",
            "spanish": "es", 
            "french": "fr",
            "german": "de",
            "italian": "it",
            "portuguese": "pt",
            "russian": "ru",
            "chinese": "zh",
            "japanese": "ja",
            "korean": "ko"
        }
        
        # Initialize default English model
        self.initialize_language_model("en")
        
        # Language detection model or method
        self.language_detector = self.setup_language_detector()
    
    def setup_language_detector(self):
        """Set up language detection (could use a lightweight model or simple heuristics)"""
        # For simple implementation, we could use phonetic characteristics
        # or a basic language identification model
        # In practice, you might use a specialized library for this
        return lambda text: "en"  # Simplified - would have actual detection logic
    
    def initialize_language_model(self, lang_code, model_size="small"):
        """Initialize Whisper model for specific language"""
        if lang_code not in self.models:
            try:
                # Load appropriate model - for some languages, specific models may be better
                model = whisper.load_model(model_size)
                self.models[lang_code] = model
                rospy.loginfo(f"Loaded Whisper model for language: {lang_code}")
            except Exception as e:
                rospy.logerr(f"Failed to load model for language {lang_code}: {e}")
    
    def detect_and_transcribe(self, audio_data):
        """Detect language and transcribe accordingly"""
        # First, try to detect the language
        detected_lang = self.language_detector(audio_data)
        
        # Fallback to default if detection fails
        if not detected_lang or detected_lang not in self.models:
            detected_lang = "en"
        
        # Use the appropriate model for transcription
        model = self.models[detected_lang]
        
        # Process with Whisper
        result = model.transcribe(
            audio_data,
            language=detected_lang,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6
        )
        
        return result["text"], detected_lang
    
    def get_user_preferred_language(self):
        """Get user's preferred language (implementation would depend on your system)"""
        # This could be from a user profile, configuration file, etc.
        # For now, returning default
        return "en"
    
    def process_multilingual_command(self, audio_data, auto_detect=True):
        """Process command with multilingual support"""
        if auto_detect:
            # Detect language and transcribe
            text, detected_lang = self.detect_and_transcribe(audio_data)
            rospy.loginfo(f"Detected language: {detected_lang}")
        else:
            # Use user's preferred language
            preferred_lang = self.get_user_preferred_language()
            model = self.models.get(preferred_lang)
            
            if not model:
                rospy.logwarn(f"No model loaded for preferred language {preferred_lang}, using English")
                model = self.models.get("en")
                preferred_lang = "en"
            
            result = model.transcribe(audio_data, language=preferred_lang)
            text = result["text"]
            detected_lang = preferred_lang
        
        # Process the command through NLU
        nlu = RobotNLU()
        intent = nlu.parse_command(text)
        
        # The NLU component would need to be adapted for the detected language
        # For now, assuming English-based parsing works
        if intent:
            rospy.loginfo(f"Command in {detected_lang}: {text} -> {intent.action}")
            return intent
        else:
            rospy.loginfo(f"No command recognized in {detected_lang}: {text}")
            return None
```

## Practical Implementation: Voice-Controlled Robot

Here's a complete example of a voice-controlled robot system:

```python
#!/usr/bin/env python3

import rospy
import numpy as np
import pyaudio
import threading
import time
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class VoiceControlledRobot:
    def __init__(self):
        rospy.init_node('voice_controlled_robot')
        
        # Audio parameters
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.silence_threshold = 0.01
        self.min_speech_duration = 0.5  # seconds
        
        # Initialize components
        self.voice_processor = VoiceCommandProcessor()
        self.cv_bridge = CvBridge()
        
        # ROS publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.voice_feedback_pub = rospy.Publisher('/voice_feedback', String, queue_size=10)
        
        # Robot state
        self.current_speed = 0.3  # Default linear speed
        self.current_angular_speed = 0.5  # Default angular speed
        
        # Audio recording
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # Control flags
        self.listening_enabled = True
        self.recording_thread = None
        self.command_buffer_lock = threading.Lock()
        self.pending_commands = []
        
        rospy.loginfo("Voice Controlled Robot initialized")
    
    def start_listening(self):
        """Start the listening loop in a separate thread"""
        self.recording_thread = threading.Thread(target=self.recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        rospy.loginfo("Started voice command listening")
    
    def recording_loop(self):
        """Main loop for audio capture and command processing"""
        rospy.loginfo("Recording loop started")
        
        while not rospy.is_shutdown() and self.listening_enabled:
            try:
                # Collect audio until speech is detected
                audio_segment = self.wait_for_speech()
                
                if audio_segment is not None and len(audio_segment) > 0:
                    # Process the speech segment
                    self.process_audio_segment(audio_segment)
                
                # Small delay to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                rospy.logerr(f"Error in recording loop: {e}")
                time.sleep(0.5)  # Brief pause before retrying
    
    def wait_for_speech(self):
        """Wait for speech activity and collect the complete segment"""
        rospy.loginfo("Listening for speech...")
        
        audio_buffer = []
        speaking = False
        silence_count = 0
        max_silence_chunks = int(0.5 * self.sample_rate / self.chunk_size)
        
        start_time = time.time()
        max_duration = 5.0  # Max 5 seconds per command
        
        while not rospy.is_shutdown():
            # Check timeout
            if time.time() - start_time > max_duration:
                rospy.loginfo("Command timeout, processing collected audio")
                break
            
            # Read audio chunk
            try:
                chunk_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(chunk_data, dtype=np.float32)
                
                # Calculate energy
                energy = np.mean(np.abs(audio_chunk) ** 2)
                
                if energy > self.silence_threshold:
                    speaking = True
                    audio_buffer.extend(audio_chunk)
                    silence_count = 0
                elif speaking:
                    # Still collecting after speech stops
                    audio_buffer.extend(audio_chunk)
                    silence_count += 1
                    
                    if silence_count >= max_silence_chunks:
                        # Speech segment is complete
                        rospy.loginfo("Speech segment complete")
                        break
                else:
                    # Waiting for speech to start
                    continue
            except Exception as e:
                rospy.logerr(f"Error reading audio chunk: {e}")
                break
        
        if audio_buffer:
            return np.array(audio_buffer, dtype=np.float32)
        else:
            return None
    
    def process_audio_segment(self, audio_segment):
        """Process collected audio segment"""
        rospy.loginfo(f"Processing audio segment of {len(audio_segment) / self.sample_rate:.2f}s")
        
        # Publish feedback
        feedback_msg = String()
        feedback_msg.data = "Processing voice command..."
        self.voice_feedback_pub.publish(feedback_msg)
        
        # Process with voice command processor
        try:
            result, confidence = self.voice_processor.process_speech_to_action(audio_segment)
            
            if result:
                rospy.loginfo(f"Command executed: {result} (confidence: {confidence:.3f})")
                
                # Send confirmation feedback
                confirmation = String()
                confirmation.data = f"Command executed: {result}"
                self.voice_feedback_pub.publish(confirmation)
            else:
                rospy.loginfo("No valid command recognized")
                
                # Send error feedback
                error_msg = String()
                error_msg.data = "Could not understand the command. Please try again."
                self.voice_feedback_pub.publish(error_msg)
                
        except Exception as e:
            rospy.logerr(f"Error processing audio segment: {e}")
            
            # Send error feedback
            error_msg = String()
            error_msg.data = f"Error processing command: {str(e)}"
            self.voice_feedback_pub.publish(error_msg)
    
    def move_robot(self, linear_vel, angular_vel):
        """Send velocity commands to robot"""
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.cmd_vel_pub.publish(cmd)
    
    def execute_navigation_command(self, intent):
        """Execute navigation commands for the robot"""
        if intent.action == "move":
            direction = intent.parameters.get("direction", "forward")
            
            linear_vel = 0.0
            angular_vel = 0.0
            
            if direction == "forward":
                linear_vel = self.current_speed
            elif direction == "backward":
                linear_vel = -self.current_speed
            elif direction == "left":
                angular_vel = self.current_angular_speed
            elif direction == "right":
                angular_vel = -self.current_angular_speed
            
            self.move_robot(linear_vel, angular_vel)
            
        elif intent.action == "stop":
            self.move_robot(0.0, 0.0)
        
        elif intent.action == "navigate_to":
            location = intent.parameters.get("location", "unknown")
            rospy.loginfo(f"Beginning navigation to {location}")
            # In a real implementation, this would trigger navigation system
            # self.navigation_system.go_to_location(location)
    
    def run(self):
        """Main run loop"""
        rospy.loginfo("Starting voice-controlled robot")
        
        self.start_listening()
        
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutdown requested by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.listening_enabled = False
        
        if hasattr(self, 'recording_thread') and self.recording_thread:
            self.recording_thread.join(timeout=1.0)
        
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        
        if hasattr(self, 'audio'):
            self.audio.terminate()
        
        rospy.loginfo("Voice-controlled robot cleaned up")

if __name__ == "__main__":
    robot = VoiceControlledRobot()
    robot.run()
```

## Exercises

1. **Voice Command Implementation**: Create a complete voice command system that integrates Whisper ASR with a simple mobile robot, allowing voice commands to control robot movement.

2. **Language Support Exercise**: Extend the voice command system to support multiple languages, implementing language detection and multilingual command parsing.

3. **Complex Command Processing**: Implement natural language understanding for complex commands that involve multiple steps (e.g., "Go to the kitchen and bring me the red apple").

## Summary

This chapter covered voice-to-action systems for robotics, focusing on the integration of OpenAI Whisper for speech recognition and the development of voice command processing pipelines. We explored the architecture of voice interfaces in robotics, implemented Whisper integration with optimization for robotics platforms, designed natural language understanding systems, and created practical implementations for voice-controlled robots.

The key takeaways include:
- Voice interfaces provide natural and accessible robot control
- Whisper offers robust multilingual ASR suitable for robotics
- Natural language understanding bridges speech to robot actions
- Multi-language support is achievable with proper architecture
- Practical implementations require careful attention to audio processing and real-time constraints

## Cross-references

For foundational concepts about Physical AI, see [Chapter 1: Introduction to Physical AI](../part-01-foundations/chapter-1). For related perception systems, see [Chapter 2: The Robotic Sensorium](../part-01-foundations/chapter-2). For integration with AI systems, see [Chapter 15: Cognitive Planning with LLMs](../part-05-vla/chapter-15).