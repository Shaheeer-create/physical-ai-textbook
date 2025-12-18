---
title: "Chapter 18: Voice Command Processing"
description: "Implementing speech recognition and natural language processing for robot control"
sidebar_position: 5
---

# Chapter 18: Voice Command Processing

## Learning Objectives

After completing this chapter, you should be able to:

- Implement speech-to-text systems for real-time voice command processing
- Design natural language understanding pipelines for robotic tasks
- Integrate voice interfaces with robot control systems
- Create multimodal interaction systems combining voice and other modalities

## Introduction to Voice Command Systems

### The Importance of Voice Interaction

Voice interfaces are crucial for natural human-robot interaction because they:

1. **Enable Natural Communication**: People naturally communicate through speech
2. **Provide Hands-Free Operation**: Especially useful in situations where users need to perform other tasks
3. **Offer Accessibility**: Makes robots usable for individuals with mobility limitations
4. **Facilitate Social Interaction**: Creates more engaging and approachable robot personality
5. **Allow Multi-Modal Communication**: Works alongside gestural and visual communication

### Voice Command Architecture

A typical voice command system in robotics includes these components:

1. **Audio Input**: Microphones or microphone arrays for capturing speech
2. **Speech Recognition**: Converting audio to text (ASR - Automatic Speech Recognition)
3. **Natural Language Understanding (NLU)**: Interpreting the meaning of text
4. **Intent Mapping**: Converting understood meanings to robot actions
5. **Response Generation**: Providing audible feedback to the user
6. **Execution**: Carrying out robot actions based on voice commands

## Speech Recognition Systems

### ASR for Robotics

Automatic Speech Recognition (ASR) in robotics must be optimized for real-time performance and accuracy in noisy environments:

```python
import torch
import whisper
import numpy as np
import pyaudio
import threading
import queue
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class RobustASRSystem:
    """Robust Automatic Speech Recognition system for robotics applications"""
    
    def __init__(self, model_size="small.en", device="cuda"):
        # Initialize Whisper model for speech recognition
        self.model = whisper.load_model(model_size, device=device)
        
        # Audio parameters
        self.sample_rate = 16000
        self.chunk_size = 1024  # Process audio in chunks
        
        # Initialize audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # Threshold for speech detection
        self.silence_threshold = 0.01
        
        # Store recent audio for context
        self.audio_buffer = queue.Queue(maxsize=100)
        
        # ROS publishers and subscribers
        self.voice_publisher = rospy.Publisher('/voice_commands', String, queue_size=10)
        self.audio_feedback_publisher = rospy.Publisher('/audio_feedback', String, queue_size=10)
        
        rospy.loginfo("ASR system initialized successfully")
    
    def is_speech_present(self, audio_chunk):
        """Detect if speech is present in audio chunk"""
        energy = np.mean(np.abs(audio_chunk) ** 2)
        return energy > self.silence_threshold
    
    def collect_speech_segment(self, timeout=5.0):
        """Collect a complete speech segment from the user"""
        rospy.loginfo("Listening for voice command...")
        
        # Buffer for storing speech data
        speech_buffer = []
        silence_count = 0
        max_silence_chunks = int(0.5 * self.sample_rate / self.chunk_size)  # 0.5 seconds of silence
        timeout_chunks = int(timeout * self.sample_rate / self.chunk_size)
        chunk_count = 0
        
        # Collect audio until timeout or complete silence
        while chunk_count < timeout_chunks:
            chunk_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio_chunk = np.frombuffer(chunk_data, dtype=np.float32)
            
            if self.is_speech_present(audio_chunk):
                speech_buffer.extend(audio_chunk)
                silence_count = 0  # Reset silence counter
            elif len(speech_buffer) > 0:
                # We're in the middle of speech, accumulate some silence before ending
                speech_buffer.extend(audio_chunk)
                silence_count += 1
                
                if silence_count >= max_silence_chunks:
                    # Speech segment is complete
                    break
            
            chunk_count += 1
        
        if len(speech_buffer) > 0:
            rospy.loginfo(f"Collected speech segment of {len(speech_buffer) / self.sample_rate:.2f}s")
            return np.array(speech_buffer, dtype=np.float32)
        else:
            rospy.loginfo("No speech detected in timeout period")
            return None
    
    def transcribe_audio(self, audio_segment):
        """Transcribe audio segment to text using Whisper"""
        try:
            # Process with Whisper
            result = self.model.transcribe(
                audio_segment, 
                language='en',
                temperature=0.0,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
            )
            
            transcription = result["text"].strip()
            confidence = result.get("avg_logprob", -2.0)  # Average log probability as confidence
            
            rospy.loginfo(f"Transcribed: '{transcription}' (confidence: {confidence:.3f})")
            return transcription, confidence
            
        except Exception as e:
            rospy.logerr(f"Error during transcription: {e}")
            return "", -2.0
    
    def run_voice_interface(self):
        """Main loop for voice command processing"""
        rospy.loginfo("Voice command interface running")
        
        rate = rospy.Rate(10)  # 10 Hz processing
        
        while not rospy.is_shutdown():
            try:
                # Collect speech segment
                audio_segment = self.collect_speech_segment()
                
                if audio_segment is not None:
                    # Transcribe the audio
                    text, confidence = self.transcribe_audio(audio_segment)
                    
                    if text and confidence > -1.0:  # Reasonable confidence threshold
                        # Publish detected command
                        cmd_msg = String()
                        cmd_msg.data = text
                        self.voice_publisher.publish(cmd_msg)
                        
                        # Provide feedback
                        feedback_msg = String()
                        feedback_msg.data = f"Heard: {text}"
                        self.audio_feedback_publisher.publish(feedback_msg)
                        
                        # Process the command
                        self.process_command(text)
                    else:
                        rospy.logwarn("Low confidence transcription, ignoring")
                
            except Exception as e:
                rospy.logerr(f"Error in voice interface: {e}")
            
            rate.sleep()
    
    def process_command(self, text):
        """Process the transcribed text and map to robot actions"""
        # In a real system, this would use NLU to parse the command
        # For now, we'll implement some basic command recognition
        text_lower = text.lower()
        
        if "move forward" in text_lower or "go forward" in text_lower:
            self.execute_command("move_forward")
        elif "move backward" in text_lower or "go back" in text_lower:
            self.execute_command("move_backward")
        elif "turn left" in text_lower:
            self.execute_command("turn_left")
        elif "turn right" in text_lower:
            self.execute_command("turn_right")
        elif "stop" in text_lower or "halt" in text_lower:
            self.execute_command("stop")
        elif "help" in text_lower:
            self.execute_command("speak_help")
        else:
            # Use more sophisticated NLU for complex commands
            rospy.loginfo(f"Unknown simple command, passing to NLU: {text}")
            self.process_with_nlu(text)
    
    def execute_command(self, cmd):
        """Execute the robot command"""
        rospy.loginfo(f"Executing command: {cmd}")
        
        # Example: publish to cmd_vel topic to move the robot
        if cmd == "move_forward":
            twist = Twist()
            twist.linear.x = 0.5  # Move forward at 0.5 m/s
            # Publish to robot movement topic
        elif cmd == "move_backward":
            twist = Twist()
            twist.linear.x = -0.5  # Move backward
            # Publish to robot movement topic
        elif cmd == "turn_left":
            twist = Twist()
            twist.angular.z = 0.5  # Turn left
            # Publish to robot movement topic
        elif cmd == "turn_right":
            twist = Twist()
            twist.angular.z = -0.5  # Turn right
            # Publish to robot movement topic
        elif cmd == "stop":
            twist = Twist()
            # Zero velocities to stop
            # Publish to robot movement topic
        elif cmd == "speak_help":
            # Command to speak help information
            self.speak_text("I can move forward, backward, turn left, turn right, or stop.")
    
    def process_with_nlu(self, command_text):
        """Process command with Natural Language Understanding"""
        # This would implement more sophisticated command parsing
        # Using techniques like intent classification and entity extraction
        rospy.loginfo(f"Processing with NLU: {command_text}")

class NaturalLanguageUnderstanding:
    """
    Natural Language Understanding (NLU) component that processes 
    voice commands and translates them to robot actions
    """
    
    def __init__(self):
        # Define command patterns and mappings
        self.intent_patterns = {
            'navigation': [
                (r'go to (?:the )?(.+)', self.parse_navigation),
                (r'move to (?:the )?(.+)', self.parse_navigation),
                (r'navigate to (?:the )?(.+)', self.parse_navigation),
                (r'walk to (?:the )?(.+)', self.parse_navigation),
                (r'drive to (?:the )?(.+)', self.parse_navigation)
            ],
            'manipulation': [
                (r'pick up (?:the )?(.+)', self.parse_pickup),
                (r'grasp (?:the )?(.+)', self.parse_pickup),
                (r'take (?:the )?(.+)', self.parse_pickup),
                (r'get (?:the )?(.+)', self.parse_pickup),
                (r'place (?:the )?(.+) (?:on|at) (?:the )?(.+)', self.parse_placement)
            ],
            'question': [
                (r'where is (?:the )?(.+)', self.parse_where_is),
                (r'what is (?:the )?(.+)', self.parse_what_is),
                (r'can you tell me about (?:the )?(.+)', self.parse_what_is),
                (r'how many (.+) are (?:there|in the room)', self.parse_quantity)
            ]
        }
        
        # Known locations and objects in the environment
        self.known_locations = {
            'kitchen': (2.0, 3.0),
            'living room': (0.0, 0.0),
            'office': (-1.5, 2.5),
            'bedroom': (3.0, -1.0)
        }
        
        self.known_objects = [
            'apple', 'bottle', 'cup', 'book', 'chair', 'table', 'phone', 'keys'
        ]
        
    def parse_command(self, command_text):
        """Parse natural language command and extract intent and parameters"""
        command_text = command_text.lower().strip()
        
        for intent_type, patterns in self.intent_patterns.items():
            for pattern, parser_func in patterns:
                match = re.search(pattern, command_text)
                if match:
                    # Extract parameters using the parser function
                    params = parser_func(match.groups())
                    
                    return {
                        'intent_type': intent_type,
                        'action': match.group(0).split()[0],  # First word often indicates action
                        'parameters': params,
                        'confidence': 0.9  # High confidence for rule-based matching
                    }
        
        # If no patterns match, could use ML-based parsing or return None
        return self.ml_parse_command(command_text)
    
    def parse_navigation(self, matches):
        """Parse navigation commands like 'go to the kitchen'"""
        if len(matches) > 0:
            destination = matches[0].strip()
            # If destination is a known location, return its coordinates
            if destination in self.known_locations:
                return {
                    'location': destination,
                    'coordinates': self.known_locations[destination]
                }
            else:
                # Try to find closest match
                closest_match = self.find_closest_match(destination, list(self.known_locations.keys()))
                if closest_match:
                    return {
                        'location': closest_match,
                        'coordinates': self.known_locations[closest_match],
                        'original_request': destination
                    }
        
        return {'location': 'unknown', 'coordinates': None}
    
    def ml_parse_command(self, command_text):
        """Use machine learning model to parse command (placeholder implementation)"""
        # In practice, this would use a trained NLU model
        # For now, provide a basic implementation
        import openai
        
        try:
            # Use OpenAI API to parse the command
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are a natural language understanding system for a robot. 
                    Parse the user's command and return the intent and relevant parameters in JSON format. 
                    Intents include: navigation, manipulation, question, system. 
                    For navigation: include destination. 
                    For manipulation: include object and action. 
                    For questions: include subject and type of information requested."""},
                    {"role": "user", "content": f"Command: {command_text}\n\nRespond in JSON format with 'intent', 'action', and 'parameters'."}
                ],
                temperature=0.0
            )
            
            import json
            result = json.loads(response.choices[0].message['content'])
            return result
            
        except Exception as e:
            rospy.logwarn(f"ML NLU failed with error: {e}")
            return {
                'intent_type': 'unknown',
                'action': 'unknown',
                'parameters': {'raw_command': command_text},
                'confidence': 0.1
            }
    
    def find_closest_match(self, target, candidates):
        """Find closest match using string similarity"""
        import difflib
        closest = difflib.get_close_matches(target, candidates, n=1, cutoff=0.6)
        return closest[0] if closest else None

class VoiceCommandProcessor:
    """Main processor that integrates ASR and NLU components"""
    
    def __init__(self):
        # Initialize ASR system
        self.asr_system = RobustASRSystem()
        
        # Initialize NLU system
        self.nlu_system = NaturalLanguageUnderstanding()
        
        # ROS interfaces
        self.voice_cmd_sub = rospy.Subscriber('/voice_commands', String, self.voice_command_callback)
        self.robot_cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Robot state tracking
        self.robot_state = {
            'position': (0, 0),
            'orientation': 0,
            'held_object': None,
            'battery_level': 1.0
        }
        
        rospy.loginfo("Voice command processor initialized")
    
    def voice_command_callback(self, msg):
        """Process incoming voice commands"""
        command_text = msg.data
        rospy.loginfo(f"Processing voice command: {command_text}")
        
        # Parse the command
        parsed_command = self.nlu_system.parse_command(command_text)
        
        if parsed_command and parsed_command.get('confidence', 0) > 0.5:
            # Execute the command based on parsed intent
            success = self.execute_parsed_command(parsed_command)
            
            if success:
                rospy.loginfo(f"Command executed successfully: {command_text}")
                self.confirm_command_execution(command_text)
            else:
                rospy.logerr(f"Command execution failed: {command_text}")
                self.report_command_failure(command_text)
        else:
            rospy.logwarn(f"Could not parse command with sufficient confidence: {command_text}")
            self.request_clarification(command_text)
    
    def execute_parsed_command(self, parsed_cmd):
        """Execute command based on parsed intent"""
        intent_type = parsed_cmd.get('intent_type')
        
        try:
            if intent_type == 'navigation':
                return self.execute_navigation_command(parsed_cmd)
            elif intent_type == 'manipulation':
                return self.execute_manipulation_command(parsed_cmd)
            elif intent_type == 'question':
                return self.execute_question_response(parsed_cmd)
            elif intent_type == 'system':
                return self.execute_system_command(parsed_cmd)
            else:
                rospy.logwarn(f"Unknown intent type: {intent_type}")
                return False
                
        except Exception as e:
            rospy.logerr(f"Error executing command: {e}")
            return False
    
    def execute_navigation_command(self, parsed_cmd):
        """Execute navigation command"""
        params = parsed_cmd.get('parameters', {})
        destination = params.get('location')
        coords = params.get('coordinates')
        
        if coords:
            rospy.loginfo(f"Navigating to {destination} at coordinates {coords}")
            # Publish navigation goal to move_base or similar
            # This would typically use actionlib for navigation
            return True
        else:
            rospy.logerr(f"Could not determine coordinates for destination: {destination}")
            return False
    
    def execute_manipulation_command(self, parsed_cmd):
        """Execute manipulation command"""
        params = parsed_cmd.get('parameters', {})
        action = parsed_cmd.get('action', '').lower()
        obj = params.get('object', '')
        
        rospy.loginfo(f"Performing manipulation: {action} on {obj}")
        
        # This would interface with robot manipulation stack
        # For now, return True as a placeholder
        return True
    
    def execute_question_response(self, parsed_cmd):
        """Handle question and provide appropriate response"""
        params = parsed_cmd.get('parameters', {})
        subject = params.get('subject', '')
        
        # Generate response based on robot's state and knowledge
        response = self.generate_response_to_question(parsed_cmd)
        
        if response:
            self.speak_text(response)
            return True
        else:
            self.speak_text("I don't have information about that.")
            return False
    
    def generate_response_to_question(self, parsed_cmd):
        """Generate appropriate response to user's question"""
        # Based on the question parameters, generate a response
        # This would access robot's knowledge base and current sensor state
        return "I'm not sure, but I can try to find that for you."
    
    def confirm_command_execution(self, command_text):
        """Provide confirmation that command was executed"""
        confirmation = f"OK, I'm executing: {command_text}"
        self.speak_text(confirmation)
    
    def report_command_failure(self, command_text):
        """Report that command execution failed"""
        error_msg = f"Sorry, I couldn't execute: {command_text}"
        self.speak_text(error_msg)
    
    def request_clarification(self, command_text):
        """Request clarification for misunderstood command"""
        clarification_request = f"I didn't understand: {command_text}. Could you please repeat or clarify?"
        self.speak_text(clarification_request)
    
    def speak_text(self, text):
        """Speak text using text-to-speech"""
        # This would publish to a TTS system
        rospy.loginfo(f"Robot says: {text}")
        # Example: publish to TTS topic
        # tts_publisher.publish(text)

# Example usage
if __name__ == "__main__":
    rospy.init_node('voice_command_processor')
    
    processor = VoiceCommandProcessor()
    
    # Start the ASR system in a separate thread
    asr_thread = threading.Thread(target=processor.asr_system.run_voice_interface)
    asr_thread.daemon = True
    asr_thread.start()
    
    try:
        rospy.loginfo("Voice command processor running")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Voice command processor shutting down")
    finally:
        # Clean up audio resources
        if hasattr(processor.asr_system, 'stream'):
            processor.asr_system.stream.stop_stream()
            processor.asr_system.stream.close()
        
        if hasattr(processor.asr_system, 'audio'):
            processor.asr_system.audio.terminate()
```

## Multi-Modal Interaction

### Combining Voice with Other Modalities

```python
class MultimodalInteractionManager:
    """Manage interactions combining voice with vision, gestures, and touch"""
    
    def __init__(self):
        # ROS interfaces
        self.voice_sub = rospy.Subscriber('/voice_commands', String, self.voice_callback)
        self.vision_sub = rospy.Subscriber('/vision_annotations', AnnotationArray, self.vision_callback)
        self.touch_sub = rospy.Subscriber('/touch_sensors', TouchSensorArray, self.touch_callback)
        
        # Publishers
        self.response_pub = rospy.Publisher('/robot_speech', String, queue_size=10)
        
        # State management
        self.interaction_context = {
            'current_focus': None,
            'last_voice_command': '',
            'last_seen_objects': [],
            'robot_attention_mode': 'listening'  # listening, processing, executing, waiting
        }
        
        # Reference to other subsystems
        self.vision_system = None
        self.navigation_system = None
        self.manipulation_system = None
        
        rospy.loginfo("Multimodal interaction manager initialized")
    
    def voice_callback(self, msg):
        """Handle voice commands in context of other modalities"""
        command = msg.data
        
        # Update context
        self.interaction_context['last_voice_command'] = command
        self.interaction_context['robot_attention_mode'] = 'processing'
        
        # Process in context of recent visual observations
        recent_objects = self.interaction_context.get('last_seen_objects', [])
        
        # If command refers to "that" or "it", link to the most recently seen object
        if "that" in command.lower() or "it" in command.lower():
            if recent_objects:
                most_recent = recent_objects[-1]  # Most recently seen object
                command = self.disambiguate_reference(command, most_recent)
        
        # Parse and execute the command
        self.process_multimodal_command(command)
    
    def vision_callback(self, msg):
        """Update vision context when new objects are detected"""
        objects = [obj.label for obj in msg.annotations]
        self.interaction_context['last_seen_objects'] = objects[-5:]  # Keep last 5 objects
        
        # If robot is waiting for visual confirmation of a previous command, check now
        if self.interaction_context.get('waiting_for_confirmation'):
            self.process_waiting_confirmation()
    
    def disambiguate_reference(self, command, referent_object):
        """Replace ambiguous references like 'that' with specific objects"""
        command = command.replace("that", f"the {referent_object['label']}")
        command = command.replace("it", f"the {referent_object['label']}")
        return command
    
    def process_multimodal_command(self, command):
        """Process command considering context from multiple modalities"""
        # Parse the command with NLU considering context
        parsed = self.parse_with_context(command, self.interaction_context)
        
        if parsed:
            # Execute the parsed command
            success = self.execute_command_with_context(parsed, self.interaction_context)
            
            if success:
                self.interaction_context['robot_attention_mode'] = 'executing'
                self.confirm_execution(command)
            else:
                self.interaction_context['robot_attention_mode'] = 'listening'
                self.request_clarification(command)
        else:
            self.interaction_context['robot_attention_mode'] = 'listening'
            self.request_clarification(command)
    
    def parse_with_context(self, command, context):
        """Parse command considering multimodal context"""
        # Create context-aware prompt for LLM
        context_description = self.describe_current_context(context)
        
        prompt = f"""
        Context: {context_description}
        
        Command: {command}
        
        Parse this command considering the visual and temporal context. 
        Identify the intention and required parameters.
        
        Respond in JSON format with:
        {{
            "intent": "...",
            "action": "...", 
            "parameters": {{...}},
            "confidence": 0.0-1.0
        }}
        """
        
        try:
            # Use OpenAI or similar for context-aware parsing
            import openai
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a multimodal command interpreter for a robot. Parse commands considering visual context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            import json
            return json.loads(response.choices[0].message['content'])
        except Exception as e:
            rospy.logerr(f"Multimodal parsing failed: {e}")
            return None
    
    def describe_current_context(self, context):
        """Describe the current interaction context"""
        description = f"Current state:\n"
        description += f"- Robot mode: {context['robot_attention_mode']}\n"
        
        if context.get('last_seen_objects'):
            description += f"- Recently seen objects: {', '.join(context['last_seen_objects'][-3:])}\n"
        
        if context.get('current_focus'):
            description += f"- Current focus: {context['current_focus']}\n"
        
        return description
    
    def execute_command_with_context(self, parsed_command, context):
        """Execute command considering multimodal context"""
        intent = parsed_command.get('intent', 'unknown')
        
        if intent == 'navigate':
            return self.execute_navigation_with_vision_context(parsed_command, context)
        elif intent == 'manipulate':
            return self.execute_manipulation_with_vision_context(parsed_command, context)
        elif intent == 'greet':
            return self.execute_greeting_with_presence_context(parsed_command, context)
        else:
            # Default to basic command execution
            return self.execute_basic_command(parsed_command)

class AdaptiveVoiceInterface:
    """Voice interface that adapts to user and environment"""
    
    def __init__(self):
        # Adaptation parameters
        self.user_profiles = {}  # Store user-specific settings and preferences
        self.environment_context = {}  # Store environment-specific settings
        self.context_history = []  # Conversation history for context
        
        # ASR model and parameters
        self.asr_model = "base"  # Could be changed based on requirements
        self.asr_confidence_threshold = 0.7
        
        # Wake word detection (optional)
        self.wake_word = "robot"
        self.wake_word_enabled = True
        self.listening_led_publisher = rospy.Publisher('/listening_led', Bool, queue_size=10)
        
        rospy.loginfo("Adaptive voice interface initialized")
    
    def process_speech_with_adaptation(self, audio_data):
        """Process speech with user and environment adaptation"""
        # Determine current user (if available)
        current_user = self.identify_user_from_voice(audio_data)
        
        # Adjust ASR parameters based on user profile
        if current_user in self.user_profiles:
            user_settings = self.user_profiles[current_user]
            confidence_thresh = user_settings.get('confidence_threshold', self.asr_confidence_threshold)
        else:
            confidence_thresh = self.asr_confidence_threshold
        
        # Transcribe with appropriate settings
        text, confidence = self.transcribe_with_settings(audio_data, confidence_thresh)
        
        if confidence > confidence_thresh:
            # Process command in context
            self.context_history.append({
                'user': current_user,
                'command': text,
                'timestamp': rospy.Time.now()
            })
            
            # Limit history to last 10 interactions
            if len(self.context_history) > 10:
                self.context_history = self.context_history[-10:]
            
            return text
        else:
            rospy.logwarn(f"Low confidence transcription ({confidence:.3f}): {text}")
            return None
    
    def identify_user_from_voice(self, audio_data):
        """Identify user from voice characteristics (simplified implementation)"""
        # In a real system, this would use speaker recognition
        # For now, return a default user
        return "unknown_user"
    
    def transcribe_with_settings(self, audio_data, confidence_threshold):
        """Transcribe audio with specific settings"""
        # This would implement the actual transcription with the specified confidence threshold
        # For now, use a placeholder implementation
        # In practice, would call Whisper or similar with the specified parameters
        rospy.loginfo("Transcribing with adaptive settings")
        return self.asr_system.transcribe_audio(audio_data)
    
    def adapt_to_environment(self, audio_characteristics):
        """Adapt voice recognition to environment (noise levels, etc.)"""
        # Analyze environment characteristics
        noise_level = audio_characteristics.get('noise_level', 0.0)
        reverb_amount = audio_characteristics.get('reverb', 0.0)
        
        # Adjust ASR parameters based on environment
        if noise_level > 0.5:  # High noise environment
            # Use more robust ASR settings
            self.asr_confidence_threshold = 0.6  # Lower threshold for noisy environments
        else:
            # Normal environment
            self.asr_confidence_threshold = 0.7
        
        if reverb_amount > 0.3:  # High reverb environment
            # Apply de-reverberation preprocessing if available
            rospy.loginfo("High reverb detected, applying preprocessing")
    
    def learn_user_preferences(self, user_id, command_success_feedback):
        """Learn from user interactions to improve future interactions"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'command_history': [],
                'preferred_phrases': [],
                'accent_modifications': [],
                'success_rate': 0.0,
                'confidence_threshold': self.asr_confidence_threshold
            }
        
        # Update user profile based on success/failure
        profile = self.user_profiles[user_id]
        profile['command_history'].append(command_success_feedback)
        
        # Update success rate
        successes = sum(1 for item in profile['command_history'] if item.get('success', False))
        profile['success_rate'] = successes / len(profile['command_history'])
        
        # Adjust confidence threshold based on success rate
        if profile['success_rate'] < 0.7:  # Low success rate
            profile['confidence_threshold'] -= 0.05  # Lower threshold
        elif profile['success_rate'] > 0.9:  # High success rate
            profile['confidence_threshold'] += 0.02  # Raise threshold slightly
        
        # Keep only recent history
        if len(profile['command_history']) > 50:
            profile['command_history'] = profile['command_history'][-50:]

# Example of a complete voice interaction system
class CompleteVoiceInteractionSystem:
    """Complete voice interaction system for robotics"""
    
    def __init__(self):
        # Initialize all components
        self.asr_system = RobustASRSystem()
        self.nlu_system = NaturalLanguageUnderstanding()
        self.voice_processor = VoiceCommandProcessor()
        self.multimodal_manager = MultimodalInteractionManager()
        self.adaptive_interface = AdaptiveVoiceInterface()
        
        # Service for manual wake-up
        self.wake_service = rospy.Service('/wake_robot', Trigger, self.wake_robot_callback)
        
        # Service for system status
        self.status_service = rospy.Service('/voice_system_status', GetStatus, self.get_status_callback)
        
        rospy.loginfo("Complete voice interaction system initialized")
    
    def wake_robot_callback(self, req):
        """Wake up the robot for voice commands"""
        # Change robot state to attentive
        self.multimodal_manager.interaction_context['robot_attention_mode'] = 'listening'
        
        # Visual feedback
        led_msg = Bool()
        led_msg.data = True
        self.multimodal_manager.listening_led_publisher.publish(led_msg)
        
        res = TriggerResponse()
        res.success = True
        res.message = "Robot is now listening"
        return res
    
    def get_status_callback(self, req):
        """Get voice system status"""
        status = self.get_system_status()
        return GetStatusResponse(status=status)
    
    def get_system_status(self):
        """Get current status of voice system"""
        status = f"ASR: Active, NLU: Ready, Mode: {self.multimodal_manager.interaction_context['robot_attention_mode']}"
        return status

if __name__ == "__main__":
    rospy.init_node('complete_voice_interaction_system')
    
    try:
        voice_system = CompleteVoiceInteractionSystem()
        rospy.loginfo("Voice interaction system running")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Voice interaction system shutting down")
    except Exception as e:
        rospy.logerr(f"Error in voice interaction system: {e}")
        raise
```

## Voice Command Processing Pipelines

### Speech-to-Intent Pipeline

```python
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple

class SpeechToIntentPipeline:
    """Complete pipeline from speech to robot intent execution"""
    
    def __init__(self):
        self.asr_component = SelfSupervisedASR()
        self.nlp_component = ContextualNLPUtility()
        self.dialogue_manager = DialogueStateManager()
        self.action_mapper = IntentToActionMapper()
        
        # Performance monitoring
        self.metrics_collector = MetricsCollector()
        
        # Error handling and recovery
        self.error_handler = ErrorHandler()
        
    async def process_voice_command(self, audio_input: bytes) -> Dict:
        """Complete pipeline: audio -> text -> intent -> action"""
        start_time = time.time()
        
        try:
            # Step 1: Speech Recognition
            transcription, confidence = await self.asr_component.transcribe(audio_input)
            
            if confidence < 0.6:  # Low confidence
                return {
                    'success': False,
                    'reason': 'low_asr_confidence',
                    'confidence': confidence,
                    'suggestions': ['please_repeat', 'speak_clearly']
                }
            
            # Step 2: Natural Language Processing
            intent_result = await self.nlp_component.parse_intent(
                transcription, 
                self.dialogue_manager.get_context()
            )
            
            if not intent_result.get('parsed_intent'):
                return {
                    'success': False,
                    'reason': 'nlp_failure',
                    'transcription': transcription,
                    'suggestions': ['use_different_phrasing', 'speak_more_specifically']
                }
            
            # Step 3: Update dialogue state
            self.dialogue_manager.update_state(transcription, intent_result)
            
            # Step 4: Map intent to action
            action_plan = await self.action_mapper.map_intent_to_action(intent_result)
            
            if not action_plan:
                return {
                    'success': False,
                    'reason': 'no_matching_action',
                    'intent': intent_result.get('parsed_intent'),
                    'suggestions': ['request_help', 'check_capabilities']
                }
            
            # Step 5: Execute action
            execution_result = await self.execute_action_plan(action_plan)
            
            # Step 6: Collect metrics
            processing_time = time.time() - start_time
            self.metrics_collector.record_interaction(
                transcription=transcription,
                intent=intent_result,
                action_plan=action_plan,
                execution_result=execution_result,
                processing_time=processing_time
            )
            
            return {
                'success': execution_result.get('success', False),
                'transcription': transcription,
                'intent': intent_result,
                'action_plan': action_plan,
                'execution_result': execution_result,
                'processing_time': processing_time
            }
            
        except Exception as e:
            # Error handling
            error_response = self.error_handler.handle_error(e, audio_input)
            
            processing_time = time.time() - start_time
            self.metrics_collector.record_error(
                error=e,
                processing_time=processing_time
            )
            
            return {
                'success': False,
                'reason': 'pipeline_error',
                'error': str(e),
                'processing_time': processing_time,
                'suggestions': error_response.get('suggestions', [])
            }
    
    async def execute_action_plan(self, action_plan: Dict) -> Dict:
        """Execute the planned actions"""
        results = []
        
        for action in action_plan.get('actions', []):
            try:
                # Determine action type and execute accordingly
                if action['type'] == 'navigation':
                    result = await self.execute_navigation_action(action)
                elif action['type'] == 'manipulation':
                    result = await self.execute_manipulation_action(action)
                elif action['type'] == 'information':
                    result = await self.execute_information_action(action)
                elif action['type'] == 'system':
                    result = await self.execute_system_action(action)
                else:
                    result = {'success': False, 'error': f'Unknown action type: {action["type"]}'}
                
                results.append(result)
                
                # If any action fails, consider the plan failed (unless explicitly recoverable)
                if not result.get('success') and not action.get('recoverable', False):
                    break
                    
            except Exception as e:
                results.append({'success': False, 'error': str(e)})
                break
        
        return {
            'success': all(r.get('success', False) for r in results),
            'individual_results': results,
            'plan_executed': len(results)
        }
    
    async def execute_navigation_action(self, action: Dict) -> Dict:
        """Execute navigation-related actions"""
        # This would interface with the actual navigation system
        try:
            target_location = action['parameters'].get('location')
            
            # In a real system, this would call navigation services
            nav_result = await self.call_navigation_service(target_location)
            
            return {
                'success': nav_result.get('success', False),
                'target_location': target_location,
                'actual_path_taken': nav_result.get('path', []),
                'time_taken': nav_result.get('time', 0)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def execute_manipulation_action(self, action: Dict) -> Dict:
        """Execute manipulation-related actions"""
        try:
            object_name = action['parameters'].get('object')
            manipulation_type = action['parameters'].get('manipulation_type')  # 'grasp', 'place', etc.
            
            # In a real system, this would call manipulation services
            manipulation_result = await self.call_manipulation_service(
                object_name, 
                manipulation_type
            )
            
            return {
                'success': manipulation_result.get('success', False),
                'object': object_name,
                'manipulation_type': manipulation_type,
                'details': manipulation_result.get('details', {})
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

class MetricsCollector:
    """Collect and analyze performance metrics for the voice pipeline"""
    
    def __init__(self):
        self.interactions = []
        self.errors = []
        self.performance_metrics = {
            'avg_processing_time': 0.0,
            'success_rate': 0.0,
            'avg_asr_confidence': 0.0,
            'most_common_intents': {},
            'peak_usage_times': [],
            'user_satisfaction': 0.0
        }
    
    def record_interaction(self, transcription: str, intent: Dict, 
                          action_plan: Dict, execution_result: Dict, 
                          processing_time: float):
        """Record a successful interaction"""
        interaction_record = {
            'timestamp': time.time(),
            'transcription': transcription,
            'intent': intent,
            'action_plan': action_plan,
            'execution_result': execution_result,
            'processing_time': processing_time,
            'success': execution_result.get('success', False)
        }
        
        self.interactions.append(interaction_record)
        
        # Update performance metrics
        self._update_performance_metrics(interaction_record)
    
    def _update_performance_metrics(self, record: Dict):
        """Update performance metrics based on recorded interaction"""
        # Update average processing time
        times = [r['processing_time'] for r in self.interactions]
        self.performance_metrics['avg_processing_time'] = sum(times) / len(times) if times else 0.0
        
        # Update success rate
        successful = [r for r in self.interactions if r['success']]
        self.performance_metrics['success_rate'] = len(successful) / len(self.interactions) if self.interactions else 0.0
        
        # Update most common intents
        intent_counts = {}
        for r in self.interactions:
            intent_type = r['intent'].get('intent_type', 'unknown')
            intent_counts[intent_type] = intent_counts.get(intent_type, 0) + 1
        
        self.performance_metrics['most_common_intents'] = intent_counts

# Error handling and recovery
class ErrorHandler:
    """Handle errors in the voice command pipeline"""
    
    def __init__(self):
        self.recovery_strategies = {
            'low_asr_confidence': self._handle_low_confidence,
            'nlp_failure': self._handle_nlp_failure,
            'no_matching_action': self._handle_no_action,
            'execution_failure': self._handle_execution_failure,
            'pipeline_error': self._handle_pipeline_error
        }
    
    def handle_error(self, error: Exception, audio_input: bytes) -> Dict:
        """Handle different types of errors with appropriate recovery"""
        error_type = self._categorize_error(error)
        
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type](error, audio_input)
        else:
            return self._handle_unknown_error(error, audio_input)
    
    def _categorize_error(self, error: Exception) -> str:
        """Categorize error for appropriate handling"""
        error_str = str(error).lower()
        
        if 'confidence' in error_str or 'threshold' in error_str:
            return 'low_asr_confidence'
        elif 'nlp' in error_str or 'parsing' in error_str:
            return 'nlp_failure'
        elif 'action' in error_str or 'mapping' in error_str:
            return 'no_matching_action'
        elif 'execution' in error_str or 'service' in error_str:
            return 'execution_failure'
        else:
            return 'pipeline_error'
    
    def _handle_low_confidence(self, error: Exception, audio_input: bytes) -> Dict:
        """Handle low ASR confidence errors"""
        return {
            'error_type': 'low_confidence',
            'suggestions': [
                'please_repeat_command',
                'speak_more_clearly',
                'move_closer_to_robot'
            ],
            'alternative_inputs': ['text_input', 'gesture_input']
        }
    
    def _handle_nlp_failure(self, error: Exception, audio_input: bytes) -> Dict:
        """Handle NLP parsing failures"""
        return {
            'error_type': 'nlp_failure',
            'suggestions': [
                'use_simpler_language',
                'be_more_specific',
                'use_robot_capabilities_as_reference'
            ]
        }
    
    def _handle_unknown_error(self, error: Exception, audio_input: bytes) -> Dict:
        """Handle unrecognized errors"""
        return {
            'error_type': 'unknown',
            'suggestions': [
                'try_again_later',
                'contact_support',
                'use_alternative_input_method'
            ]
        }
```

## Practical Implementation: Voice-Controlled Robot

Here's a complete example of a voice-controlled robot system:

```python
#!/usr/bin/env python3

import rospy
import pyaudio
import struct
import wave
import numpy as np
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import whisper
import openai
import json
import re
import time

class VoiceControlledRobot:
    """Complete implementation of a voice-controlled robot"""
    
    def __init__(self):
        rospy.init_node('voice_controlled_robot')
        
        # Audio parameters
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.silence_threshold = 0.01
        self.min_speech_duration = 0.5  # seconds
        self.max_recording_duration = 5.0  # seconds
        
        # Initialize Whisper model
        rospy.loginfo("Loading Whisper model...")
        self.whisper_model = whisper.load_model("small.en")  # Use smaller model for faster processing
        
        # ROS publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.voice_cmd_pub = rospy.Publisher('/voice_command', String, queue_size=10)
        self.voice_feedback_pub = rospy.Publisher('/voice_feedback', String, queue_size=10)
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # Robot state
        self.robot_state = {
            'x': 0.0,
            'y': 0.0,
            'theta': 0.0,  # Heading angle
            'linear_vel': 0.0,
            'angular_vel': 0.0,
            'battery': 1.0,
            'gripper': 'open'  # or 'closed'
        }
        
        # Command mappings
        self.command_mappings = {
            # Movement commands
            r'.*\b(move|go)\s+(forward|ahead|straight)\b.*': self.move_forward,
            r'.*\b(move|go)\s+(back|backward|reverse)\b.*': self.move_backward,
            r'.*\b(turn|rotate)\s+(left|anti-clockwise)\b.*': self.turn_left,
            r'.*\b(turn|rotate)\s+(right|clockwise)\b.*': self.turn_right,
            r'.*\b(spin|rotate)\s+(left|anti-clockwise)\b.*': self.spin_left,
            r'.*\b(spin|rotate)\s+(right|clockwise)\b.*': self.spin_right,
            r'.*\b(stop|halt|freeze)\b.*': self.stop_robot,
            r'.*\b(speed|faster)\b.*': self.increase_speed,
            r'.*\b(slow|slower|crawl)\b.*': self.decrease_speed,
            
            # Gripper commands
            r'.*\b(open|release|let go)\b.*': self.open_gripper,
            r'.*\b(close|grab|hold|tighten)\b.*': self.close_gripper,
            r'.*\b(grip|gripper)\s+(strong|tight)\b.*': self.tighten_grip,
            r'.*\b(grip|gripper)\s+(loose|soft)\b.*': self.loosen_grip,
            
            # Navigation commands
            r'.*\b(navigate|go to|move to)\s+(?:the\s+)?(kitchen|living room|bedroom|office|lab|hallway)\b.*': self.navigate_to_location,
            
            # Inquiry commands
            r'.*\b(how are you|status|report|state)\b.*': self.report_status,
            r'.*\b(battery|power|energy)\b.*': self.report_battery,
            r'.*\b(who are you|what are you|describe yourself)\b.*': self.describe_self,
        }
        
        # Robot speed settings
        self.current_linear_speed = 0.3  # m/s
        self.current_angular_speed = 0.5  # rad/s
        self.max_linear_speed = 1.0
        self.min_linear_speed = 0.1
        self.max_angular_speed = 1.0
        self.min_angular_speed = 0.2
        
        # Control flags
        self.listening_enabled = True
        self.recording = False
        self.speech_detected = False
        
        rospy.loginfo("Voice controlled robot initialized")
    
    def start_voice_control(self):
        """Start the main voice control loop"""
        rospy.loginfo("Starting voice control system...")
        
        try:
            while not rospy.is_shutdown() and self.listening_enabled:
                # Listen for speech
                rospy.loginfo("Listening for voice command...")
                audio_data = self.listen_for_speech()
                
                if audio_data is not None:
                    # Transcribe audio
                    transcription = self.transcribe_audio(audio_data)
                    
                    if transcription:
                        rospy.loginfo(f"Heard: {transcription}")
                        
                        # Publish transcription
                        trans_msg = String()
                        trans_msg.data = transcription
                        self.voice_cmd_pub.publish(trans_msg)
                        
                        # Process voice command
                        success = self.process_voice_command(transcription)
                        
                        if success:
                            feedback = f"Executed: {transcription}"
                        else:
                            feedback = f"Failed to execute: {transcription}"
                        
                        feedback_msg = String()
                        feedback_msg.data = feedback
                        self.voice_feedback_pub.publish(feedback_msg)
                    else:
                        rospy.logwarn("Could not transcribe audio")
                else:
                    rospy.loginfo("No speech detected, continuing to listen")
                
                time.sleep(0.5)  # Brief pause between listening attempts
                
        except KeyboardInterrupt:
            rospy.loginfo("Voice control interrupted by user")
        finally:
            self.cleanup()
    
    def listen_for_speech(self):
        """Listen for and capture a speech segment"""
        rospy.loginfo("Listening for speech...")
        
        # Buffer to store audio chunks
        audio_buffer = []
        silence_count = 0
        speech_started = False
        max_silence_chunks = int(0.3 * self.sample_rate / self.chunk_size)  # 0.3s silence threshold
        max_total_chunks = int(self.max_recording_duration * self.sample_rate / self.chunk_size)
        
        chunk_count = 0
        
        while chunk_count < max_total_chunks and not rospy.is_shutdown():
            # Read audio chunk
            chunk_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio_chunk = np.frombuffer(chunk_data, dtype=np.float32)
            
            # Calculate energy to detect speech
            energy = np.mean(np.abs(audio_chunk) ** 2)
            
            if energy > self.silence_threshold:
                # Speech detected
                audio_buffer.extend(audio_chunk)
                speech_started = True
                silence_count = 0
            elif speech_started:
                # We're past the initial silence, continue recording for a bit more
                audio_buffer.extend(audio_chunk)
                silence_count += 1
                
                # If sufficient silence is detected after speech, stop recording
                if silence_count >= max_silence_chunks:
                    rospy.loginfo("End of speech detected")
                    break
            # If we haven't started speech yet, just continue waiting
            
            chunk_count += 1
        
        if len(audio_buffer) > 0:
            rospy.loginfo(f"Captured audio of length: {len(audio_buffer) / self.sample_rate:.3f}s")
            return np.array(audio_buffer, dtype=np.float32)
        else:
            rospy.loginfo("No speech captured")
            return None
    
    def transcribe_audio(self, audio_array):
        """Transcribe audio using Whisper"""
        try:
            # Transcribe the audio
            result = self.whisper_model.transcribe(audio_array, language='en')
            transcription = result["text"].strip()
            
            rospy.loginfo(f"Transcription: {transcription}")
            return transcription
            
        except Exception as e:
            rospy.logerr(f"Error in transcription: {e}")
            return ""
    
    def process_voice_command(self, command_text):
        """Process the transcribed voice command"""
        if not command_text:
            return False
        
        command_text_lower = command_text.lower()
        
        # Match command patterns
        for pattern, command_func in self.command_mappings.items():
            match = re.match(pattern, command_text_lower)
            if match:
                try:
                    # Extract matched groups for parameterized commands
                    groups = match.groups() if match.groups() else ()
                    success = command_func(*groups)
                    return success
                except Exception as e:
                    rospy.logerr(f"Error executing command {command_func.__name__}: {e}")
                    return False
        
        # If no pattern matched, try to use NLU to interpret the command
        success = self.process_with_nlu(command_text_lower)
        return success
    
    def process_with_nlu(self, command_text):
        """Use NLU to interpret more complex commands"""
        # For complex commands, we can use an LLM to interpret
        # This is a simplified version - in practice, you'd use a more sophisticated NLP system
        
        # Example: Look for keywords that indicate different types of commands
        if any(word in command_text for word in ['move', 'go', 'navigate', 'drive', 'turn', 'rotate']):
            return self.handle_navigation_command(command_text)
        elif any(word in command_text for word in ['pick', 'grasp', 'take', 'get', 'catch', 'hold', 'drop', 'release', 'put']):
            return self.handle_manipulation_command(command_text)
        elif any(word in command_text for word in ['where', 'what', 'how', 'tell me', 'describe', 'status', 'report']):
            return self.handle_inquiry_command(command_text)
        else:
            rospy.logwarn(f"Unknown command type: {command_text}")
            self.speak_response(f"I'm not sure how to handle: {command_text}")
            return False
    
    def handle_navigation_command(self, command_text):
        """Handle navigation-related commands"""
        rospy.loginfo(f"Processing navigation command: {command_text}")
        
        # Example of handling navigation commands with more complex parsing
        if 'kitchen' in command_text:
            return self.navigate_to_location('kitchen')
        elif 'living room' in command_text or 'livingroom' in command_text:
            return self.navigate_to_location('living room')
        elif 'bedroom' in command_text:
            return self.navigate_to_location('bedroom')
        else:
            self.speak_response("I'm not sure where you want me to go")
            return False
    
    def handle_manipulation_command(self, command_text):
        """Handle manipulation-related commands"""
        rospy.loginfo(f"Processing manipulation command: {command_text}")
        
        # This would typically interface with a manipulation stack
        # For now, we'll just provide feedback
        self.speak_response("I can't perform that manipulation task yet, but I'm learning!")
        return False  # For now, return false as manipulation is not implemented
    
    def handle_inquiry_command(self, command_text):
        """Handle inquiry-related commands"""
        rospy.loginfo(f"Processing inquiry command: {command_text}")
        
        if 'status' in command_text or 'how are you' in command_text:
            return self.report_status()
        elif 'battery' in command_text or 'power' in command_text:
            return self.report_battery()
        elif 'who are you' in command_text or 'what are you' in command_text:
            return self.describe_self()
        else:
            self.speak_response("I can't answer that right now, but I'm constantly learning!")
            return False
    
    # Movement commands
    def move_forward(self):
        """Move robot forward"""
        cmd = Twist()
        cmd.linear.x = self.current_linear_speed
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        self.robot_state['linear_vel'] = self.current_linear_speed
        rospy.loginfo(f"Moving forward at {self.current_linear_speed} m/s")
        return True
    
    def move_backward(self):
        """Move robot backward"""
        cmd = Twist()
        cmd.linear.x = -self.current_linear_speed
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        self.robot_state['linear_vel'] = -self.current_linear_speed
        rospy.loginfo(f"Moving backward at {self.current_linear_speed} m/s")
        return True
    
    def turn_left(self):
        """Turn robot left in place"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = self.current_angular_speed
        self.cmd_vel_pub.publish(cmd)
        self.robot_state['angular_vel'] = self.current_angular_speed
        rospy.loginfo(f"Turning left at {self.current_angular_speed} rad/s")
        return True
    
    def turn_right(self):
        """Turn robot right in place"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = -self.current_angular_speed
        self.cmd_vel_pub.publish(cmd)
        self.robot_state['angular_vel'] = -self.current_angular_speed
        rospy.loginfo(f"Turning right at {self.current_angular_speed} rad/s")
        return True
    
    def spin_left(self):
        """Spin robot left"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = self.current_angular_speed * 2  # Spin faster
        self.cmd_vel_pub.publish(cmd)
        self.robot_state['angular_vel'] = self.current_angular_speed * 2
        rospy.loginfo(f"Spinning left at {self.current_angular_speed * 2} rad/s")
        return True
    
    def spin_right(self):
        """Spin robot right"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = -self.current_angular_speed * 2  # Spin faster
        self.cmd_vel_pub.publish(cmd)
        self.robot_state['angular_vel'] = -self.current_angular_speed * 2
        rospy.loginfo(f"Spinning right at {self.current_angular_speed * 2} rad/s")
        return True
    
    def stop_robot(self):
        """Stop robot movement"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        self.robot_state['linear_vel'] = 0.0
        self.robot_state['angular_vel'] = 0.0
        rospy.loginfo("Stopping robot")
        return True
    
    def increase_speed(self):
        """Increase robot movement speed"""
        self.current_linear_speed = min(self.max_linear_speed, self.current_linear_speed * 1.2)
        self.current_angular_speed = min(self.max_angular_speed, self.current_angular_speed * 1.2)
        self.speak_response(f"Speed increased. Linear: {self.current_linear_speed:.1f}, Angular: {self.current_angular_speed:.1f}")
        return True
    
    def decrease_speed(self):
        """Decrease robot movement speed"""
        self.current_linear_speed = max(self.min_linear_speed, self.current_linear_speed / 1.2)
        self.current_angular_speed = max(self.min_angular_speed, self.current_angular_speed / 1.2)
        self.speak_response(f"Speed decreased. Linear: {self.current_linear_speed:.1f}, Angular: {self.current_angular_speed:.1f}")
        return True
    
    # Gripper commands
    def open_gripper(self):
        """Open robot gripper"""
        self.robot_state['gripper'] = 'open'
        self.speak_response("Gripper opened")
        return True
    
    def close_gripper(self):
        """Close robot gripper"""
        self.robot_state['gripper'] = 'closed'
        self.speak_response("Gripper closed")
        return True
    
    def tighten_grip(self):
        """Tighten robot grip"""
        self.speak_response("Grip tightened")
        return True
    
    def loosen_grip(self):
        """Loosen robot grip"""
        self.speak_response("Grip loosened")
        return True
    
    # Navigation commands
    def navigate_to_location(self, location):
        """Navigate to a known location"""
        # In a real implementation, this would use the navigation stack
        # For now, just acknowledge the command
        self.speak_response(f"Going to {location}")
        return True
    
    # Inquiry commands
    def report_status(self):
        """Report robot status"""
        status = f"I am a voice-controlled educational robot. Current linear speed is {self.current_linear_speed:.1f} m/s and angular speed is {self.current_angular_speed:.1f} rad/s."
        self.speak_response(status)
        return True
    
    def report_battery(self):
        """Report battery status"""
        battery_status = f"My battery level is {self.robot_state['battery']*100:.1f}%."
        self.speak_response(battery_status)
        return True
    
    def describe_self(self):
        """Describe the robot's capabilities"""
        description = "I am an educational robot designed to understand voice commands and navigate environments. I can move in different directions, adjust my speed, and respond to simple queries."
        self.speak_response(description)
        return True
    
    def speak_response(self, text):
        """Function to provide verbal response (would interface with TTS)"""
        rospy.loginfo(f"Robot says: {text}")
        # In practice, you'd use a TTS system to speak the response
        # For example, calling a TTS ROS service or publishing to a TTS topic
    
    def cleanup(self):
        """Clean up resources"""
        rospy.loginfo("Cleaning up voice control resources...")
        
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()

        rospy.loginfo("Voice control resources cleaned up")

if __name__ == '__main__':
    robot = VoiceControlledRobot()
    
    try:
        robot.start_voice_control()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutdown requested")
    finally:
        robot.cleanup()
```

## Exercises

1. **Voice Command Implementation**: Implement a complete voice command system that integrates Whisper ASR with a simple mobile robot, allowing voice commands to control robot movement.

2. **Natural Language Understanding Exercise**: Create a natural language understanding module that can interpret complex robot commands that involve multiple steps or conditional actions.

3. **Multimodal Interaction Exercise**: Extend the voice command system to incorporate visual feedback, so the robot can identify objects mentioned in voice commands and act accordingly.

## Summary

This chapter covered voice command processing systems for robotics, focusing on Automatic Speech Recognition (ASR), Natural Language Understanding (NLU), and multimodal interaction design. We explored how to build voice interfaces that allow natural human-robot interaction, how to process natural language commands into structured robot actions, and how to combine voice with other modalities for richer interaction.

The key takeaways include:
- Voice interfaces enable natural, hands-free robot interaction
- ASR systems must be optimized for real-time performance and noisy environments
- NLU systems bridge natural language to structured robot commands
- Multimodal systems combine voice with vision and other modalities for better context
- Error handling and adaptation improve user experience

## Cross-references

For foundational concepts about Physical AI, see [Chapter 1: Introduction to Physical AI](../part-01-foundations/chapter-1). For related perception systems, see [Chapter 16: Computer Vision Integration](../part-05-vla/chapter-16). For cognitive planning aspects, see [Chapter 15: Cognitive Planning with LLMs](../part-05-vla/chapter-15).