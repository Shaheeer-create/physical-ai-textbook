---
title: "Chapter 15: Cognitive Planning with LLMs"
description: "Implementing LLM-based task decomposition and natural language processing for robotics"
sidebar_position: 2
---

# Chapter 15: Cognitive Planning with LLMs

## Learning Objectives

After completing this chapter, you should be able to:

- Decompose complex tasks into action sequences using LLMs
- Translate natural language to action sequences for robots
- Implement planning algorithms enhanced with LLMs
- Design failure recovery and replanning systems with LLM assistance

## Introduction to LLM-Based Cognitive Planning

### The Role of LLMs in Cognitive Robotics

Large Language Models (LLMs) play a transformative role in cognitive robotics by providing:

1. **Natural Language Interpretation**: Converting human instructions into machine-understandable commands
2. **Task Decomposition**: Breaking complex tasks into primitive actions
3. **Reasoning Under Uncertainty**: Making decisions when information is incomplete
4. **Contextual Understanding**: Incorporating environmental context into planning
5. **Adaptive Planning**: Modifying plans based on changing conditions

### LLM Integration Architecture

The cognitive planning system integrates LLMs with robot systems through several key components:

1. **Natural Language Interface**: Converts human commands to structured representations
2. **World Model Integration**: Incorporates environmental state into LLM reasoning
3. **Action Mapping**: Translates LLM outputs to robot executable commands
4. **Feedback Loop**: Updates LLM with execution results

## Task Decomposition with LLMs

### Hierarchical Task Networks (HTN) with LLMs

Traditional HTN systems decompose high-level tasks into primitive actions. With LLMs, we can make this process more flexible and adaptable:

```python
import openai
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Action:
    """Represents a primitive action for the robot"""
    name: str
    parameters: Dict[str, Any]
    preconditions: List[str]  # Conditions that must be true before executing
    effects: List[str]        # Facts that become true after execution
    
    def to_json(self):
        return {
            "name": self.name,
            "parameters": self.parameters,
            "preconditions": self.preconditions,
            "effects": self.effects
        }

@dataclass
class Task:
    """Represents a high-level task that can be decomposed"""
    name: str
    description: str
    subtasks: List['Task'] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    parent: Optional['Task'] = None
    
    def add_subtask(self, subtask: 'Task'):
        subtask.parent = self
        self.subtasks.append(subtask)
    
    def to_json(self):
        return {
            "name": self.name,
            "description": self.description,
            "subtasks": [subtask.to_json() for subtask in self.subtasks],
            "actions": [action.to_json() for action in self.actions],
            "status": self.status.value
        }

class LLMTaskDecomposer:
    """Uses LLMs to decompose complex tasks into action sequences"""
    
    def __init__(self, model_name: str = "gpt-4", api_key: str = None):
        if api_key:
            openai.api_key = api_key
        self.model_name = model_name
        self.task_history = []
        
        # Robot capabilities and action vocabulary
        self.action_vocabulary = {
            "navigation": [
                "move_to(location)",
                "navigate_to(x, y, z)",
                "follow_path(path)",
                "reach_waypoint(location)"
            ],
            "manipulation": [
                "grasp(object)",
                "release(object)",
                "pickup(item)",
                "place(item, location)",
                "transport(item, location)"
            ],
            "perception": [
                "detect(objects)",
                "identify(object)",
                "find_object(type)",
                "inspect(location)"
            ],
            "communication": [
                "speak(text)",
                "listen()",
                "request_help()",
                "report_status()"
            ]
        }
    
    def decompose_task(self, task_description: str, world_state: Dict[str, Any]) -> Task:
        """
        Decompose a high-level task into subtasks and actions using LLM
        """
        prompt = self._create_decomposition_prompt(task_description, world_state)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                functions=[
                    {
                        "name": "decompose_task",
                        "description": "Decompose a complex task into subtasks and actions",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "high_level_task": {
                                    "type": "string",
                                    "description": "The original high-level task"
                                },
                                "subtasks": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "description": {"type": "string"},
                                            "actions": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "action": {"type": "string"},
                                                        "parameters": {"type": "object"}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                ],
                function_call={"name": "decompose_task"}
            )
            
            # Parse the LLM response
            function_args = json.loads(response.choices[0].message.function_call.arguments)
            return self._create_task_structure(function_args)
            
        except Exception as e:
            rospy.logerr(f"Error in task decomposition: {e}")
            # Fallback: create a simple task with the original description
            return Task(name=task_description, description=task_description)
    
    def _create_decomposition_prompt(self, task_description: str, world_state: Dict[str, Any]):
        """Create a detailed prompt for task decomposition"""
        return f"""
        Task: {task_description}
        
        Current World State: {json.dumps(world_state, indent=2)}
        
        Available Actions: {json.dumps(self.action_vocabulary, indent=2)}
        
        Please decompose this high-level task into a sequence of subtasks and specific actions that a robot can execute.
        
        For each subtask:
        1. Give it a clear name and description
        2. List the specific actions required to complete the subtask
        3. Specify parameters for each action
        
        The actions should be from the available actions list above.
        
        Respond in JSON format using the function decomposition.
        """
    
    def _get_system_prompt(self):
        """System prompt to guide the LLM's behavior"""
        return """
        You are an expert in robotics task planning and decomposition. Your role is to take high-level human instructions and break them down into concrete, executable actions for a robot.
        
        When decomposing tasks:
        1. Think step by step about what needs to be done
        2. Consider the current state of the world
        3. Break complex tasks into smaller, manageable subtasks
        4. Use specific, executable actions from the provided vocabulary
        5. Include all necessary parameters for each action
        6. Ensure logical order and dependencies between actions
        
        Always respond with structured JSON using the provided function format.
        """
    
    def _create_task_structure(self, decomposition_data: Dict[str, Any]) -> Task:
        """Convert LLM response to Task structure"""
        main_task = Task(
            name=decomposition_data["high_level_task"],
            description=decomposition_data["high_level_task"]
        )
        
        for subtask_data in decomposition_data["subtasks"]:
            subtask = Task(
                name=subtask_data["name"],
                description=subtask_data["description"]
            )
            
            # Convert actions to Action objects
            for action_data in subtask_data.get("actions", []):
                if "action" in action_data:
                    action = Action(
                        name=action_data["action"],
                        parameters=action_data.get("parameters", {}),
                        preconditions=[],  # Could be inferred based on action parameters
                        effects=[]  # Could be inferred based on action parameters
                    )
                    subtask.actions.append(action)
            
            main_task.add_subtask(subtask)
        
        return main_task

# Example of the decomposer in action
if __name__ == "__main__":
    # Example usage
    world_state = {
        "robot_location": "kitchen",
        "available_objects": ["apple", "orange", "banana", "water_bottle"],
        "object_locations": {
            "apple": "counter",
            "orange": "bowl",
            "banana": "basket",
            "water_bottle": "table"
        },
        "battery_level": 0.85,
        "robot_capabilities": ["navigation", "manipulation", "speech"]
    }
    
    decomposer = LLMTaskDecomposer()
    complex_task = "Bring me the red apple from the counter in the kitchen and then return to the living room"
    
    task_plan = decomposer.decompose_task(complex_task, world_state)
    print(json.dumps(task_plan.to_json(), indent=2))
```

### Multi-Step Task Planning

For complex multi-step tasks, we need more sophisticated decomposition:

```python
import asyncio
from typing import Callable, Awaitable

class MultiStepPlanner:
    """Handles complex multi-step task planning with LLM assistance"""
    
    def __init__(self, llm_decomposer: LLMTaskDecomposer):
        self.decomposer = llm_decomposer
        self.execution_context = {}
        self.completed_steps = []
        self.failed_steps = []
    
    async def plan_multi_step_task(
        self, 
        task_description: str, 
        world_state: Dict[str, Any],
        max_depth: int = 3
    ) -> List[Task]:
        """Plan a multi-step task with depth-limited decomposition"""
        plan = []
        
        # Initial decomposition
        main_task = await self.decomposer.decompose_task_async(task_description, world_state)
        plan.append(main_task)
        
        # Further decompose if needed
        await self._deep_decompose(main_task, world_state, max_depth - 1)
        
        return plan
    
    async def _deep_decompose(self, task: Task, world_state: Dict[str, Any], remaining_depth: int):
        """Recursively decompose complex subtasks"""
        if remaining_depth <= 0:
            return
        
        # Recursively decompose subtasks that are still high-level
        for subtask in task.subtasks:
            if self._is_high_level_task(subtask):
                # Further decompose this subtask
                detailed_subtask = await self.decomposer.decompose_task_async(
                    subtask.description, 
                    world_state
                )
                
                # Replace the subtask with its decomposition
                subtask.subtasks = detailed_subtask.subtasks
                subtask.actions = detailed_subtask.actions
                
                # Continue decomposition recursively
                await self._deep_decompose(subtask, world_state, remaining_depth - 1)
    
    def _is_high_level_task(self, task: Task) -> bool:
        """Determine if a task is still abstract and needs more decomposition"""
        # A task is high-level if it has no primitive actions
        return len(task.actions) == 0 and len(task.subtasks) > 0
    
    async def execute_with_adaptation(
        self,
        plan: List[Task],
        world_state_callback: Callable[[], Awaitable[Dict[str, Any]]],
        action_executor: Callable[[Action], Awaitable[bool]]
    ):
        """Execute plan with real-time adaptation based on changing world state"""
        for task in plan:
            current_world_state = await world_state_callback()
            
            # Check if current task is still valid given the world state
            if not self._is_task_valid(task, current_world_state):
                # Need to replan this task
                rospy.loginfo(f"Replanning task {task.name} due to world state changes")
                task = await self.decomposer.decompose_task_async(
                    task.description,
                    current_world_state
                )
            
            # Execute the task
            success = await self._execute_task(
                task, 
                world_state_callback, 
                action_executor
            )
            
            if not success:
                self.failed_steps.append(task)
                # Decide whether to continue, fail, or replan
                if await self._can_recover(task, current_world_state):
                    continue  # Try to recover and continue
                else:
                    raise Exception(f"Failed to execute task {task.name}")
            else:
                self.completed_steps.append(task)
    
    def _is_task_valid(self, task: Task, world_state: Dict[str, Any]) -> bool:
        """Check if the task is still valid given current world state"""
        # For now, a simple check - in practice this would be more complex
        # Check if preconditions are still satisfied
        for action in task.actions:
            if not self._action_preconditions_met(action, world_state):
                return False
        return True
    
    def _action_preconditions_met(self, action: Action, world_state: Dict[str, Any]) -> bool:
        """Check if action preconditions are met"""
        # Simplified check - in practice, this would be more sophisticated
        # This is a basic example - real implementation would be more detailed
        return True
    
    async def _execute_task(
        self,
        task: Task,
        world_state_callback: Callable[[], Awaitable[Dict[str, Any]]],
        action_executor: Callable[[Action], Awaitable[bool]]
    ) -> bool:
        """Execute a single task (subtasks or primitive actions)"""
        if task.actions:  # Has primitive actions to execute
            for action in task.actions:
                success = await action_executor(action)
                if not success:
                    return False
            return True
        elif task.subtasks:  # Need to execute and complete subtasks
            for subtask in task.subtasks:
                current_world_state = await world_state_callback()
                success = await self._execute_task(subtask, world_state_callback, action_executor)
                if not success:
                    return False
            return True
        else:
            # Task has neither actions nor subtasks - invalid
            return False
    
    async def _can_recover(self, task: Task, world_state: Dict[str, Any]) -> bool:
        """Check if robot can recover from task failure"""
        # Use LLM to determine recovery options
        recovery_prompt = f"""
        The robot failed to execute the following task: {task.description}
        
        Current world state: {json.dumps(world_state, indent=2)}
        
        What recovery options are available? Can the robot retry the task with different parameters?
        Should the robot abandon this task and proceed to the next one?
        
        Respond with a JSON object containing 'can_recover' (boolean) and 'recovery_options' (array of strings).
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.decomposer.model_name,
                messages=[
                    {"role": "system", "content": "You are a robotics expert helping with failure recovery."}, 
                    {"role": "user", "content": recovery_prompt}
                ]
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("can_recover", False)
        except:
            return False  # Conservative: assume can't recover if LLM fails
```

## Natural Language to Action Sequence Translation

### Semantic Parsing with LLMs

```python
class SemanticParser:
    """Converts natural language instructions to executable action sequences"""
    
    def __init__(self, llm_model: str = "gpt-4"):
        self.model = llm_model
        self.known_entities = set()  # Objects, locations, etc. that robot knows about
        self.action_templates = {}   # Templates for common action patterns
    
    def parse_instruction(
        self, 
        instruction: str, 
        world_state: Dict[str, Any],
        robot_capabilities: List[str]
    ) -> List[Action]:
        """
        Parse a natural language instruction into a sequence of robot actions
        """
        prompt = self._create_parsing_prompt(instruction, world_state, robot_capabilities)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_parsing_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                functions=[
                    {
                        "name": "parse_instruction",
                        "description": "Parse natural language instruction into robot actions",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "intention": {"type": "string"},
                                "action_sequence": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "action": {"type": "string"},
                                            "parameters": {"type": "object"},
                                            "justification": {"type": "string"}
                                        }
                                    }
                                },
                                "required_objects": {"type": "array", "items": {"type": "string"}},
                                "required_locations": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    }
                ],
                function_call={"name": "parse_instruction"}
            )
            
            function_args = json.loads(response.choices[0].message.function_call.arguments)
            return self._actions_from_parsed_data(function_args, world_state)
            
        except Exception as e:
            rospy.logerr(f"Error parsing instruction: {e}")
            # Return empty list if parsing fails
            return []
    
    def _create_parsing_prompt(
        self, 
        instruction: str, 
        world_state: Dict[str, Any], 
        robot_capabilities: List[str]
    ) -> str:
        """Create prompt for semantic parsing"""
        return f"""
        Instruction: "{instruction}"
        
        Available Robot Capabilities: {', '.join(robot_capabilities)}
        
        Current World State: 
        {json.dumps(world_state, indent=2)}
        
        Please parse this natural language instruction into a sequence of specific actions that the robot can execute.
        
        For each action in the sequence:
        1. Specify the exact action (from the robot capabilities)
        2. Include all necessary parameters (object IDs, coordinates, etc.)
        3. Provide a brief justification for why this action is needed
        
        Also identify:
        - Any required objects that are needed but not currently in the world state
        - Any required locations that are needed but not currently known
        
        Remember to consider the world state when selecting objects and locations.
        If an object is not available, include it in required_objects.
        If a location is not known, include it in required_locations.
        
        Respond in JSON format using the function.
        """
    
    def _get_parsing_system_prompt(self) -> str:
        """System prompt for semantic parsing"""
        return """
        You are a semantic parsing expert for robotics. Your job is to convert natural language instructions into specific, executable robot actions.
        
        When parsing instructions:
        1. Identify the overall intention behind the instruction
        2. Break down the high-level goal into specific, low-level actions
        3. Use the robot's available capabilities
        4. Extract specific parameters (objects, locations, values) from the natural language
        5. Consider the current world state to contextualize the instruction
        6. Maintain logical order and dependencies between actions
        
        Actions should be directly executable by the robot.
        Always respond in the specified JSON format.
        """
    
    def _actions_from_parsed_data(
        self, 
        parsed_data: Dict[str, Any], 
        world_state: Dict[str, Any]
    ) -> List[Action]:
        """Convert parsed data to Action objects"""
        actions = []
        
        for action_data in parsed_data.get("action_sequence", []):
            action_name = action_data["action"]
            parameters = action_data.get("parameters", {})
            justification = action_data.get("justification", "")
            
            # Add world context to parameters if needed
            processed_params = self._process_parameters(parameters, world_state)
            
            action = Action(
                name=action_name,
                parameters=processed_params,
                preconditions=[],  # Will be computed based on action
                effects=[]  # Will be computed based on action
            )
            
            actions.append(action)
        
        # Update known entities
        for obj in parsed_data.get("required_objects", []):
            self.known_entities.add(obj)
        
        for loc in parsed_data.get("required_locations", []):
            self.known_entities.add(loc)
        
        return actions
    
    def _process_parameters(self, parameters: Dict[str, Any], world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process parameters, resolving references to objects and locations in world state"""
        processed = parameters.copy()
        
        # Resolve object references
        for key, value in parameters.items():
            if isinstance(value, str):
                # Check if this is a reference to an object in world state
                if value in world_state.get("object_locations", {}):
                    # Resolve to actual location/object info
                    actual_location = world_state["object_locations"][value]
                    processed[key] = actual_location
                elif value in world_state.get("available_objects", []):
                    # Value is an object reference
                    processed[key] = self._get_object_info(value, world_state)
        
        return processed
    
    def _get_object_info(self, object_name: str, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed information about an object from world state"""
        # This would query the robot's object database or perception system
        # For now, return basic info
        location = world_state.get("object_locations", {}).get(object_name, "unknown")
        return {
            "name": object_name,
            "type": "object",
            "location": location
        }

# Integration with robot execution system
class NLInstructionExecutor:
    """Executes natural language instructions on a robot using LLM-based planning"""
    
    def __init__(self):
        self.parser = SemanticParser()
        self.planner = MultiStepPlanner(LLMTaskDecomposer())
        self.robot_interface = None  # Will be initialized with actual robot interface
        
    def execute_instruction(
        self, 
        instruction: str, 
        world_state: Dict[str, Any],
        robot_capabilities: List[str]
    ) -> bool:
        """
        Execute a natural language instruction on the robot
        """
        rospy.loginfo(f"Processing instruction: {instruction}")
        
        # Parse the instruction into actions
        actions = self.parser.parse_instruction(
            instruction, 
            world_state, 
            robot_capabilities
        )
        
        if not actions:
            rospy.logerr("No actions generated from instruction")
            return False
        
        # Execute the action sequence
        success = self._execute_action_sequence(actions, world_state)
        
        if success:
            rospy.loginfo("Instruction executed successfully")
        else:
            rospy.logerr("Instruction execution failed")
        
        return success
    
    def _execute_action_sequence(self, actions: List[Action], world_state: Dict[str, Any]) -> bool:
        """Execute a sequence of actions on the robot"""
        for i, action in enumerate(actions):
            rospy.loginfo(f"Executing action {i+1}/{len(actions)}: {action.name}")
            
            # Check preconditions
            if not self._check_preconditions(action, world_state):
                rospy.logerr(f"Preconditions not met for action: {action.name}")
                return False
            
            # Execute the action
            success = self._execute_single_action(action)
            
            if not success:
                rospy.logerr(f"Action failed: {action.name}")
                return False
            
            # Update world state based on action effects
            self._update_world_state(action, world_state)
        
        return True
    
    def _check_preconditions(self, action: Action, world_state: Dict[str, Any]) -> bool:
        """Check if action preconditions are satisfied in the current world state"""
        # This would implement sophisticated precondition checking
        # For now, return True for all actions
        return True
    
    def _execute_single_action(self, action: Action) -> bool:
        """Execute a single robot action"""
        # This would interface with the actual robot control system
        # Implementation depends on the specific robot platform
        
        # Log the action for debugging
        rospy.loginfo(f"Robot executing: {action.name} with parameters: {action.parameters}")
        
        # In a real implementation, this would call robot-specific methods
        # For example, if using ROS:
        # self.robot_move_to(action.parameters.get('location'))
        # self.robot_grasp(action.parameters.get('object'))
        
        # For demonstration, just return True
        return True
    
    def _update_world_state(self, action: Action, world_state: Dict[str, Any]):
        """Update world state based on action effects"""
        # This would update the robot's internal world model
        # based on the effects of the executed action
        
        # For example, if action was 'move_to(location)':
        # world_state['robot_location'] = action.parameters.get('location')
        
        # For example, if action was 'grasp(object)':
        # world_state['held_object'] = action.parameters.get('object')
        # del world_state['object_locations'][action.parameters.get('object')]
        
        # Implementation depends on action types and world state structure
        pass

# Example usage
if __name__ == "__main__":
    executor = NLInstructionExecutor()
    
    # Example world state
    world_state = {
        "robot_location": "kitchen",
        "available_objects": ["apple", "orange", "water_bottle"],
        "object_locations": {
            "apple": "counter_front_left",
            "orange": "basket_center",
            "water_bottle": "table_near_window"
        }
    }
    
    robot_capabilities = ["navigation", "manipulation", "perception"]
    
    # Natural language instruction
    instruction = "Go to the counter, pick up the apple, and bring it to me in the living room"
    
    success = executor.execute_instruction(instruction, world_state, robot_capabilities)
    print(f"Instruction execution result: {success}")
```

## Planning Algorithms Enhanced with LLMs

### LLM-Augmented A* Pathfinding

```python
import heapq
from typing import Tuple, List, Dict, Optional
import math

class LLMEnhancedPathfinder:
    """A* pathfinding enhanced with LLM for dynamic cost calculation"""
    
    def __init__(self, llm_model: str = "gpt-4"):
        self.model = llm_model
        self.cost_cache = {}  # Cache LLM-based cost calculations
    
    def find_path(
        self, 
        start: Tuple[int, int], 
        goal: Tuple[int, int], 
        grid: List[List[int]],  # 0 = free, 1 = obstacle
        world_state: Dict[str, Any]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find path using A* with LLM-augmented cost calculation
        """
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        closed_set = set()
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            closed_set.add(current)
            
            for neighbor in self._get_neighbors(current, grid):
                if neighbor in closed_set:
                    continue
                
                # Calculate cost using traditional and LLM-enhanced factors
                tentative_g = g_score[current] + self._llm_augmented_cost(
                    current, neighbor, world_state
                )
                
                if neighbor not in [item[1] for item in open_set] or tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def _llm_augmented_cost(
        self, 
        from_node: Tuple[int, int], 
        to_node: Tuple[int, int], 
        world_state: Dict[str, Any]
    ) -> float:
        """
        Calculate movement cost using LLM for contextual factors
        """
        cache_key = (from_node, to_node, tuple(sorted(world_state.items())))
        if cache_key in self.cost_cache:
            return self.cost_cache[cache_key]
        
        # Traditional cost (Manhattan distance or Euclidean)
        base_cost = self._euclidean_distance(from_node, to_node)
        
        # Query LLM for contextual cost factors
        llm_prompt = f"""
        Calculate the movement cost from position {from_node} to {to_node}.
        
        World state: {json.dumps(world_state, indent=2)}
        
        Consider the following factors which might increase the movement cost:
        - Areas known to be dangerous or difficult to traverse
        - Areas with fragile or valuable objects nearby
        - Crowded areas where movement might disturb people
        - Areas currently undergoing activities
        - Visibility conditions at the location
        - Surface conditions (slippery, uneven, etc.)
        
        Return only the additional cost factor (0.0 to 2.0) as a floating-point number.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a navigation expert assessing movement costs."},
                    {"role": "user", "content": llm_prompt}
                ]
            )
            
            additional_cost = float(response.choices[0].message.content.strip())
            total_cost = base_cost + max(0, additional_cost)  # Ensure non-negative
            
        except Exception as e:
            rospy.logwarn(f"LLM cost calculation failed, using base cost: {e}")
            total_cost = base_cost
        
        # Cache the result
        self.cost_cache[cache_key] = total_cost
        return total_cost
    
    def _euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Heuristic function for A* (Euclidean distance)"""
        return self._euclidean_distance(pos, goal)
    
    def _get_neighbors(self, pos: Tuple[int, int], grid: List[List[int]]) -> List[Tuple[int, int]]:
        """Get valid neighbors for a position"""
        neighbors = []
        rows, cols = len(grid), len(grid[0])
        
        # 4-connected neighbors
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            r, c = pos[0] + dr, pos[1] + dc
            if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 0:
                neighbors.append((r, c))
        
        return neighbors
    
    def _reconstruct_path(
        self, 
        came_from: Dict[Tuple[int, int], Tuple[int, int]], 
        current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
```

### LLM-Based Task Replanning

```python
class LLMTaskReplanner:
    """Handles task replanning when execution fails or conditions change"""
    
    def __init__(self, llm_model: str = "gpt-4"):
        self.model = llm_model
        self.execution_history = []
    
    def replan_task(
        self,
        original_task: Task,
        world_state: Dict[str, Any],
        failure_reason: str,
        execution_trace: List[Dict[str, Any]]
    ) -> Optional[Task]:
        """Generate a new plan when the original plan fails"""
        prompt = self._create_replanning_prompt(
            original_task, world_state, failure_reason, execution_trace
        )
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_replanning_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                functions=[
                    {
                        "name": "generate_new_plan",
                        "description": "Generate a new task plan to achieve the goal",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "rationale": {"type": "string"},
                                "new_plan": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "description": {"type": "string"},
                                            "actions": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "action": {"type": "string"},
                                                        "parameters": {"type": "object"}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                ],
                function_call={"name": "generate_new_plan"}
            )
            
            function_args = json.loads(response.choices[0].message.function_call.arguments)
            return self._create_task_from_replanning(function_args)
            
        except Exception as e:
            rospy.logerr(f"Error in task replanning: {e}")
            return None
    
    def _create_replanning_prompt(
        self,
        original_task: Task,
        world_state: Dict[str, Any],
        failure_reason: str,
        execution_trace: List[Dict[str, Any]]
    ) -> str:
        """Create prompt for task replanning"""
        return f"""
        Original Task: {original_task.description}
        
        Execution Failed Reason: {failure_reason}
        
        World State: {json.dumps(world_state, indent=2)}
        
        Execution Trace: {json.dumps(execution_trace, indent=2)}
        
        The robot attempted to execute the original task but failed due to the reason above. 
        Please generate a new plan that achieves the same goal while addressing the failure cause.
        
        Consider:
        - Alternative approaches that avoid the failure
        - Current world state that might provide new opportunities
        - Resources that are now available or unavailable
        - Lessons learned from the failed execution
        
        Provide a new plan with specific actions that have better chances of success.
        """
    
    def _get_replanning_system_prompt(self) -> str:
        """System prompt for task replanning"""
        return """
        You are an expert in robotics task planning and adaptation. When a robot's plan fails, you generate a new plan that achieves the same goal while avoiding the cause of failure.
        
        When creating a new plan:
        1. Analyze why the original plan failed
        2. Consider what has changed in the world state
        3. Identify alternative approaches to achieve the same goal
        4. Generate specific, executable actions
        5. Explain your reasoning for the changes
        
        Always provide a complete plan with specific actions.
        """
    
    def _create_task_from_replanning(self, replan_data: Dict[str, Any]) -> Task:
        """Create Task object from replanning data"""
        main_task = Task(
            name=f"replan-{replan_data.get('rationale', 'adaptive plan')}",
            description=replan_data.get("rationale", "Adaptive plan after failure")
        )
        
        for plan_item in replan_data.get("new_plan", []):
            subtask = Task(
                name=plan_item["name"],
                description=plan_item["description"]
            )
            
            for action_data in plan_item.get("actions", []):
                action = Action(
                    name=action_data["action"],
                    parameters=action_data.get("parameters", {}),
                    preconditions=[],  # Could be inferred from context
                    effects=[]  # Could be inferred from context
                )
                subtask.actions.append(action)
            
            main_task.add_subtask(subtask)
        
        return main_task
    
    def assess_success_probability(
        self, 
        task: Task, 
        world_state: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """
        Use LLM to assess the probability of task success and identify potential issues
        """
        prompt = f"""
        Task: {task.description}
        
        World State: {json.dumps(world_state, indent=2)}
        
        Assess the probability that the following task will succeed (0.0 to 1.0) and identify potential issues:
        {json.dumps(task.to_json(), indent=2)}
        
        Return a JSON object with:
        - 'success_probability' (float): Probability of success (0.0-1.0)
        - 'potential_issues' (array of strings): Potential problems that might arise
        - 'recommendations' (array of strings): Suggestions to improve success probability
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a robotics task assessment expert."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            assessment = json.loads(response.choices[0].message.content)
            return (
                assessment.get("success_probability", 0.5),
                assessment.get("potential_issues", []),
                assessment.get("recommendations", [])
            )
            
        except Exception as e:
            rospy.logwarn(f"Success probability assessment failed: {e}")
            return 0.5, [], []

# Integration with execution system
class AdaptivePlanner:
    """Integrates all planning components with adaptive capabilities"""
    
    def __init__(self):
        self.llm_decomposer = LLMTaskDecomposer()
        self.llm_pathfinder = LLMEnhancedPathfinder()
        self.llm_replanner = LLMTaskReplanner()
        self.executor = NLInstructionExecutor()
    
    def execute_with_adaptation(
        self,
        instruction: str,
        initial_world_state: Dict[str, Any],
        robot_capabilities: List[str],
        success_callback: Optional[Callable] = None,
        failure_callback: Optional[Callable] = None
    ) -> bool:
        """Execute instruction with adaptive planning and replanning"""
        # Initial planning
        rospy.loginfo(f"Initial planning for: {instruction}")
        
        # Parse and plan
        actions = self.executor.parser.parse_instruction(
            instruction,
            initial_world_state,
            robot_capabilities
        )
        
        if not actions:
            rospy.logerr("Failed to parse instruction into actions")
            if failure_callback:
                failure_callback(instruction, "Failed to parse instruction")
            return False
        
        # Execute with monitoring and adaptation
        execution_result = self._execute_with_monitoring(
            actions, initial_world_state, instruction, 
            success_callback, failure_callback
        )
        
        return execution_result
    
    def _execute_with_monitoring(
        self,
        actions: List[Action],
        world_state: Dict[str, Any],
        instruction: str,
        success_callback: Optional[Callable],
        failure_callback: Optional[Callable]
    ) -> bool:
        """Execute actions with monitoring for failures and adaptation"""
        execution_trace = []
        
        for i, action in enumerate(actions):
            rospy.loginfo(f"Executing action {i+1}/{len(actions)}: {action.name}")
            
            # Assess success probability before execution
            success_prob, issues, recommendations = self.llm_replanner.assess_success_probability(
                Task(name=action.name, description=str(action.parameters)),
                world_state
            )
            
            if success_prob < 0.3:  # Low confidence
                rospy.logwarn(f"Low success probability ({success_prob:.2f}) for action: {action.name}")
                rospy.logwarn(f"Potential issues: {issues}")
                rospy.logwarn(f"Recommendations: {recommendations}")
                
                # Maybe replan or seek human intervention
                if not self._handle_low_confidence_action(
                    action, world_state, recommendations
                ):
                    if failure_callback:
                        failure_callback(instruction, f"Aborted due to low confidence: {action.name}")
                    return False
            
            # Attempt to execute the action
            try:
                success = self.executor._execute_single_action(action)
                
                action_result = {
                    "action": action.name,
                    "parameters": action.parameters,
                    "success": success,
                    "timestamp": time.time()
                }
                
                execution_trace.append(action_result)
                
                if not success:
                    rospy.logerr(f"Action failed: {action.name}")
                    
                    # Attempt replanning
                    replanned_task = self.llm_replanner.replan_task(
                        Task(name=action.name, description="Failed action"),
                        world_state,
                        f"Action {action.name} failed",
                        execution_trace
                    )
                    
                    if replanned_task and replanned_task.actions:
                        rospy.loginfo("Executing replanned actions...")
                        replan_success = self._execute_with_monitoring(
                            replanned_task.actions, 
                            world_state, 
                            instruction, 
                            success_callback, 
                            failure_callback
                        )
                        
                        if replan_success:
                            return True
                        else:
                            # Replanning also failed
                            if failure_callback:
                                failure_callback(instruction, f"Both original and replanned actions failed for: {action.name}")
                            return False
                    else:
                        # No valid replan available
                        if failure_callback:
                            failure_callback(instruction, f"Action failed and replanning unsuccessful: {action.name}")
                        return False
                
                # Update world state after each action
                self.executor._update_world_state(action, world_state)
                
            except Exception as e:
                rospy.logerr(f"Execution error for action {action.name}: {e}")
                if failure_callback:
                    failure_callback(instruction, f"Execution error: {action.name} - {str(e)}")
                return False
        
        # All actions succeeded
        rospy.loginfo("All actions executed successfully")
        if success_callback:
            success_callback(instruction, execution_trace)
        return True
    
    def _handle_low_confidence_action(
        self, 
        action: Action, 
        world_state: Dict[str, Any], 
        recommendations: List[str]
    ) -> bool:
        """Handle actions with low success probability"""
        # For now, just log recommendations
        rospy.logwarn(f"Recommended actions for low-confidence execution: {recommendations}")
        # In practice, this might involve requesting human help, sensor verification, etc.
        return True  # Continue execution for now
```

## Practical Implementation Example: Autonomous Household Assistant

Here's a practical example combining all concepts:

```python
#!/usr/bin/env python3

import rospy
import json
import time
from geometry_msgs.msg import PoseStamped
from actionlib_msgs.msg import GoalStatusArray
from std_msgs.msg import String
from sensor_msgs.msg import Image
import threading

class AutonomousHouseholdAssistant:
    """Complete implementation of LLM-based cognitive planning for household tasks"""
    
    def __init__(self):
        rospy.init_node('autonomous_household_assistant')
        
        # Initialize planning components
        self.adaptive_planner = AdaptivePlanner()
        
        # Robot state
        self.current_world_state = {
            "robot_location": {"x": 0.0, "y": 0.0, "z": 0.0},
            "held_object": None,
            "battery_level": 1.0,
            "available_objects": [],
            "object_locations": {},
            "rooms_visited": [],
            "people_present": []
        }
        
        # ROS interfaces
        self.nav_goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.voice_cmd_sub = rospy.Subscriber('/voice_commands', String, self.voice_command_callback)
        self.status_pub = rospy.Publisher('/robot_status', String, queue_size=10)
        
        # Task queue
        self.task_queue = []
        self.current_task = None
        self.task_lock = threading.Lock()
        
        # Robot capabilities
        self.robot_capabilities = [
            "navigation", "manipulation", "perception", 
            "speech", "grasping", "place_object"
        ]
        
        rospy.loginfo("Autonomous Household Assistant initialized")
    
    def voice_command_callback(self, msg):
        """Handle incoming voice commands"""
        command = msg.data
        rospy.loginfo(f"Received voice command: {command}")
        
        with self.task_lock:
            # Add to task queue
            self.task_queue.append({
                "instruction": command,
                "timestamp": rospy.Time.now(),
                "status": "queued"
            })
            
            # Publish status update
            status_msg = String()
            status_msg.data = f"Queued task: {command}"
            self.status_pub.publish(status_msg)
        
        # Process queue in background
        self.process_task_queue()
    
    def process_task_queue(self):
        """Process the task queue asynchronously"""
        thread = threading.Thread(target=self._process_tasks)
        thread.daemon = True
        thread.start()
    
    def _process_tasks(self):
        """Process tasks in the queue"""
        while not rospy.is_shutdown() and self.task_queue:
            with self.task_lock:
                if not self.task_queue:
                    break
                
                task = self.task_queue.pop(0)
                self.current_task = task
            
            rospy.loginfo(f"Processing task: {task['instruction']}")
            
            # Update status
            status_msg = String()
            status_msg.data = f"Processing: {task['instruction']}"
            self.status_pub.publish(status_msg)
            
            # Execute with adaptive planning
            success = self.adaptive_planner.execute_with_adaptation(
                task['instruction'],
                self.current_world_state,
                self.robot_capabilities,
                success_callback=self.on_task_success,
                failure_callback=self.on_task_failure
            )
            
            # Update task status
            self.current_task = None
            
            if success:
                rospy.loginfo(f"Task completed successfully: {task['instruction']}")
            else:
                rospy.logerr(f"Task failed: {task['instruction']}")
    
    def on_task_success(self, instruction: str, execution_trace: List[Dict]):
        """Callback for successful task completion"""
        rospy.loginfo(f"Task completed: {instruction}")
        
        # Update world state based on execution
        for action_result in execution_trace:
            self._update_world_state_from_execution(action_result)
        
        # Publish success status
        status_msg = String()
        status_msg.data = f"Task completed: {instruction}"
        self.status_pub.publish(status_msg)
        
        # Speak confirmation
        self._speak(f"I have completed the task: {instruction}")
    
    def on_task_failure(self, instruction: str, failure_reason: str):
        """Callback for task failure"""
        rospy.logerr(f"Task failed: {instruction} - {failure_reason}")
        
        # Publish failure status
        status_msg = String()
        status_msg.data = f"Task failed: {instruction} - {failure_reason}"
        self.status_pub.publish(status_msg)
        
        # Request human assistance
        self._speak(f"I'm sorry, I couldn't complete the task: {instruction}. {failure_reason}")
        self._request_human_help()
    
    def _update_world_state_from_execution(self, action_result: Dict):
        """Update world state based on action execution results"""
        action_name = action_result['action']
        
        if action_name.startswith('move_to') or action_name.startswith('navigate_to'):
            # Update robot location
            params = action_result['parameters']
            if 'x' in params and 'y' in params:
                self.current_world_state['robot_location']['x'] = params['x']
                self.current_world_state['robot_location']['y'] = params['y']
                if 'z' in params:
                    self.current_world_state['robot_location']['z'] = params['z']
        
        elif action_name.startswith('grasp') or action_name.startswith('pickup'):
            # Update held object
            params = action_result['parameters']
            if 'object' in params:
                self.current_world_state['held_object'] = params['object']
                # Remove from available objects
                obj_name = params['object']
                if obj_name in self.current_world_state['available_objects']:
                    self.current_world_state['available_objects'].remove(obj_name)
                if obj_name in self.current_world_state['object_locations']:
                    del self.current_world_state['object_locations'][obj_name]
        
        elif action_name.startswith('place') or action_name.startswith('release'):
            # Release held object
            held_obj = self.current_world_state['held_object']
            if held_obj:
                self.current_world_state['held_object'] = None
                # Add back to available objects at new location
                params = action_result['parameters']
                if 'location' in params:
                    location = params['location']
                    self.current_world_state['available_objects'].append(held_obj)
                    self.current_world_state['object_locations'][held_obj] = location
    
    def _speak(self, text: str):
        """Speak text using robot's speech system"""
        # This would interface with the robot's text-to-speech system
        rospy.loginfo(f"Robot says: {text}")
        # In a real implementation, publish to speech topic, etc.
    
    def _request_human_help(self):
        """Request human assistance when tasks fail repeatedly"""
        # This would implement various help-requesting mechanisms
        rospy.logwarn("Requesting human help...")
        # Could send notification, light LEDs, play sound, etc.
    
    def get_world_state(self) -> Dict[str, Any]:
        """Get current world state with real-time sensor updates"""
        # This would integrate with perception systems to get current world state
        # For now, return the maintained state
        return self.current_world_state
    
    def run(self):
        """Main run loop"""
        rospy.loginfo("Autonomous Household Assistant starting...")
        
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutdown requested")

if __name__ == "__main__":
    assistant = AutonomousHouseholdAssistant()
    
    # Add a test task
    test_instruction = "Go to the kitchen, pick up the red mug from the table, and bring it to the living room"
    
    # In practice, tasks would come from voice commands or other interfaces
    # For testing, we can add directly
    with assistant.task_lock:
        assistant.task_queue.append({
            "instruction": test_instruction,
            "timestamp": rospy.Time.now(),
            "status": "queued"
        })
    
    assistant.run()
```

## Exercises

1. **Task Decomposition Exercise**: Implement a system that uses an LLM to decompose complex household tasks (e.g., "Prepare coffee for John") into a sequence of primitive robot actions.

2. **Natural Language Translation Exercise**: Create a semantic parser that translates natural language instructions into robot action sequences, handling variations in how the same task can be expressed (e.g., "Get the book" vs. "Bring me the novel").

3. **Adaptive Planning Exercise**: Implement a failure recovery system that detects when a robot's plan is failing and uses an LLM to generate an alternative approach to achieve the same goal.

## Summary

This chapter covered cognitive planning with LLMs, focusing on how Large Language Models can enhance robot planning capabilities. We explored task decomposition techniques that break complex instructions into executable actions, natural language processing that translates human commands into robot tasks, planning algorithms augmented with LLM reasoning, and adaptive systems that replan when faced with failures or changing conditions.

The key takeaways include:
- LLMs excel at decomposing complex tasks into primitive actions
- Natural language interfaces make robots more accessible to non-experts
- LLM-augmented planning considers contextual factors beyond traditional algorithms
- Adaptive planning with LLMs enables recovery from failures and changing conditions
- Successful implementation requires careful integration of LLM reasoning with robotic action execution

## Cross-references

For foundational concepts about Physical AI, see [Chapter 1: Introduction to Physical AI](../part-01-foundations/chapter-1). For perception systems that support cognitive planning, see [Chapter 2: The Robotic Sensorium](../part-01-foundations/chapter-2). For voice-to-action systems that often complement cognitive planning, see [Chapter 14: Voice-to-Action Systems](../part-05-vla/chapter-14).