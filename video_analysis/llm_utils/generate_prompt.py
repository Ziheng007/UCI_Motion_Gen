import os

import json
import re


def generate_sequence_explanation_prompt(original_action):
    """
    Generate a prompt for the LLM to think about fine motion control for a given action.
    :param original_action: Perform a signature James Bond pose with a dramatic turn and gunpoint.
    :return:
    :example:
    step1: The man pivots on his heel, shifting his weight to turn his body sharply while extending his opposite arm outward, creating a dramatic stance.
    step2: The man's wrist flexes as he raises his hand to point the imaginary gun forward, aligning his arm with his shoulder for precision and balance.
    <SEQUENCEEND>
    """
    prompt = f"""
    The action 'original_action: {original_action}' may require detailed control over specific body parts. 
    Please evaluate the action and think carefully about how the movement breaks down into smaller, distinct actions. 
    Each step should represent a single, concrete movement without including states or transitional descriptions or stationary motion or pose.
    Each step should represent a single, concrete movement without including states or transitional descriptions or stationary motion or pose.
    
    After thinking, provide a structured list of the steps involved in performing this action.
    
    <original_action>
    {original_action}
    </original_action>
    
    
    <REQUIREMENT>
    - Focus on describing the dynamic movement.
    - Highlight the necessary coordination between body parts.
    - Emphasize the importance of actions: Each step must include key movement details, avoiding redundancy or state descriptions.
    - Ensure each step represents a distinct action rather than an intermediate state.
    - Streamline the steps: Merge steps as much as possible, ensuring each step contains actual dynamic movements rather than empty descriptions.
    
        <FORMAT>
    
        The number of steps should be 1, 2, 3, or 4, depending on the TEMPORAL complexity of the action. Do not use too many steps if the action is simple. 2~3 steps are usually enough.
    
        For each step, use the words 'The man...' or 'The man's ...(body part)' to describe the action.
        Ensure the explanation follows this structure:
        step1: The ...
        step2: The ...
        ...
    
    </FORMAT>
    
    注意 一定要符合格式 我会按照这个拆分：
    <code>
              # 清理输入的 sequence_explanation
      sequence_explanation = sequence_explanation.strip()
      
      # 使用正则表达式匹配所有的步骤和对应的描述
      # 模式解释：
      #   - (?m): 多行模式，使 ^ 和 $ 匹配每一行的开头和结尾
      #   - step\d+: 匹配步骤标签，如 step1:
      #   - \s*: 匹配标签后的任意空白字符
      #   - (.*?)(?=(\nstep\d+:)|$): 非贪婪地匹配描述内容，直到下一个步骤标签或字符串结尾
      pattern = r'(?m)^step\d+:\s*(.*?)(?=(\nstep\d+:)|$)'
      matches = re.findall(pattern, sequence_explanation, re.DOTALL)
      
      result = []
      for match in matches:
          step_description = match[0].strip()
          if step_description:
              step_json = {{
                  "prompt": step_description,
                  "original prompt": action
              }}
              result.append(step_json)
      
      return result
      </code>

    </REQUIREMENT>
    
    
    
    <EXCLUSION>
    - Do not include any description of facial expressions or emotions.
    - Focus solely on the action and movement itself.
    </EXCLUSION>
    
    <END_SIGNAL>
    - When your explanation is finished, signal completion by saying '<SEQUENCEEND>'.
    </END_SIGNAL>
    
    Now think:
    """
    return prompt

def generate_sequence_explanation_prompt_useless(original_action):
    """
    Generate a prompt for the LLM to think about fine motion control for a given action.
    :param original_action: Perform a signature James Bond pose with a dramatic turn and gunpoint.
    :return:
    :example:
    The man pivots on his heel, shifting his weight to turn his body sharply while extending his opposite arm outward, creating a dramatic stance.
    The man's wrist flexes as he raises his hand to point the imaginary gun forward, aligning his arm with his shoulder for precision and balance.
    <DONE>
    """
    prompt = f"""
    The action 'original_action: {original_action}' may require detailed control over specific body parts. 
    Please evaluate the action and think carefully about how the movement breaks down into smaller steps. 
    You should independently decide the steps involved in completing this action.

    After thinking, provide a structured list of the steps involved in performing this action.

    <original_action>
    {original_action}
    </original_action>


    <REQUIREMENT>
    - Focus on describing the dynamic movement.
    - Highlight the necessary coordination between body parts.
    -Emphasize the importance of actions: Clearly require that each step must include key movement details, avoiding redundancy. 
    -Streamline the steps: Remind that the generated steps should be merged as much as possible, ensuring each step contains actual dynamic movements rather than empty descriptions.

        <FORMAT>

        The number of steps should be 1 or 2 or 3 or 4, depending on the TEMPORAL complexity of the action.Do not use too many steps if the action is simple. 2~3 steps are usually enough.  
        Do not use too many steps if the action is simple. 1 or 2 or 3 steps are usually enough.

         eg. 'run in a circle with one hand swinging',even if the action is complex by Spatial Composition, it is simple by Temporal. Apparently there is only one step of action and don't need to provide multiple steps. In this case, you can just provide one step. 

        For each step, use the words 'The man...'or 'The man's ...(body part)' to describe the action.
        Ensure the explanation is like:
        step1: The ...
        step2: The ...
        ...

        </FORMAT>
        
        注意 一定要符合格式 我会按照这个拆分：
        <code>
                  # 清理输入的 sequence_explanation
          sequence_explanation = sequence_explanation.strip()
          
          # 使用正则表达式匹配所有的步骤和对应的描述
          # 模式解释：
          #   - (?m): 多行模式，使 ^ 和 $ 匹配每一行的开头和结尾
          #   - step\d+: 匹配步骤标签，如 step1:
          #   - \s*: 匹配标签后的任意空白字符
          #   - (.*?)(?=(\nstep\d+:)|$): 非贪婪地匹配描述内容，直到下一个步骤标签或字符串结尾
          pattern = r'(?m)^step\d+:\s*(.*?)(?=(\nstep\d+:)|$)'
          matches = re.findall(pattern, sequence_explanation, re.DOTALL)
          
          result = []
          for match in matches:
              step_description = match[0].strip()
              if step_description:
                  step_json = {{
                      "prompt": step_description,
                      "original prompt": action
                  }}
                  result.append(step_json)
          
          return result
          </code>

    </REQUIREMENT>



    <EXCLUSION>
    - Do not include any description of facial expressions or emotions.
    - Focus solely on the action and movement itself.
    </EXCLUSION>

    <END_SIGNAL>
    - When your explanation is finished, signal completion by saying '<SEQUENCEEND>'.
    </END_SIGNAL>

    Now think:
    """
    return prompt

def generate_fine_motion_control_prompt(original_action, sequence_explanation):
    prompt = f"""
<TASK>
Generate fine motion control descriptions for each body part based on the action 'original_action' and the sequence_explanation.

<original_action>
{original_action}
</original_action>

<sequence_explanation>
{sequence_explanation}
</sequence_explanation>


</TASK>

<REQUIREMENT>
Focus on describing the specific actions for each body part without providing explanations or reasoning. Avoid vague terms such as "at an appropriate angle" or "adjust for balance." Instead, use precise descriptions like "Raise the right arm to shoulder height." 
Do not include any details about facial expressions, eyes, or emotions.
I don't like overly abstract language, such as "as if aiming a firearm." Understanding this action requires a high level of comprehension that general models may not possess, as this description does not clearly specify what kind of action it is. Please use more specific action descriptions.
</REQUIREMENT>

<EXAMPLES>
EXAMPLE1:
{{"body part": "left arm", "description": "The man's left arm remains stationary at his side."}}
{{"body part": "right arm", "description": "The man's right arm moves in a waving motion."}}
{{"body part": "left leg", "description": "The man's left leg is stationary."}}
{{"body part": "right leg", "description": "The man's right leg is lifted slightly off the ground."}}
{{"body part": "spine", "description": "The man's spine moves in a wave-like motion."}}

EXAMPLE2:
{{"body part": "left arm", "description": "The man's left arm is bent at the elbow and held close to his body."}}
{{"body part": "right arm", "description": "The man's right arm moves in a circular motion."}}
{{"body part": "left leg", "description": "The man's left leg is bent at the knee."}}
{{"body part": "right leg", "description": "The man's right leg is bent at the knee."}}
{{"body part": "spine", "description": "The man's spine moves in a rhythmic motion."}}

EXAMPLE3:
{{"body part": "left arm", "description": "The man's left arm is raised in a bent position."}}
{{"body part": "right arm", "description": "The man's right arm is raised and bent at the elbow."}}
{{"body part": "left leg", "description": "The man's left leg is lifted and bent at the knee."}}
{{"body part": "right leg", "description": "The man's right leg is bent at the knee and lifted slightly off the ground."}}
{{"body part": "spine", "description": "The spine is arched slightly forward."}}
</EXAMPLES>

<INPUT>
Using the action original_action and the sequence_explanation, generate fine motion control descriptions for the following body parts:

- spine
- left arm
- right arm
- left leg
- right leg

<original_action>
{original_action}
</original_action>

<sequence_explanation>
{sequence_explanation}
</sequence_explanation>

<FORMAT>
For each body part, provide a concise and specific action description in the following JSON format:

{{
    "body part": "left arm",
    "description": "The man's left arm [specific movement description]."
}},
{{
    "body part": "right arm",
    "description": "The man's right arm [specific movement description]."
}},
{{
    "body part": "left leg",
    "description": "The man's left leg [specific movement description]."
}},
{{
    "body part": "right leg",
    "description": "The man's right leg [specific movement description]."
}},
{{
    "body part": "spine",
    "description": "The man's spine [specific movement description]."
}}
</FORMAT>
</INPUT>

<END_SIGNAL>
When you finish, say '<CONTROLEND>'.
</END_SIGNAL>
"""
    return prompt

def generate_sequence_explanation_prompt_json(original_action, sequence_explanation_prompt):
    prompt = f"""
<TASK>
Based on your breakdown of the action and the most important original_action, evaluate fine motion control for the following body parts:

    <original_action>
    {original_action}
    </original_action>

    <breakdown of the action>
    {sequence_explanation_prompt}
    </breakdown of the action>  


    <BODY_PARTS>
    - spine
    - Left Arm
    - Right Arm
    - Left Leg
    - Right Leg
    </BODY_PARTS>


</TASK>



<EXAMPLES>
EXAMPLE1:
{{"body part": "left arm", "description": "The man's left arm remains stationary at his side."}}
{{"body part": "right arm", "description": "The man's right arm moves in a waving motion."}}
{{"body part": "left leg", "description": "The man's left leg is stationary."}}
{{"body part": "right leg", "description": "The man's right leg is lifted slightly off the ground."}}
{{"body part": "spine", "description": "The man's spine moves in a wave-like motion."}}

EXAMPLE2:
{{"body part": "left arm", "description": "The man's left arm is bent at the elbow and held close to his body."}}
{{"body part": "right arm", "description": "The man's right arm moves in a circular motion."}}
{{"body part": "left leg", "description": "The man's left leg is bent at the knee."}}
{{"body part": "right leg", "description": "The man's right leg is bent at the knee."}}
{{"body part": "spine", "description": "The man's spine moves in a rhythmic motion."}}

EXAMPLE3:
{{"body part": "left arm", "description": "The man's left arm is raised in a bent position."}}
{{"body part": "right arm", "description": "The man's right arm is raised and bent at the elbow."}}
{{"body part": "left leg", "description": "The man's left leg is lifted and bent at the knee."}}
{{"body part": "right leg", "description": "The man's right leg is bent at the knee and lifted slightly off the ground."}}
{{"body part": "spine", "description": "The spine is arched slightly forward."}}
</EXAMPLES>

<FORMAT>
Ensure the explanation is in the following JSON-like format for each step and body part:



{{
    "step1": [
        {{
            "body part": "left arm",
            "description": "The man's left arm [specific movement description]."
        }},
        {{
            "body part": "right arm",
            "description": "The man's right arm [specific movement description]."
        }},
        {{
            "body part": "left leg",
            "description": "The man's left leg [specific movement description]."
        }},
        {{
            "body part": "right leg",
            "description": "The man's right leg [specific movement description]."
        }},
        {{
            "body part": "spine",
            "description": "The man's spine [specific movement description]."
        }}
    ],
    "step2": [
        {{
            "body part": "left arm",
            "description": "The man's left arm [specific movement description]."
        }},
        {{
            "body part": "right arm",
            "description": "The man's right arm [specific movement description]."
        }},
        {{
            "body part": "left leg",
            "description": "The man's left leg [specific movement description]."
        }},
        {{
            "body part": "right leg",
            "description": "The man's right leg [specific movement description]."
        }},
        {{
            "body part": "spine",
            "description": "The man's spine [specific movement description]."
        }}
    ],
    ...(continue for each step)
}}

Focus on the movement and positioning of each body part, similar to the provided examples. Be concise and avoid vague terms. Use clear and specific descriptions.
</FORMAT>

<REQUIREMENT>
Focus only on these body parts. DO NOT include any details about facial expressions, eyes, or emotions.
Be concise and AVOID providing any reasoning or explanation—focus only on the action of each body part.
</REQUIREMENT>

<END_SIGNAL>
When you finish the explanation for all steps, say '<SEQUENCEEND>'.
</END_SIGNAL>
    """
    return prompt


def generate_comparison_feedback_prompt(description1, description2):
    prompt = f"""
<TASK>
You are tasked with analyzing two motion descriptions, `description1` (generated by a model) and `description2` (a gold-standard or reference description). Your goal is to carefully compare the two descriptions for each of the following body parts: spine, left arm, right arm, left leg, and right leg. For each body part, identify any discrepancies between the two descriptions and provide a single piece of feedback to help improve `description1` so it more closely matches `description2`. It is better to use standard command words from the Standard_Manual.

<BODY_PARTS>
- spine
- Left Arm
- Right Arm
- Left Leg
- Right Leg
</BODY_PARTS>

<description1>
{description1}
</description1>

<description2>
{description2}
</description2>

<Standard_Manual>
**Standard Manual for Human Motion Description**

This manual aims to standardize the language used by models when describing human poses and motions, making it more precise and specific while avoiding vague and abstract expressions. By employing a unified set of terms, phrases, and descriptive standards, it ensures that the content generated by the model is clear and consistent, making it easier for readers to understand and apply.

---

### **1. Introduction**

In the fields of artificial intelligence and computer vision, accurately describing human motion is crucial for the training and application of models. This manual provides a comprehensive and detailed set of guidelines to assist models in generating precise human motion descriptions, avoiding ambiguity and vagueness.

### **2. Basic Principles**

- **Precision**: Descriptions should be as specific as possible, providing clear angles, directions, positions, and motion details.
- **Consistency**: Use a uniform set of terms and phrases to ensure the descriptions are coherent and comprehensible.
- **Objectivity**: Avoid subjective evaluations or emotional tones, focusing on the statement of objective facts.
- **Conciseness**: Expressions should be brief and to the point, highlighting key information while avoiding redundancy.

### **3. Description Standards**

#### **3.1 Angle and Direction Descriptions**

**Angle Categories:**

- **Vertical**: Angle ≤ 10°
  - **Example**: *“The arm is raised vertically.”*
- **Near Vertical**: 10° < Angle ≤ 30°
  - **Example**: *“The thigh is nearly vertical to the ground.”*
- **Diagonal**: 30° < Angle ≤ 60°
  - **Example**: *“The arm extends at a 45-degree angle to the front side.”*
- **Near Horizontal**: 60° < Angle ≤ 80°
  - **Example**: *“The lower leg is nearly horizontal.”*
- **Horizontal**: Angle > 80°
  - **Example**: *“The arms are extended horizontally to the sides.”*

**Direction Descriptions:**

- **Front, Back, Left, Right, Up, Down**
  - **Example**: *“The head turns to the left.”*, *“The leg steps forward.”*
- **Inner, Outer**
  - **Example**: *“The palm rotates inward.”*, *“The toes point outward.”*
- **Clockwise, Counterclockwise**
  - **Example**: *“The arm rotates in a clockwise direction.”*

#### **3.2 Motion Type Descriptions**

- **Static Motion**: Maintaining a pose without movement.
  - **Example**: *“He stands still with his hands by his sides.”*
- **Continuous Motion**: Motion ongoing without interruption.
  - **Example**: *“She continues to run with a steady pace.”*
- **Repetitive Motion**: Actions that are repeated in a regular pattern.
  - **Example**: *“He jumps up and down repeatedly.”*
- **Single Motion**: An action that is completed once.
  - **Example**: *“She raises her arm and waves.”*

#### **3.3 Motion Amplitude and Degree Descriptions**

- **Complete**: The action reaches its maximum range.
  - **Example**: *“The knee is bent completely to bring the calf against the thigh.”*
- **Large Amplitude**: The action is close to its maximum range.
  - **Example**: *“He swings his arm widely.”*
- **Medium Amplitude**: The action reaches a general range.
  - **Example**: *“She raises her leg at a moderate height.”*
- **Small Amplitude**: The action has a small range.
  - **Example**: *“He slightly shakes his head.”*
- **Slightly**: The action amplitude is minimal, with little change.
  - **Example**: *“She slightly lifts her chin.”*

#### **3.4 Time and Frequency Descriptions**

- **One-time**: The action occurs only once.
  - **Example**: *“He presses the button.”*
- **Continuous**: The action is performed uninterrupted over time.
  - **Example**: *“She is continuously stretching her body.”*
- **Intermittent**: The action occurs at intervals.
  - **Example**: *“He intermittently types on the keyboard.”*
- **Frequent**: The action repeats multiple times in a short period.
  - **Example**: *“She blinks frequently.”*
- **Slow**: The action is performed at a slow pace.
  - **Example**: *“He slowly raises his arm.”*
- **Fast**: The action is performed at a fast pace.
  - **Example**: *“She quickly turns around.”*

#### **3.5 Position and Spatial Relationship Descriptions**

- **Contact**: Between body parts or with an object.
  - **Example**: *“The palm is pressed against the wall.”*
- **Distance**: Describing the space between two body parts or objects.
  - **Close**: Distance ≤ 0.2 meters
    - **Example**: *“The feet are close together.”*
  - **Apart**: 0.2 meters < Distance ≤ 0.5 meters
    - **Example**: *“The hands are shoulder-width apart.”*
  - **Far Apart**: Distance > 0.5 meters
    - **Example**: *“The arms are extended far from the body.”*
- **Height**: Relative to the ground or other reference points.
  - **Below, Equal to, Above**
    - **Example**: *“The shoulder height is equal to the table height.”*

### **4. Terms and Definitions**

#### **4.1 Body Part Names**

```
<BODY_PARTS>
- spine
- Left Arm
- Right Arm
- Left Leg
- Right Leg
</BODY_PARTS>
```

#### **4.2 Action Verbs**

- **Basic Actions**: Raise, lower, bend, extend, rotate, kick, grasp, push, pull, jump, squat, stand, sit, lie
- **Compound Actions**: Step over, squat down, flip forward, flip backward, kick leg, stretch, twist

#### **4.3 Directional Terms**

- **Horizontal Directions**: Front, back, left, right
- **Vertical Directions**: Up, down
- **Spatial Relationships**: Inner, outer, front, back, side, opposite

#### **4.4 Angle and Amplitude Terms**

- **Angle Unit**: Degrees (°)
- **Angle Descriptions**: Vertical, horizontal, diagonal, right angle (90°)
- **Amplitude Descriptions**: Complete, large amplitude, medium amplitude, small amplitude, slight



### **7. Notes**

#### **7.1 Avoid Vague and Subjective Descriptions**

- **Non-standard**: *“He moves in a strange way.”*
- **Standard**: *“He crosses his left foot over his right foot and leans to the right.”*

#### **7.2 Ensure Consistency and Precision**

- Always use the terms and standards outlined in this manual, avoiding different terms for the same entity.



### **8. Appendix**

#### **8.1 Common Angle and Distance Reference Table**

| Angle Description | Range     |
| ----------------- | --------- |
| Vertical          | 0° - 10°  |
| Near Vertical     | 10° - 30° |
| Diagonal          | 30° - 60° |
| Near Horizontal   | 60° - 80° |
| Horizontal        | 80° - 90° |

| Distance Description | Range                   |
| -------------------- | ----------------------- |
| Close                | ≤ 0.2 meters            |
| Apart                | 0.2 meters - 0.5 meters |
| Far Apart            | > 0.5 meters            |



</Standard_Manual>

<INSTRUCTIONS>

Editing command: to help improve `description1` so it more closely matches `description2`.
1. **Comparison**:
    - Compare the descriptions for each body part mentioned above. Focus on specific aspects of the motion, such as:
        - **Movement**: How the body part is moving (e.g., raised, lowered, rotated).
        - **Direction**: In which direction the movement occurs (e.g., upward, downward, forward, backward).
        - **Intensity**: Is the movement fast or slow? Sudden or smooth?
        - **Positioning**: The final position of the body part (e.g., shoulder height, bent at the knee).
    - If a body part is mentioned in `description2` but not in `description1`, note that as a discrepancy.

2. **Feedback**:
    - For each body part, if a discrepancy is found, generate **one** piece of feedback that includes:
        - **Body part**: Identify which body part the feedback is for (e.g., left arm, right leg, spine).
        - **Issue**: Briefly describe the difference between `description1` and `description2` for this body part.
        - **Suggested Change**: Provide a clear, actionable suggestion to adjust `description1` to match `description2`. Be specific about how the movement should be changed.
    - Ensure that only one piece of feedback is given for each body part, even if there are multiple issues. Prioritize the most significant discrepancy.

3. **Output**:
    - Format the feedback as follows:
    {{
        "body part": "spine",
        "issue": "[description of discrepancy in spine movement]",
        "editing command": "The man..../ The man's spine ... [specific suggestion to correct the spine movement,should use Standard_Manual style of words]"
    }},
    {{
        "body part": "left arm",
        "issue": "[description of discrepancy in left arm movement]",
        "suggestion": "The man..../ The man's left arm ... [specific suggestion to correct the left arm movement,should use Standard_Manual style of words]"
    }},
    {{
        "body part": "right arm",
        "issue": "[description of discrepancy in right arm movement]",
        "suggestion": "The man..../ The man's right arm ... [specific suggestion to correct the right arm movement,should use Standard_Manual style of words]"
    }},
    {{
        "body part": "left leg",
        "issue": "[description of discrepancy in left leg movement]",
        "suggestion": "The man..../ The man's left leg ... [specific suggestion to correct the left leg movement,should use Standard_Manual style of words]"
    }},
    {{
        "body part": "right leg",
        "issue": "[description of discrepancy in right leg movement]",
        "suggestion": "The man..../ The man's right leg ... [specific suggestion to correct the right leg movement,should use Standard_Manual style of words]"
    }}

4. **Exclusions**:
    - Do not analyze facial expressions, emotions, or any abstract descriptions.
    - Focus only on the physical actions of the body parts mentioned in `<BODY_PARTS>`.

<END_SIGNAL>
When you have finished generating the feedback, end the output with '<FEEDBACKEND>'.
</END_SIGNAL>
</INSTRUCTIONS>

Now, compare the two descriptions and generate feedback.
    """
    return prompt

def decide_which_part_needs_editing_prompt(description1, description2):
    prompt = f"""
<TASK>
You are tasked with determining which parts of the body need editing in `description1` (generated by a model) compared to `description2` (a gold-standard or reference description). Your goal is to carefully compare the two descriptions for each of the following body parts: spine, left arm, right arm, left leg, and right leg.

For each body part, identify discrepancies between the two descriptions and provide a boolean value indicating whether `description1` needs editing for that body part to match `description2`.

<BODY_PARTS>
- Spine
- Left Arm
- Right Arm
- Left Leg
- Right Leg
</BODY_PARTS>

<description1>
{description1}
</description1>

<description2>
{description2}
</description2>

<OUTPUT>
{{
  "action_description": "...",
  "gold_standard": "...",
  "output": {{
  "Spine": true/false,
  "Left_Arm": true/false,
  "Right_Arm": true/false,
  "Left_Leg": true/false,
  "Right_Leg": true/false,
    }}
}}

</OUTPUT>
 

Carefully judge whether the action itself needs modification, rather than comparing the similarity of descriptions. For example:

**Description 1**:  
The person performs a rowing motion with their legs spread wide.

**Description 2**:  
The man sits on a stable surface with his legs spread wide apart and engages his core while leaning forward slightly. After that, the man's arms pull back in a rowing motion while simultaneously pushing his legs outward, resulting in a coordinated back and leg movement that mimics the action of rowing.

These two descriptions refer to the same action; the second one is simply more detailed, so no modification is needed, and each part should be marked as 'false'.
 This example gives us an insight: if one description is a subset of another, then that part does not need to be modified; otherwise, it requires modification.

Modification is necessary only when there are differences between Description 1 and Description 2. For example, if Description 1 requires raising the right hand while Description 2 specifies swinging the right hand, then there is a conflict between the two actions, and modification is needed.

<example>
    description1 = 
    A person walks in a  circle, while shaking his right hand throughout the motion.

    description2 =  
    The man is walking clockwise in a circle while holding right hand up to his ear.


    <OUTPUT>
{{
  "action_description": "A person walks in a clockwise circle, while holding their right hand & arm up throughout the motion.",
  "gold_standard": "The man is walking clockwise in a circle while holding something up to his ear with his left arm.",
  "output": {{
    "Spine": false,
    "Left Arm": false,
    "Right Arm": true,
    "Left Leg": false,
    "Right Leg": false,
  }}
}}

    </OUTPUT>
</example>

Strictly output in JSON format. No additional explanations or elaborations are allowed.

Specifically, after "Right_Leg": false/true, a comma is expected before closing the curly braces (}}). This issue is causing the JSON decoding process to fail. MAKE sure the json is correct ,I don't want to see sth like"JSONDecodeError: Expecting ',' delimiter: line 10 column 4 (char 378)
Comparison Feedback Results saved to "


<END_SIGNAL>
When you have finished generating the feedback, end the output with '<FEEDBACKEND>'.
</END_SIGNAL>

"""
    return prompt


