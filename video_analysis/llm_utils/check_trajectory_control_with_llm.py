



from llm_config  import llm

def check_trajectory_control_with_llm(action):
    # First call: Let the LLM think using chain of thought reasoning
    first_prompt = f"""
    Examine the action: "{action}".

    Trajectory control is only necessary if the action includes explicit instructions for movement along a specific two-dimensional path. This typically includes instructions like "turn", "move in a circle", "zigzag", or "change direction", which clearly involve a shift in direction or orientation.

    If an action involves simple movement in a straight line (such as "run forward" or "jump in place") or basic rotations (like "spin in place"), trajectory control is not required because there is no complex or changing path to follow.

    Consider whether the action explicitly requires a change in direction or follows a curved or angular path, then conclude if trajectory control is necessary. If the path is simple or linear, trajectory control is not needed.

    Explain your reasoning step by step and conclude with whether trajectory control is required.
    """

    first_response = llm(first_prompt)

    # Second call: Ask LLM to give a simple boolean response based on its reasoning
    second_prompt = f"""
Based on your previous analysis: "{first_response}", answer with only 'True' if trajectory control is required, and 'False' if it is not.
"""
    second_response = llm(second_prompt).strip()

    return second_response


# Convert the second response to a boolean value
def response_to_bool(response):
    if response.lower() == "true":
        return True
    elif response.lower() == "false":
        return False
    else:
        raise ValueError("Unexpected response from LLM: " + response)
