
import json
import re

from llm_utils import generate_comparison_feedback_prompt
from llm_utils import decide_which_part_needs_editing_prompt
from .llm_config  import llm

def decide_which_part_needs_editing(description1, description2):
    # Generate comparison feedback prompt
    comparison_prompt = decide_which_part_needs_editing_prompt(description1, description2)

    # Get feedback from LLM and remove "<FEEDBACKEND>" tag
    comparison_feedback = llm(comparison_prompt, stop=["<END_FEEDBACK>"]).replace("<FEEDBACKEND>", "").strip()

    print(f"Raw feedback: {comparison_feedback}\n")

    # Initialize an empty list to store valid JSON objects
    feedback_results = []

    # Extract the JSON content from feedback
    json_start = comparison_feedback.find('{')
    json_end = comparison_feedback.rfind('}') + 1  # Include the closing brace

    if json_start != -1 and json_end != -1:
        json_str = comparison_feedback[json_start:json_end]
        print(f"Extracted JSON string: {json_str}\n")
        try:
            parsed_obj = json.loads(json_str)
            feedback_results.append(parsed_obj)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    else:
        print("No JSON content found in the feedback.")

    # Save the results to a JSON file
    output_filename = "comparison_feedback_results.json"
    with open(output_filename, "w") as file:
        json.dump(feedback_results, file, ensure_ascii=False, indent=2)

    # Print output
    print(f"Comparison Feedback Results saved to {output_filename}")
    print(json.dumps(feedback_results, ensure_ascii=False, indent=2))

    # Return the generated feedback results
    return feedback_results



def test_generate_comparison_feedback(description1, description2):
    # Generate comparison feedback prompt
    comparison_prompt = generate_comparison_feedback_prompt(description1, description2)

    # Get feedback from LLM and remove "<FEEDBACKEND>" tag
    comparison_feedback = llm(comparison_prompt, stop=["<END_FEEDBACK>"]).replace("<FEEDBACKEND>", "").strip()

    # Check if the feedback is valid JSON
    try:
        # Convert feedback directly to a JSON object
        feedback_results = json.loads(comparison_feedback)
    except json.JSONDecodeError:
        # If not directly valid JSON, try to find and extract JSON-like objects using regex
        json_objects = re.findall(r'\{[^}]+\}', comparison_feedback)
        feedback_results = [json.loads(obj) for obj in json_objects]

    # Save the results to a JSON file
    output_filename = "comparison_feedback_results.json"
    with open(output_filename, "w") as file:
        json.dump(feedback_results, file, ensure_ascii=False, indent=2)

    # Print output
    print(f"Comparison Feedback Results saved to {output_filename}")
    print(json.dumps(feedback_results, ensure_ascii=False, indent=2))

    # Return the generated feedback results
    return feedback_results


