CAPTION_DIR="./data/base_complex/caption"
VID_DIR="./data/base_complex/video"
PROMPT_DIR="./data/base_complex/prompt"
INSTRUCTION_DIR="./data/base_complex/edit_instruction"
DEVICE="cuda:5"


python caption.py --caption_dir "$CAPTION_DIR" --vid_dir "$VID_DIR" --device "$DEVICE"

export OPENAI_API_KEY="your OPENAI_API_KEY"

python edit_suggestion.py --caption_dir "$CAPTION_DIR" --prompt_dir "$PROMPT_DIR" --instruction_dir "$INSTRUCTION_DIR"