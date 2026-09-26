import numpy as np

POINTING_PROMPT_DIRECT = (
    "Please directly output the pixel location of the target point.\n"
    "Format your answer as (x, y), where:\n"
    "- x is the horizontal coordinate (left → right),\n"
    "- y is the vertical coordinate (top → bottom).\n"
    "Both x and y must be normalized to the range [0, 1] as floating-point values, "
    "indicating the relative position of the point within the image."
)

POINTING_PROMPT_COT = (
   "Please reason step by step to locate the target point, and then output the final pixel location.\n"
    "Format your answer as (x, y), where:\n"
    "- x is the horizontal coordinate (left → right),\n"
    "- y is the vertical coordinate (top → bottom).\n"
    "Both x and y must be normalized to the range [0, 1] as floating-point values, "
    "indicating the relative position of the point within the image."
)

BBOX_PROMPT_DIRECT = (
    "Please directly output the bounding box.\n"
    "Format your answer as (x1, y1, x2, y2), where:\n"
    "- (x1, y1) is the top-left corner of the bounding box,\n"
    "- (x2, y2) is the bottom-right corner of the bounding box.\n"
    "- x represents the horizontal coordinate (left → right),\n"
    "- y represents the vertical coordinate (top → bottom).\n"
    "All coordinates must be normalized to the range [0, 1] as floating-point values, "
    "indicating the relative bounding box location within the image."
)

BBOX_PROMPT_COT = (
    "Please reason step by step and then output the bounding box.\n"
    "Format your answer as (x1, y1, x2, y2), where:\n"
    "- (x1, y1) is the top-left corner of the bounding box,\n"
    "- (x2, y2) is the bottom-right corner of the bounding box.\n"
    "- x represents the horizontal coordinate (left → right),\n"
    "- y represents the vertical coordinate (top → bottom).\n"
    "All coordinates must be normalized to the range [0, 1] as floating-point values, "
    "indicating the relative bounding box location within the image."
)

TRAJECTORY_PROMPT_DIRECT = (
    "Please directly provide the correct option."
)

TRAJECTORY_PROMPT_COT = (
    "Please reason step by step and provide the correct option at the end."
)

SPATIAL_PROMPT_DIRECT = (
    "Please directly provide the correct option."
)

SPATIAL_PROMPT_COT = (
    "Please reason step by step and provide the correct option at the end."
)

# category分为哪些：
# pointing, bbox, trajectory
# object localization, relative direction, path planning, 
# action prediction, history action reasoning,

# pointing: identify the ...
# bbox: identify the ...
# trajectory: 问题都要是很全很详细的，包括具体的embodiment， 图里有三个选项等等，哪一个。。。
# object localization: which one is correct about the location of the ...
# relative direction: which one is correct about the relative direction of the ...
# path planning: which one is correct about the path from the current observation to ...

# model category = "general, image, video"
def generate_question_prompt(format="direct", category="pointing", original_question="", original_options="", model_category="general"):
    if format == "direct" and category == "pointing":
        prompt = (
        f"The question is: {original_question}\n"
        f"{POINTING_PROMPT_DIRECT}"
    )
    elif format == "cot" and category == "pointing":
        prompt = (
            f"The question is: {original_question}\n"
            f"{POINTING_PROMPT_COT}"
        )
    elif format == "direct" and category == "bbox":
        prompt = (
            f"The question is: {original_question}\n"
            f"{BBOX_PROMPT_DIRECT}"
        )
    elif format == "cot" and category == "bbox":
        prompt = (
            f"The question is: {original_question}\n"
            f"{BBOX_PROMPT_COT}"
        )
    elif format == "direct" and category == "trajectory":
        # in trajectory understanding, formulate your answer within the original_question
        # options is a dict
        option_A = original_options["A"] if original_options["A"].strip() else "N/A"
        option_B = original_options["B"] if original_options["B"].strip() else "N/A"
        option_C = original_options["C"] if original_options["C"].strip() else "N/A"
        option_D = original_options["D"] if original_options["D"].strip() else "N/A"

        # handle the four different options
        prompt = (
        f"{original_question}\n\n"
        f"Options:\n"
        f"A: {option_A}\n"
        f"B: {option_B}\n"
        f"C: {option_C}\n"
        f"D: {option_D}\n\n"
        f"{TRAJECTORY_PROMPT_DIRECT}\n"
        )
    elif format == 'cot' and category == "trajectory":
        option_A = original_options["A"] if original_options["A"].strip() else "N/A"
        option_B = original_options["B"] if original_options["B"].strip() else "N/A"
        option_C = original_options["C"] if original_options["C"].strip() else "N/A"
        option_D = original_options["D"] if original_options["D"].strip() else "N/A"

        prompt = (
            f"{original_question}\n\n"
            f"Options:\n"
            f"A: {option_A}\n"
            f"B: {option_B}\n"
            f"C: {option_C}\n"
            f"D: {option_D}\n\n"
            f"{TRAJECTORY_PROMPT_COT}\n"
        )

    # object localization的输入是一段video
    # 如果是generalVLM的话 ...
    # 如果是imageVLM的话 ...
    elif format == "direct" and category == "object localization":
        if model_category == "general":
            option_A = original_options["A"] if original_options["A"].strip() else "N/A"
            option_B = original_options["B"] if original_options["B"].strip() else "N/A"
            option_C = original_options["C"] if original_options["C"].strip() else "N/A"
            option_D = original_options["D"] if original_options["D"].strip() else "N/A"

            first_prompt = "Please watch the following video and answer the question.\n"
            second_prompt = (f"{original_question}\n"
            f"Options:\n"
            f"A: {option_A}\n"
            f"B: {option_B}\n"
            f"C: {option_C}\n"
            f"D: {option_D}\n\n"
            f"{SPATIAL_PROMPT_DIRECT}\n")
        
            # first_prompt之后是视频的内容
            prompt = (first_prompt, second_prompt)
        elif model_category == "image":
            option_A = original_options["A"] if original_options["A"].strip() else "N/A"
            option_B = original_options["B"] if original_options["B"].strip() else "N/A"
            option_C = original_options["C"] if original_options["C"].strip() else "N/A"
            option_D = original_options["D"] if original_options["D"].strip() else "N/A"

            first_prompt = (
            "The image you are given is a merged visual summary of a video.\n"
            "It contains multiple frames stitched together in a single image to represent the temporal progression.\n"
            )

            second_prompt = (f"{original_question}\n"
             f"Options:\n"
             f"A: {option_A}\n"
             f"B: {option_B}\n"
             f"C: {option_C}\n"
             f"D: {option_D}\n\n"
             f"{SPATIAL_PROMPT_DIRECT}\n")
        
            prompt = first_prompt + second_prompt
    elif format == "cot" and category == "object localization":
        if model_category == "general":
            option_A = original_options["A"] if original_options["A"].strip() else "N/A"
            option_B = original_options["B"] if original_options["B"].strip() else "N/A"
            option_C = original_options["C"] if original_options["C"].strip() else "N/A"
            option_D = original_options["D"] if original_options["D"].strip() else "N/A"

            first_prompt = "Please watch the following video and answer the question.\n"
            second_prompt = (f"{original_question}\n"
            f"Options:\n"
            f"A: {option_A}\n"
            f"B: {option_B}\n"
            f"C: {option_C}\n"
            f"D: {option_D}\n\n"
            f"{SPATIAL_PROMPT_COT}\n")
        
            prompt = (first_prompt, second_prompt)

        elif model_category == "image":
            option_A = original_options["A"] if original_options["A"].strip() else "N/A"
            option_B = original_options["B"] if original_options["B"].strip() else "N/A"
            option_C = original_options["C"] if original_options["C"].strip() else "N/A"
            option_D = original_options["D"] if original_options["D"].strip() else "N/A"

            first_prompt = (
            "The image you are given is a merged visual summary of a video.\n"
            "It contains multiple frames stitched together in a single image to represent the temporal progression.\n"
            )

            second_prompt = (f"{original_question}\n"
             f"Options:\n"
             f"A: {option_A}\n"
             f"B: {option_B}\n"
             f"C: {option_C}\n"
             f"D: {option_D}\n\n"
             f"{SPATIAL_PROMPT_COT}\n")
        
            prompt = first_prompt + second_prompt

    # relative direction 输入是一段video和一个image(当前frame的observation)
    # 如果是generalVLM的话 ...
    # 如果是imageVLM的话 ...
    elif category == "relative direction":
        if model_category == "general":
            option_A = original_options["A"] if original_options["A"].strip() else "N/A"
            option_B = original_options["B"] if original_options["B"].strip() else "N/A"
            option_C = original_options["C"] if original_options["C"].strip() else "N/A"
            option_D = original_options["D"] if original_options["D"].strip() else "N/A"

            first_prompt = "This is the history video in which agent is navigating in the room\n"
            second_prompt = "This is the current observation representing the agent's view in the room\n"

            if format == "direct":
                cot_prompt = SPATIAL_PROMPT_DIRECT
            else:
                cot_prompt = SPATIAL_PROMPT_COT

            third_prompt = (f"{original_question}\n"
            f"Options:\n"
            f"A: {option_A}\n"
            f"B: {option_B}\n"
            f"C: {option_C}\n"
            f"D: {option_D}\n\n"
            f"{cot_prompt}\n")
        
            prompt = (first_prompt, second_prompt, third_prompt)

        elif model_category == "image":
            option_A = original_options["A"] if original_options["A"].strip() else "N/A"
            option_B = original_options["B"] if original_options["B"].strip() else "N/A"
            option_C = original_options["C"] if original_options["C"].strip() else "N/A"
            option_D = original_options["D"] if original_options["D"].strip() else "N/A"
            
            first_prompt = (
                "The image shown is a composite created by merging multiple frames into a single image.\n"
                "All frames except the last one represent the video history.\n"
                "The final frame, located in the bottom-right corner, shows the current observation."
            )

            if format == "direct":
                cot_prompt = SPATIAL_PROMPT_DIRECT
            else:
                cot_prompt = SPATIAL_PROMPT_COT

            second_prompt = (f"{original_question}\n"
             f"Options:\n"
             f"A: {option_A}\n"
             f"B: {option_B}\n"
             f"C: {option_C}\n"
             f"D: {option_D}\n\n"
             f"{cot_prompt}\n")

            prompt = first_prompt + second_prompt
      
    elif category == "path planning":
        if model_category == "general":
            option_A = original_options["A"] if original_options["A"].strip() else "N/A"
            option_B = original_options["B"] if original_options["B"].strip() else "N/A"
            option_C = original_options["C"] if original_options["C"].strip() else "N/A"
            option_D = original_options["D"] if original_options["D"].strip() else "N/A"

            first_prompt = "This is the current video\n"

            second_prompt = "This is the current observation\n"

            if format == "direct":
                cot_prompt = SPATIAL_PROMPT_DIRECT
            else:
                cot_prompt = SPATIAL_PROMPT_COT

            third_prompt = (f"{original_question}\n"
            f"Options:\n"
            f"A: {option_A}\n"
            f"B: {option_B}\n"
            f"C: {option_C}\n"
            f"D: {option_D}\n\n"
            f"{cot_prompt}\n")
        
            prompt = (first_prompt, second_prompt, third_prompt)

        elif model_category == "image":
            option_A = original_options["A"] if original_options["A"].strip() else "N/A"
            option_B = original_options["B"] if original_options["B"].strip() else "N/A"
            option_C = original_options["C"] if original_options["C"].strip() else "N/A"
            option_D = original_options["D"] if original_options["D"].strip() else "N/A"

            first_prompt = (
                "The image shown is a composite created by merging multiple frames into a single image.\n"
                "All frames except the last one represent the video history.\n"
                "The final frame, located in the bottom-right corner, shows the current observation."
            )

            if format == "direct":
                cot_prompt = SPATIAL_PROMPT_DIRECT
            else:
                cot_prompt = SPATIAL_PROMPT_COT

            second_prompt = (f"{original_question}\n"
             f"Options:\n"
             f"A: {option_A}\n"
             f"B: {option_B}\n"
             f"C: {option_C}\n"
             f"D: {option_D}\n\n"
             f"{cot_prompt}\n")

            prompt = first_prompt + second_prompt

    # next action prediction 输入是一段video和一个image(当前frame的observation)
    # 如果是generalVLM的话 ...
    # 如果是imageVLM的话 ...
    elif category == "next action prediction":
        if model_category == "general":
            option_A = original_options["A"] if original_options["A"].strip() else "N/A"
            option_B = original_options["B"] if original_options["B"].strip() else "N/A"
            option_C = original_options["C"] if original_options["C"].strip() else "N/A"
            option_D = original_options["D"] if original_options["D"].strip() else "N/A"

            # first_prompt = "This is the current activity video\n"
            first_prompt = "This is the history activity video\n"

            if format == "direct":
                cot_prompt = SPATIAL_PROMPT_DIRECT
            else:
                cot_prompt = SPATIAL_PROMPT_COT

            second_prompt = (f"{original_question}\n"
            f"Options:\n"
            f"A: {option_A}\n"
            f"B: {option_B}\n"
            f"C: {option_C}\n"
            f"D: {option_D}\n\n"
            f"{cot_prompt}\n")
        
            prompt = (first_prompt, second_prompt)

        elif model_category == "image":
            option_A = original_options["A"] if original_options["A"].strip() else "N/A"
            option_B = original_options["B"] if original_options["B"].strip() else "N/A"
            option_C = original_options["C"] if original_options["C"].strip() else "N/A"
            option_D = original_options["D"] if original_options["D"].strip() else "N/A"

            first_prompt = (
                "The image shown is a composite created by merging multiple frames into a single image.\n"
                "All frames represent the video history.\n"
            )

            if format == "direct":
                cot_prompt = SPATIAL_PROMPT_DIRECT
            else:
                cot_prompt = SPATIAL_PROMPT_COT

            second_prompt = (f"{original_question}\n"
             f"Options:\n"
             f"A: {option_A}\n"
             f"B: {option_B}\n"
             f"C: {option_C}\n"
             f"D: {option_D}\n\n"
             f"{cot_prompt}\n")

            prompt = first_prompt + second_prompt

    
    # task progress reasoning
    # if this is generalVLM ...
    # if this is imageVLM ...
    elif category == "task progress reasoning":
        if model_category == "general":
            option_A = original_options["A"] if original_options["A"].strip() else "N/A"
            option_B = original_options["B"] if original_options["B"].strip() else "N/A"
            option_C = original_options["C"] if original_options["C"].strip() else "N/A"
            option_D = original_options["D"] if original_options["D"].strip() else "N/A"

            first_prompt = "This is the history activity video\n"

            if format == "direct":
                cot_prompt = SPATIAL_PROMPT_DIRECT
            else:
                cot_prompt = SPATIAL_PROMPT_COT

            second_prompt = (f"{original_question}\n"
            f"Options:\n"
            f"A: {option_A}\n"
            f"B: {option_B}\n"
            f"C: {option_C}\n"
            f"D: {option_D}\n\n"
            f"{cot_prompt}\n")
        
            prompt = (first_prompt, second_prompt)

        elif model_category == "image":
            option_A = original_options["A"] if original_options["A"].strip() else "N/A"
            option_B = original_options["B"] if original_options["B"].strip() else "N/A"
            option_C = original_options["C"] if original_options["C"].strip() else "N/A"
            option_D = original_options["D"] if original_options["D"].strip() else "N/A"

            first_prompt = (
                "The image shown is a composite created by merging multiple frames into a single image.\n"
                "All frames represent the video history.\n"
            )

            if format == "direct":
                cot_prompt = SPATIAL_PROMPT_DIRECT
            else:
                cot_prompt = SPATIAL_PROMPT_COT

            second_prompt = (f"{original_question}\n"
             f"Options:\n"
             f"A: {option_A}\n"
             f"B: {option_B}\n"
             f"C: {option_C}\n"
             f"D: {option_D}\n\n"
             f"{cot_prompt}\n")

            prompt = first_prompt + second_prompt

    # turn this to red
    print("\033[91mGenerated prompt:\033[0m", prompt)
    return prompt