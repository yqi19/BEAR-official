from main import run_agent

# run a example for vision tasks. save the execution trace to outputs/z_agent/pointing/general_object_pointing/entry_26

run_agent("../tasks/z_agent/trajectory/gripper_trajectory/entry_6", "../outputs/z_agent/trajectory/gripper_trajectory", task_type="trajectory")
# run_agent("../tasks/z_agent/pointing/semantic_part_pointing/entry_32", "../outputs/z_agent/pointing/semantic_part_pointing", task_type="vision")
# run_agent("../tasks/z_agent/pointing/spatial_relationship_pointing/entry_26", "../outputs/z_agent/pointing/spatial_relationship_pointing", task_type="vision")
# run_agent("../tasks/z_agent/pointing/spatial_relationship_pointing/entry_163", "../outputs/z_agent/pointing/spatial_relationship_pointing", task_type="vision")