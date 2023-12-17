

# CUDA_VISIBLE_DEVICES=1 python scripts/train_slot.py\
#         train_dataset.tasks=[close_jar,insert_onto_square_peg]\
#         val_dataset.tasks=[close_jar,insert_onto_square_peg]



# CUDA_VISIBLE_DEVICES=0 python scripts/train_slot.py\
#         slot_model=slot_autoencoder_isa

# CUDA_VISIBLE_DEVICES=0 python scripts/train_slot.py\
#         slot_model=invariant_slot_autoencoder

# CUDA_VISIBLE_DEVICES=0 python scripts/train_slot.py\
#         slot_model=slot_autoencoder\
#         train_dataset.views=[front_rgb,left_shoulder_rgb,overhead_rgb,right_shoulder_rgb,wrist_rgb]

# CUDA_VISIBLE_DEVICES=0 python scripts/train_slot.py\
#         slot_model=slot_autoencoder\
#         train_dataset.views=[front_rgb,left_shoulder_rgb,overhead_rgb,right_shoulder_rgb,wrist_rgb]\
#         train_dataset.tasks=[close_jar,insert_onto_square_peg,light_bulb_in,meat_off_grill,open_drawer]

# CUDA_VISIBLE_DEVICES=0 python scripts/train_slot.py\
#         slot_model=resnet34_slot_autoencoder\
#         train_dataset.tasks=[close_jar,insert_onto_square_peg,light_bulb_in,meat_off_grill,open_drawer,place_cups,place_shape_in_shape_sorter,place_wine_at_rack_location,push_buttons,put_groceries_in_cupboard,put_item_in_drawer,put_money_in_safe,reach_and_drag,slide_block_to_color_target,stack_blocks,stack_cups,sweep_to_dustpan_of_size,turn_tap]

CUDA_VISIBLE_DEVICES=0 python scripts/train_slot.py\
        --config-name train_slot_real\
        slot_model=resnet34_slot_autoencoder\
        slot_model.slot_attention.num_slots=3\
        slot_model.decoder.num_slots=3