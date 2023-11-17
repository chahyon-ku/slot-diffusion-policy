

# CUDA_VISIBLE_DEVICES=1 python scripts/train_slot.py\
#         train_dataset.tasks=[close_jar,insert_onto_square_peg]\
#         val_dataset.tasks=[close_jar,insert_onto_square_peg]



CUDA_VISIBLE_DEVICES=0 python scripts/train_slot.py\
        train_dataset.is_pairs=False\
        slot_model=invariant_slot_attention\
        train.f_eval=1000\