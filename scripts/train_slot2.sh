

# CUDA_VISIBLE_DEVICES=1 python scripts/train_slot.py\
#         train_dataset.tasks=[close_jar,insert_onto_square_peg]\
#         val_dataset.tasks=[close_jar,insert_onto_square_peg]



CUDA_VISIBLE_DEVICES=0 python scripts/train_slot.py\
        train_dataset.is_pairs=True\
        slot_model=slot_transport\
        train.f_eval=5000\
        slot_model.num_slots=16