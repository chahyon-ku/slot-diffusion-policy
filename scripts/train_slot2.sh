

# CUDA_VISIBLE_DEVICES=1 python scripts/train_slot.py\
#         train_dataset.tasks=[close_jar,insert_onto_square_peg]\
#         val_dataset.tasks=[close_jar,insert_onto_square_peg]



CUDA_VISIBLE_DEVICES=1 python scripts/train_slot.py\
        slot_model=slot_autoencoder_isa\
        train_dataset.tasks=[close_jar,insert_onto_square_peg,light_bulb_in,meat_off_grill,open_drawer]