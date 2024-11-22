# Step 1
Prepare the dataset

'''bash
sh 1_create_dataset.sh
'''


# Step 2
Train the model

'''bash
sh 2_train_kl.sh
''' 

# Step 3
Convert the checkpoint to FP32

'''bash
sh 3_create_ckpt.sh
'''