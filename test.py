import tensorflow as tf

new_list=[145,56,89,56]
print(type(new_list))
con_lis = tf.convert_to_tensor(new_list)
print("Convert list to tensor:",con_lis)