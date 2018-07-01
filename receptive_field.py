def f(output_size, ksize, stride):
    return (output_size - 1) * stride + ksize


#CYCLEGAN
last_layer = f(output_size=1, ksize=4, stride=1)
# Receptive field: 4
fourth_layer = f(output_size=last_layer, ksize=4, stride=1)
# Receptive field: 7
third_layer = f(output_size=fourth_layer, ksize=4, stride=2)
# Receptive field: 16
second_layer = f(output_size=third_layer, ksize=4, stride=2)
# Receptive field: 34
first_layer = f(output_size=second_layer, ksize=4, stride=2)
# Receptive field: 70
print("RECEPTIVE FIELD FOR CYCLEGAN: ",first_layer)


