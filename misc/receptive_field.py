def f(im_size, ksize, stride, padding):
    return ((im_size + (2*padding) - ksize)/ float(stride)) + 1.


#CYCLEGAN
# last_layer = f(output_size=1, ksize=4, stride=1)
# # Receptive field: 4
# fourth_layer = f(output_size=last_layer, ksize=4, stride=1)
# # Receptive field: 7
# third_layer = f(output_size=fourth_layer, ksize=4, stride=2)
# # Receptive field: 16
# second_layer = f(output_size=third_layer, ksize=4, stride=2)
# # Receptive field: 34
# first_layer = f(output_size=second_layer, ksize=4, stride=2)
# # Receptive field: 70
# print("RECEPTIVE FIELD FOR CYCLEGAN: ",first_layer)


f5 = f(im_size=128, ksize=4, stride=2, padding=1)
f4 = f(im_size=f5, ksize=4, stride=2, padding=1)
f3 = f(im_size=f4, ksize=4, stride=2, padding=1)
f2 = f(im_size=f3, ksize=4, stride=2, padding=1)
f1 = f(im_size=f2, ksize=4, stride=2, padding=1)
f0 = f(im_size=f1, ksize=4, stride=2, padding=1)
ll = f(im_size=f0, ksize=3, stride=1, padding=1)
print("Last size Discriminator layer FOR STARTGAN: ",ll)


def f(output_size, ksize, stride, padding):
    return ((output_size - 1) * stride) + ksize - (2*padding)

ll = f(output_size=2, ksize=3, stride=1, padding=1)
f1 = f(output_size=ll, ksize=4, stride=2, padding=1)
f0 = f(output_size=f0, ksize=4, stride=2, padding=1)
f0 = f(output_size=f0, ksize=4, stride=2, padding=1)
f0 = f(output_size=f0, ksize=4, stride=2, padding=1)
f0 = f(output_size=f0, ksize=4, stride=2, padding=1)
f0 = f(output_size=f0, ksize=4, stride=2, padding=1)
print("RECEPTIVE FIELD FOR STARTGAN: ",f0)