file_1 = 'R_cor_img.txt'
file_2 = 'OR_fail_img.txt'

list_1 = []
list_2 = []

with open(file_1) as f:
    list_1 = f.read().split()

with open(file_2) as f:
    list_2 = f.read().split()

# print(len(list_1))
# print(len(list_2))

def intersect(a, b):
    return list(set(a) & set(b))

# print(intersect(list_1, list_2))
# print(len(intersect(list_1, list_2)))

img_file = open('img_list.txt', 'w')

intersect_list = intersect(list_1, list_2)
intersect_list = list(map(int, intersect_list))
intersect_list.sort()

for img_id in intersect_list:
    img_file.write("{} {}\n".format(img_id, 0))