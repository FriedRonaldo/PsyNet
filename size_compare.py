file_1 = 'R_fail_size.txt'
file_2 = 'OR_cor_size.txt'

list_1 = []
size_1 = []
list_2 = []
size_2 = []

R_s = []
R_l = []

with open(file_1) as f:
    for line in f:
        l, s = line.strip().split()
        list_1.append(l)
        size_1.append(int(s))

with open(file_2) as f:
    for line in f:
        l, s = line.strip().split()
        list_2.append(l)
        size_2.append(int(s))

small = open('R_small.txt', 'w')
large = open('R_large.txt', 'w')

for idx, id in enumerate(list_1):
    if size_1[idx] < size_2[idx]:
        R_s.append(id)
    elif size_1[idx] > size_2[idx]:
        R_l.append(id)

for img_id in R_s:
    small.write("{} {}\n".format(img_id, 0))
for img_id in R_l:
    large.write("{} {}\n".format(img_id, 0))