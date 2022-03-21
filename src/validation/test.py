label_map = {0: 1, 1: 0}
test = [0,0,1,1]

print(label_map[0])

my_function = lambda x: label_map[x]

for i, val in enumerate(test): 
    test[i] = label_map[val]

print(test)
# print(my_function(val) for val in test)