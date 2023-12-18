import numpy as np
from sklearn.metrics import f1_score

def calc_3class_f1(y_true, y_pred):
    # calculate f1 score for 3 class
    f1 = 0
    for i in range(3):
        TP = np.sum((y_true == i) & (y_pred == i))
        FP = np.sum((y_true != i) & (y_pred == i))
        FN = np.sum((y_true == i) & (y_pred != i))
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        if np.isnan(precision + recall) or (precision + recall) == 0:
            f1 += 0
            print(f"f1{i}: 0")
        else:
            f1 += 2 * precision * recall / (precision + recall)
            print(f"f1{i}: {2 * precision * recall / (precision + recall)}")
    return f1 / 3


test = np.ones(232).astype(int)
i = 70
j = 140
golden = np.zeros(232)
golden[i:] += 1
golden[j:] += 1
golden = golden.astype(int)
print("golden:", golden)
print("test[0]:", test[0])
print("f1score:", calc_3class_f1(golden, test))
print("sklearn f1score:", f1_score(golden, test, average="macro"))
test[0] = 0
print("test[0]:", test[0])
print("f1score:", calc_3class_f1(golden, test))
print("sklearn f1score:", f1_score(golden, test, average="macro"))
test[0] = 2
print("test[0]:", test[0])
print("f1score:", calc_3class_f1(golden, test))
print("sklearn f1score:", f1_score(golden, test, average="macro"))

# for k in range(1000):
#     test_set = {0, 1, 2}
#     test = np.random.randint(0, 3, 232)
#     test[0] = golden[0]
#     test_set.remove(test[0])
#     f11 = calc_3class_f1(golden, test)

#     test[0] = test_set.pop()
#     f12 = calc_3class_f1(golden, test)

#     test[0] = test_set.pop()
#     f13 = calc_3class_f1(golden, test)

#     if f11 < f12 or f11 < f13:
#         print(i, j, f11, f12, f13)
#         # print(golden)
#         # print(test)
#         print('----------------------')
    # let test[0] = others


# test = np.zeros(232).astype(int)
# test[68:] += 1
# test[155:] += 1

# test = np.zeros(232)

# result = np.zeros((232, 232))

# for i in range(232):
#     for j in range(232 - i):
#         golden = np.zeros(232)
#         golden[i:] += 1
#         golden[j:] += 1
#         golden = golden.astype(int)

#         f1 = calc_3class_f1(golden, test)
#         result[i, j] = f1
#         # print(f1)
#         if f1 > 0.371 and f1 < 0.372:
#             print(i, j, f1)
#             print(golden)
#             print('----------------------')

# # save result to csv
# np.savetxt('f1_score.csv', result, delimiter=',')