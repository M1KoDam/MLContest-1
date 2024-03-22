from data_handler import handle_features, handle_labels, get_classes
import pandas as pd
import numpy as np

train_data = pd.read_csv("data/train.csv")
useless = \
    {'Var30', 'Var86', 'Var79', 'Var180', 'Var53', 'Var124', 'Var166', 'Var104', 'Var157', 'Var117', 'Var108', 'Var37',
     'Var36', 'Var23', 'Var32', 'Var121', 'Var111', 'Var179', 'Var45', 'Var122', 'Var12', 'Var68', 'Var156', 'Var89',
     'Var127', 'Var42', 'Var142', 'Var217', 'Var43', 'Var16', 'Var70', 'Var40', 'Var183', 'Var141', 'Var147', 'Var9',
     'Var88', 'Var82', 'Var131', 'Var87', 'Var199', 'Var50', 'Var161', 'Var66', 'Var202', 'Var19', 'Var110', 'Var60',
     'Var62', 'Var114', 'Var116', 'Var31', 'Var1', 'Var63', 'Var33', 'Var26', 'Var54', 'Var120', 'Var96', 'Var152',
     'Var175', 'Var129', 'Var164', 'Var128', 'Var184', 'Var148', 'Var47', 'Var55', 'Var29', 'Var71', 'Var192', 'Var39',
     'Var105', 'Var64', 'Var101', 'Var171', 'Var69', 'Var174', 'Var98', 'Var150', 'Var159', 'Var107', 'Var18', 'Var102',
     'Var145', 'Var95', 'Var58', 'Var204', 'Var91', 'Var80', 'Var158', 'Var188', 'Var177', 'Var61', 'Var84', 'Var230',
     'Var27', 'Var52', 'Var103', 'Var172', 'Var92', 'Var118', 'Var48', 'Var186', 'Var139', 'Var41', 'Var59', 'Var46',
     'Var115', 'Var169', 'Var138', 'Var5', 'Var49', 'Var151', 'Var100', 'Var154', 'Var162', 'Var185', 'Var135', 'Var56',
     'Var99', 'Var170', 'Var8', 'Var93', 'Var17', 'Var137', 'Var11', 'Var34', 'Var15', 'Var136', 'Var176', 'Var67',
     'Var3', 'Var209', 'Var167', 'Var182', 'Var4', 'Var2', 'Var168', 'Var97', 'Var14', 'Var10', 'Var20', 'Var75',
     'Var165', 'Var190', 'Var130', 'Var178', 'Var90', 'Var77', 'Var155', 'Var106', 'Var146', 'Var187'}

for i in train_data.columns.tolist():
    if i not in useless:
        input("NEXT?")
        print(train_data[i].head(10))

train_labels_pd = train_data['Target'].values
train_features_pd = train_data.drop(columns=['Target', 'ID']).values

test_features_pd = pd.read_csv("data/data_predict.csv").drop(columns=['ID']).values
test_labels_pd = pd.read_csv("data/sample_submission.csv").drop(columns=['ID']).values

# x_train, x_test = handle_features(train_features_pd, test_features_pd)
# y_train = handle_labels(train_labels_pd)
