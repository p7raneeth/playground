df = pd.read_csv('/Users/praneeth/Desktop/Watson_healthcare_num_categorical_with_missing_values - Sheet1.csv')

obj_data = df.select_dtypes(include=['object'])

list(obj_data.iloc[1])

user_inputs = ['No', 
               'Travel_Rarely', 
               'Maternity', 
               'Medical', 
               'Male',
               'Therapist',
               'Single']

for i, col in enumerate(list(obj_data.columns)):
    obj_data[col].fillna(user_inputs[i], inplace=True)

obj_data

def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

## making of inp dictionary

col_list_names = []

for col in list(obj_data.columns):
    for val in np.unique(obj_data[col]):
        col_list_names.append(col+'.'+val)

col_list_names

d = {}

for i, col in enumerate(list(obj_data.columns)):
    d[col] = [user_inputs[i], len(np.unique(obj_data[col]))]

m=0
master_list = []
for k, v in d.items():
#     print(m, m+v[1])
    answer_list = zerolistmaker(len(col_list_names[m:m+v[1]]))
    for i, val in enumerate(col_list_names[m:m+v[1]]):
#         print(inp[k][0])
        if d[k][0] == val.split('.')[1]:
#             print('---i---', i)
            answer_list[i] = 1
            master_list.extend(answer_list)
    m = m + v[1]

len(master_list)

master_list

online_prediction = pd.DataFrame(pd.Series(master_list)).T

online_prediction.columns = col_list_names

online_prediction

user_inputs






