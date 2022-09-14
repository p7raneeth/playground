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

Attrition.No	Attrition.Yes	BusinessTravel.Travel_Frequently	BusinessTravel.Travel_Rarely	Department.Cardiology	Department.Maternity	EducationField.Life Sciences	EducationField.Medical	EducationField.Other	Gender.Female	Gender.Male	JobRole.Administrative	JobRole.Nurse	JobRole.Other	JobRole.Therapist	MaritalStatus.Divorced	MaritalStatus.Married	MaritalStatus.Single
1               	0	                 0	                              1   	                      0	                    1	                         0	                        1	                       0	                   0             	1	         0                 	0	              0	             1                     	0	                 0                    	1

In [ ]:

# fillin the missing values with most_frequent value
for i, col in enumerate(list(obj_data.columns)):
    obj_data[col].fillna(user_inputs[i], inplace=True)

obj_data


## initialize a list of 'n' zeros 
# e.g if input is 5 then output is [0,0,0,0,0]
def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

  

# create col_list_names which contains encoded format of each column
# E.g Attrition has two unique values: ['Yes', 'No']. So, Attrition.Yes and Attrition.No will be final column names for Attrition column
col_list_names = []

for col in list(obj_data.columns):
    for val in np.unique(obj_data[col]):
        col_list_names.append(col+'.'+val)


# creating a dictionary in such a way that user inputs are stored in a list and in integer value to specify unique values for that column

## E.g if User input for Attrition is No, then the dictionary d will contain the values 

# d = {'Attrition' : ['No', 2]} so on and so forth a]for all the remaining values
d = {}

for i, col in enumerate(list(obj_data.columns)):
    d[col] = [user_inputs[i], len(np.unique(obj_data[col]))]

    

m=0
master_list = []
# iterate over the user_input dictionary for all the items
for k, v in d.items():
  # make a list of required number of zero's based on unique values for a particular key in the dictionary
    answer_list = zerolistmaker(len(col_list_names[m:m+v[1]]))
    for i, val in enumerate(col_list_names[m:m+v[1]]):
        # only update that index value of the list where 'No' is present and leave the rest as it is 
        if d[k][0] == val.split('.')[1]:
            answer_list[i] = 1
            master_list.extend(answer_list)
    m = m + v[1]

len(master_list)

master_list

online_prediction = pd.DataFrame(pd.Series(master_list)).T

online_prediction.columns = col_list_names

online_prediction

user_inputs






