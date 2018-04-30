import os


data_dir = "data/orm_data/"
input_files = ["setE-500.tok.eng", "setE-500.tok.orm"]
output_files = ["setE-500.clean.tok.eng", "setE-500.clean.tok.orm"]

for i in range(len(input_files)):
  input_files[i] = os.path.join(data_dir, input_files[i]) 
  output_files[i] = os.path.join(data_dir, output_files[i]) 

in_lines_1 = open(input_files[0], 'r').readlines()
in_lines_2 = open(input_files[1], 'r').readlines()

out_file_1 = open(output_files[0], 'w') 
out_file_2 = open(output_files[1], 'w') 

for i1, i2, in zip(in_lines_1, in_lines_2):
  if "#untranslated" in i1:
    continue
  out_file_1.write(i1)
  out_file_2.write(i2)

out_file_1.close()
out_file_2.close()
