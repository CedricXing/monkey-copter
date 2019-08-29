# insert lines to print coverage information

import glob, os
import subprocess
from injector import *

inputpath = "/home/cedric/Desktop/ArduPilot_Back/"
# inputpath2 = "/home/cedric/ArduPilot/libraries/AC_WPNav_back/"


def insertCover(folderName,fileName):
    filename_list = []
    filename_list.append(fileName)

#   os.chdir(folderName)
#   for file in glob.glob("*.cpp"):
      #print(file)
  	# filename_list.append(file)
  #subprocess.call("dir/s", shell=True)


    #print filename_list
    lookup = 'if'
    lookup2 = '{'
    lookup3 = 'else if'
    lookup4 = 'else'

    addstr  = "  EXECUTE_MARK();\n"

    num_list = []

    os.chdir(folderName)
    for i in range(len(filename_list)):
        input_f = filename_list[i]
        print input_f
        del num_list[:]
        with open(input_f,"r+") as myFile:
            for num, line in enumerate(myFile, 1):
                if lookup in line:
                    if lookup3 in line:
                        do_nothing = 0
                        #print 'found else if at line:', num	
                    elif lookup2 in line:
                        #print "index is", line.index(lookup2)    
                        comindex = 	line.index(lookup2) 		   
                        num_list.append((num, comindex))
  			   
        list_length = len(num_list)	
        out_dir = folderName
        output_f = out_dir + input_f + ".tmp"
        if list_length == 0:
            with open(input_f) as fin, open(output_f,'w') as fout:
                for i, item in enumerate(fin, 1):
  		            fout.write(item) 
        else:
            with open(input_f) as fin, open(output_f,'w') as fout:
                j = 0
                for i, item in enumerate(fin, 1):
                    if i == num_list[j][0]: 
                        new_str = item.replace("\n",addstr)
                        fout.write(new_str)
                        if j < list_length - 1 :
  			                j = j+1
                        else:
  			                j = j
                    else:
  		                fout.write(item)




    filename_tmp_list = []
    num_tmp_list = []

    os.chdir(folderName)
    for file in glob.glob("*.tmp"):
        #print(file)
  	    filename_tmp_list.append(file)



    for i in range(len(filename_tmp_list)):
        input_tmp_f = filename_tmp_list[i]
      #print input_f
        del num_tmp_list[:]
        with open(input_tmp_f,"r+") as myFile_tmp:
            for num_tmp, line_tmp in enumerate(myFile_tmp, 1):
                if lookup4 in line_tmp  and lookup2 in line_tmp:
                 #print "index is", line_tmp.index(lookup2)    
                    comindex = 	line_tmp.index(lookup2) 		   
                    num_tmp_list.append((num_tmp, comindex))
  			   
        list_tmp_length = len(num_tmp_list)	
        out_dir = folderName
        output_f = out_dir + input_tmp_f + ".tmpp"
        if list_tmp_length == 0:
            with open(input_tmp_f) as fin, open(output_f,'w') as fout:
                for i, item_tmp in enumerate(fin, 1):
  		            fout.write(item_tmp) 
            os.remove(input_tmp_f)
        else:
            with open(input_tmp_f) as fin, open(output_f,'w') as fout:
                j = 0
                for i, item_tmp in enumerate(fin, 1):
                    if i == num_tmp_list[j][0]: 
                        new_str = item_tmp.replace("\n",addstr)
                        fout.write(new_str)
                        if j < list_tmp_length -1 :
    			            j = j+1
                        else:
    			            j = j
                    else:
    		            fout.write(item_tmp)
            os.remove(input_tmp_f)
  			
  			
    filename_add_list = []


    os.chdir(folderName)
    for file_add in glob.glob("*.tmp.tmpp"):
      #print(file)
  	    filename_add_list.append(file_add)
  	
    def insert(originalfile,string):
        with open(originalfile,'r') as f:
            with open('newfile.txt','w') as f2: 
                # f2.write(string)
                f2.write(f.read())
        os.remove(originalfile)
        os.rename('newfile.txt',originalfile.split(".tmp")[0])
  	
    add_string = '#include <AP_HAL/AP_HAL.h> \n'
  	
    for i in range(len(filename_add_list)):
        insert(filename_add_list[i], add_string)


if __name__ == '__main__':
    file_already_inserted = []
    for bug in bug_group + real_life_bug_group:
        if bug['file'] not in file_already_inserted:
            split_pos = bug['file'].rfind('/') + 1
            insertCover(inputpath+bug['file'][:split_pos],bug['file'][split_pos:])
            file_already_inserted.append(bug['file'])