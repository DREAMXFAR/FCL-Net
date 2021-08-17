import sys

import os,pdb,time

# os.environ['CUDA_VISIBLE_DEVICES']='0,2'

#######################################################################################
## relative to config

config_file = 'standard_RCF.yaml'

#######################################################################################
ckpt_dir = '../ckpt'
main_dirs = [ckpt_dir]

# Get job name
with open('config/'+config_file, 'r') as f:  # , encoding="gbk"
    lines = f.readlines()
    # lines.decode("utf8","ignore")
    job_name = lines[0][:-1].split(': ')[1][1:-1]

config = config_file.split('/')
if len(config) == 1:
    filename = config[0]
    dirs = []
else:
    dirs,filename = config[0:-1], config[-1]

filename = filename.split('.')[0]
dirs.append(filename)
dirs.append('log')

for each_main_dir in main_dirs:
    for ind, each_dir in enumerate(dirs):
        each_dir = '/'.join( dirs[0:ind+1])
        new_dir = os.path.join(each_main_dir, each_dir) 
        if not os.path.exists( new_dir ):
            os.mkdir( new_dir )
    print('create ckpt dir: ', each_main_dir, '/',  each_dir)

time.ctime()
cur_time = time.strftime('_%b%d_%H-%M-%S') 

#######################################################################################
# run script
cmd = '''\
    LOG="''' + ckpt_dir + '/' + '/'.join(dirs) + '/' + filename  + '''-`date +'%Y-%m-%d_%H-%M-%S'`_train" 
    echo $LOG ;
    python run.py --mode train --cfg ''' +  config_file + ''' --time ''' + cur_time + '''$2>&1 | tee ${LOG}
'''

os.system(cmd)

















