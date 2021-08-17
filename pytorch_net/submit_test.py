
import os,pdb
import time

#######################################################################################
## relative to config
# config_file = 'standard_RCF_LSTM.yaml'
config_file = 'standard_RCF_ms.yaml'
# config_file = 'standard_BDCN.yaml'
# config_file = 'standard_RCF_LSTM_CLS.yaml'

#######################################################################################
ckpt_dir = '../ckpt'
main_dirs = [ckpt_dir]

### process config
config=config_file.split('/')
if len(config)==1:
    filename = config[0]
    dirs = []
else:
    dirs,filename = config[0:-1], config[-1]
filename = filename.split('.')[0]
dirs.append(filename)

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
    LOG="''' + ckpt_dir + '/' + '/'.join(dirs) + '/' + filename  + '''-`date +'%Y-%m-%d_%H-%M-%S'`_test";
    echo $LOG ;
    python run.py --mode test_ms --cfg ''' +  config_file + ''' --time ''' + cur_time + '''$2>&1 | tee ${LOG}
'''

os.system(cmd)

















