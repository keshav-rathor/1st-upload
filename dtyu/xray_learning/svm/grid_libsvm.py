'''
Perform multi-threaded grid search
'''

import threading
import subprocess
import itertools

param_list = [
    ['t', [2]],
    ['c', [2**-5, 2**-3, 2**-1, 2, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]],
    ['g', [2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2, 2**3]],
    ['b', [1]],
    ['w1', [5.7]],
    ['w-1', [1]]
]

'''
Utility class for iterating grid options and generate arguments and description
strings.
'''
class GridOptions:
    def __init__(self, param_list):
        self.param_list = param_list
        self.param_names = [row[0] for row in param_list]
        self.param_values = [row[1] for row in param_list]
        self.param_tuples = list(itertools.product(*self.param_values))

    '''
    Generate the argument list for command line call.
    '''
    def get_args_list(self, param_tuple):
        l = []
        for i, name in enumerate(self.param_names):
            l.append('-' + name)
            l.append(str(param_tuple[i]))
        return l

    '''
    Generate the argument description suffixes to name the output files.
    '''
    def get_args_suffix(self, param_tuple):
        s = ''
        for i, name in enumerate(self.param_names):
            if len(self.param_values[i]) > 1:   # discard the fixed args
                s = s + '_' + name + '_' + str(param_tuple[i])
        return s


'''
Thread object to handle a single SVM run.
'''
class SVMThread(threading.Thread):
    def __init__(self, cmd_train, cmd_predict):
        threading.Thread.__init__(self)
        self.cmd_train = cmd_train
        self.cmd_predict = cmd_predict

    def run(self):
        #return subprocess.check_output(cmd)
        #print(' '.join(cmd_train))
        subprocess.call(cmd_train)
        subprocess.call(cmd_predict)


if __name__ == '__main__':
    DATASET = 'train'
    DATASET_VAL = 'val'
    FC_LAYER = 1
    LABEL_INDEX = 2

    bsvm_train_path = '/home/zquan/libsvm/svm-train'
    bsvm_predict_path = '/home/zquan/libsvm/svm-predict'
    oned_binary_dir = '../../xray_data/synthetic_oned_binary/'
    input_path = oned_binary_dir + 'data_fc%d_label%d_%s' % (FC_LAYER, LABEL_INDEX, DATASET)
    output_path_prefix = oned_binary_dir + 'runs/model_fc%d_label%d' % (FC_LAYER, LABEL_INDEX)
    input_path_val = oned_binary_dir + 'data_fc%d_label%d_%s' % (FC_LAYER, LABEL_INDEX, DATASET_VAL)
    output_path_val_prefix = oned_binary_dir + 'runs/predict_fc%d_label%d' % (FC_LAYER, LABEL_INDEX)

    go = GridOptions(param_list)
    svm_runs = []
    for t in go.param_tuples:
        # prepare command calls
        output_path = output_path_prefix + go.get_args_suffix(t)
        output_path_val = output_path_val_prefix + go.get_args_suffix(t)

        cmd_train = [bsvm_train_path]
        cmd_train.extend(go.get_args_list(t))
        cmd_train.extend([input_path, output_path])
        cmd_predict = [bsvm_predict_path]
        cmd_predict.extend(['-b', '1', input_path_val, output_path, output_path_val])
        print(' '.join(cmd_train))
        svm_run = SVMThread(cmd_train, cmd_predict)
        svm_run.start()
        svm_run.join()