import glob

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series,DataFrame
from scipy.sparse import coo_matrix
import math

def visualization(data_dict):

    for key in data_dict.keys():
        data = data_dict[key]
        remove_axis = data[0]
        add_axis = data[1]
        value = data[2]
        sp_matrix = coo_matrix((value,(remove_axis,add_axis)))
        data = DataFrame(sp_matrix.todense())

        plt.figure(dpi=120)
        #sns.set_context({"figure.figsize": (8, 8)})
        sns.heatmap(data=data, square=True)
        plt.title('%s'%key)
        plt.show()
        # plt.savefig('./for_show/heatmap/%s.png'%key)
        plt.close()

def visualization_fid_new(data_dict,save_path=None,k_p = 1, k_m = 1 ):
    # 0 -> fid_plus prob
    # 2 -> fid_plus acc
    # 4 -> fid_minus prob
    # 6 -> fid_minus acc
    # 8 -> fid_delta prob
    # 10 -> fid_delta acc
    dict_name = {0: "fid_plus prob",
                 2: "fid_plus acc",
                 4: "fid_minus prob",
                 6: "fid_minus acc",
                 8: "fid_Delta prob",
                 10: "fid_Delta acc"}

    for value_count in dict_name.keys():
        name = dict_name[value_count]
        length = int(math.sqrt(len(data_dict.keys())))
        matrix = np.zeros([length,length])

        for key in data_dict.keys():
            data = data_dict[key]
            remove_axis = key[0] # remove
            add_axis = key[1]   #  add
            value = data[value_count]
            matrix[remove_axis,add_axis] = value

        data = DataFrame(matrix)
        plt.figure(dpi=120)
        #sns.set_context({"figure.figsize": (8, 8)})
        sns.heatmap(data=data, square=True)
        # plt.title('%s'%name)
        if 'plus' in name:
            plt.savefig(save_path +'_%s.png'%(name))
            plt.savefig(save_path +'_%s.pdf'%(name))
        elif 'minus' in name:
            plt.savefig(save_path +'_%s.png'%(name))
            plt.savefig(save_path +'_%s.pdf'%(name))
        else:
            plt.savefig(save_path +'_%s.png'%(name))
            plt.savefig(save_path +'_%s.pdf'%(name))
        plt.close()

def visualization_fid_new_ratio(data_dict,save_path=None,k_p = 1, k_m = 1 ):
    # 0 -> fid_plus prob
    # 2 -> fid_plus acc
    # 4 -> fid_minus prob
    # 6 -> fid_minus acc
    # 8 -> fid_delta prob
    # 10 -> fid_delta acc
    dict_name = {0: "fid_plus prob",
                 2: "fid_plus acc",
                 4: "fid_minus prob",
                 6: "fid_minus acc",
                 8: "fid_Delta prob",
                 10: "fid_Delta acc"}

    dict_ratio = {'0.0':0,'0.1':1,
                  '0.3':2,'0.5':3,
                  '0.7':4,
                  '0.9':5,}

    x_ticks = ['0.0', '0.1',
               '0.3',  '0.5',
               '0.7',
               '0.9']
    for value_count in dict_name.keys():
        name = dict_name[value_count]
        length = int(math.sqrt(len(data_dict.keys())))
        matrix = np.zeros([len(x_ticks),len(x_ticks)])

        for key in data_dict.keys():
            data = data_dict[key]
            remove_axis = key[0] # remove
            add_axis = key[1]   #  add
            if remove_axis in x_ticks and add_axis in x_ticks:
                value = data[value_count]
                matrix[dict_ratio[remove_axis],dict_ratio[add_axis]] = value

        data = DataFrame(matrix)
        plt.figure(dpi=120)
        #sns.set_context({"figure.figsize": (8, 8)})
        sns.heatmap(data=data, square=True,xticklabels=x_ticks,yticklabels=x_ticks,fmt='.2f',annot=True)
        plt.tick_params(labeltop=True)
        plt.tick_params(labelbottom=False)
        # plt.title('%s'%name)
        if 'plus' in name:
            plt.savefig(save_path +'_%s.png'%(name))
            plt.savefig(save_path +'_%s.pdf'%(name))
        elif 'minus' in name:
            plt.savefig(save_path +'_%s.png'%(name))
            plt.savefig(save_path +'_%s.pdf'%(name))
        else:
            plt.savefig(save_path +'_%s.png'%(name))
            plt.savefig(save_path +'_%s.pdf'%(name))
        plt.close()

def visualization_fid_new_k(data_dict,save_path=None,k_p = 1, k_m = 1 ):
    # 0 -> fid_plus prob
    # 2 -> fid_plus acc
    # 4 -> fid_minus prob
    # 6 -> fid_minus acc
    # 8 -> fid_delta prob
    # 10 -> fid_delta acc
    dict_name = {0: "fid_plus prob",
                 2: "fid_plus acc",
                 4: "fid_minus prob",
                 6: "fid_minus acc",
                 8: "fid_Delta prob",
                 10: "fid_Delta acc"}

    dict_ratio = {0:0,1:1,
                  2:2,3:3,
                  4:4}

    x_ticks = [0, 1,
               2,  3,
               4]
    for value_count in dict_name.keys():
        name = dict_name[value_count]
        length = int(math.sqrt(len(data_dict.keys())))
        matrix = np.zeros([len(x_ticks),len(x_ticks)])

        for key in data_dict.keys():
            data = data_dict[key]
            remove_axis = key[0] # remove
            add_axis = key[1]   #  add
            if remove_axis in x_ticks and add_axis in x_ticks:
                value = data[value_count]
                matrix[dict_ratio[remove_axis],dict_ratio[add_axis]] = value

        data = DataFrame(matrix)
        plt.figure(dpi=120)
        #sns.set_context({"figure.figsize": (8, 8)})
        sns.heatmap(data=data, square=True,xticklabels=x_ticks,yticklabels=x_ticks,fmt='.2f',annot=True)
        plt.tick_params(labeltop=True)
        plt.tick_params(labelbottom=False)
        # plt.title('%s'%name)
        if 'plus' in name:
            plt.savefig(save_path +'_%s.png'%(name))
            plt.savefig(save_path +'_%s.pdf'%(name))
        elif 'minus' in name:
            plt.savefig(save_path +'_%s.png'%(name))
            plt.savefig(save_path +'_%s.pdf'%(name))
        else:
            plt.savefig(save_path +'_%s.png'%(name))
            plt.savefig(save_path +'_%s.pdf'%(name))
        plt.close()

def visualization_fid_ori(data_dict,save_path=None):
    # 0 -> fid_plus prob
    # 2 -> fid_plus acc
    # 4 -> fid_minus prob
    # 6 -> fid_minus acc
    # 8 -> fid_delta prob
    # 10 -> fid_delta acc
    dict_name = {0: "fid_plus prob",
                 2: "fid_plus acc",
                 4: "fid_minus prob",
                 6: "fid_minus acc",
                 8: "fid_Delta prob",
                 10: "fid_Delta acc"}
    dict_ratio = {'0.0':0,'0.1':1,
                  '0.3':2,'0.5':3,
                  '0.7':4,
                  '0.9':5,}

    x_ticks = ['0.0', '0.1',
               '0.3',  '0.5',
               '0.7',
               '0.9']
    for value_count in dict_name.keys():
        name = dict_name[value_count]
        length = int(math.sqrt(len(data_dict.keys())))
        matrix = np.zeros([len(x_ticks),len(x_ticks)])

        for key in data_dict.keys():
            data = data_dict[key]
            remove_axis = key[0] # remove
            add_axis = key[1]   #  add
            if remove_axis in x_ticks and add_axis in x_ticks:
                value = data[value_count]
                matrix[dict_ratio[remove_axis],dict_ratio[add_axis]] = value

        data = DataFrame(matrix)
        plt.figure(dpi=120)
        #sns.set_context({"figure.figsize": (8, 8)})
        sns.heatmap(data=data, square=True)
        # plt.title('%s'%name)
        # plt.show()
        plt.savefig(save_path+'%s.png'%name)
        plt.savefig(save_path+'%s.pdf'%name)
        plt.close()

def visualization_fid_ori_ratio(data_dict,save_path=None):
    # 0 -> fid_plus prob
    # 2 -> fid_plus acc
    # 4 -> fid_minus prob
    # 6 -> fid_minus acc
    # 8 -> fid_delta prob
    # 10 -> fid_delta acc
    dict_name = {0: "fid_plus prob",
                 2: "fid_plus acc",
                 4: "fid_minus prob",
                 6: "fid_minus acc",
                 8: "fid_Delta prob",
                 10: "fid_Delta acc"}

    dict_ratio = {'0.0':0,'0.1':1,
                  '0.3':2,'0.5':3,
                  '0.7':4,
                  '0.9':5,}

    x_ticks = ['0.0', '0.1',
               '0.3',  '0.5',
               '0.7',
               '0.9']
    for value_count in dict_name.keys():
        name = dict_name[value_count]
        length = int(math.sqrt(len(data_dict.keys())))
        matrix = np.zeros([len(x_ticks),len(x_ticks)])

        for key in data_dict.keys():
            data = data_dict[key]
            remove_axis = key[0] # remove
            add_axis = key[1]   #  add
            if remove_axis in x_ticks and add_axis in x_ticks:
                value = data[value_count]
                matrix[dict_ratio[remove_axis],dict_ratio[add_axis]] = value

        data = DataFrame(matrix)
        plt.figure(dpi=120)
        #sns.set_context({"figure.figsize": (8, 8)})
        sns.heatmap(data=data, square=True,xticklabels=x_ticks,yticklabels=x_ticks,fmt='.2f',annot=True)
        plt.tick_params(labeltop=True)
        plt.tick_params(labelbottom=False)
        # plt.title('%s'%name)
        # plt.show()
        plt.savefig(save_path+'%s.png'%name)
        plt.savefig(save_path+'%s.pdf'%name)
        plt.close()
def visualization_fid_ori_k(data_dict,save_path=None):
    # 0 -> fid_plus prob
    # 2 -> fid_plus acc
    # 4 -> fid_minus prob
    # 6 -> fid_minus acc
    # 8 -> fid_delta prob
    # 10 -> fid_delta acc
    dict_name = {0: "fid_plus prob",
                 2: "fid_plus acc",
                 4: "fid_minus prob",
                 6: "fid_minus acc",
                 8: "fid_Delta prob",
                 10: "fid_Delta acc"}

    dict_ratio = {0:0,1:1,
                  2:2,3:3,
                  4:4}

    x_ticks = [0, 1,
               2,  3,
               4]

    for value_count in dict_name.keys():
        name = dict_name[value_count]
        length = int(math.sqrt(len(data_dict.keys())))
        matrix = np.zeros([len(x_ticks),len(x_ticks)])

        for key in data_dict.keys():
            data = data_dict[key]
            remove_axis = key[0] # remove
            add_axis = key[1]   #  add
            if remove_axis in x_ticks and add_axis in x_ticks:
                value = data[value_count]
                matrix[dict_ratio[remove_axis],dict_ratio[add_axis]] = value

        data = DataFrame(matrix)
        plt.figure(dpi=120)
        #sns.set_context({"figure.figsize": (8, 8)})
        sns.heatmap(data=data, square=True,xticklabels=x_ticks,yticklabels=x_ticks,fmt='.2f',annot=True)
        plt.tick_params(labeltop=True)
        plt.tick_params(labelbottom=False)
        # plt.title('%s'%name)
        # plt.show()
        plt.savefig(save_path+'%s.png'%name)
        plt.savefig(save_path+'%s.pdf'%name)
        plt.close()



if __name__ == "__main__":
    # testvalue()
    # gcn
    dir = './redata'

    namelist = glob.glob(dir + '/*new_fid*_1_k.npy')
    for path in namelist:
        # if 'syn1' in path:
        #     pass
        # else:
        #     continue
        data = np.load(path, allow_pickle=True).item()
        visualization_fid_new_k(data, path.replace('.npy', ''))


    # graph
    namelist = glob.glob(dir+'/*ori_fid*_1_k.npy')
    for path in namelist:
        data = np.load(path, allow_pickle=True).item()
        visualization_fid_ori_k(data, path.replace('.npy',''))

    exit(0)

