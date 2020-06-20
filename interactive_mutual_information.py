import matplotlib.pyplot as plt
import numpy as np
from pprint import pformat

from sparse_discrete_table import SparseDiscreteTable
from utils.mutual_information import construct_mi_grid
from utils.plots import interactive_mi_grid
from utils.setup import setup
import logging as log

_info = log.info


def main():
    conf, shaper, sparse_crimes = setup(data_sub_path="T24H-X850M-Y880M_2012-01-01_2019-01-01")

    squeezed_crimes = shaper.squeeze(sparse_crimes)
    # squeezed_crimes[squeezed_crimes > 0] = 1
    squeezed_crimes[squeezed_crimes > 40] = 40
    squeezed_crimes = np.round(np.log2(1 + squeezed_crimes))
    _info(f"squeezed_crimes values =>{np.unique(squeezed_crimes)}")

    n, c, l = squeezed_crimes.shape

    # setup the day of the week variables
    dow_n = np.arange(n)%7
    dow_nc = np.expand_dims(dow_n,(1,2))
    dow_ncl = np.ones((n,c,l))*dow_nc

    data = np.concatenate([squeezed_crimes,dow_ncl],axis=1)


    # todo abstract into function
    K = 22
    rv_names = ['RV0_Ct','RV0_Dt','RV1_Ct-k','RV1_Dt-k'] # Ct: crime at t. Dt: day of week at t
    mi_list = []
    cmi_list = []
    for i in range(l):
        if i % (l//10) == 0:
            print(f"{i+1}/{l} => {(i+1)/l*100}%")
        mi_list.append([])
        cmi_list.append([])
        for k in range(0,K+1): # K is the maximum
            if k == 0:
                joint = np.concatenate([data[:,:,i],data[:,:,i]],axis=1)
            else:
                joint = np.concatenate([data[k:,:,i],data[:-k,:,i]],axis=1)
            val,cnt = np.unique(joint,return_counts=True,axis=0)
            prb = cnt / np.sum(cnt)
            table = {}
            for k_,v_ in zip(list(map(tuple, val)),list(prb)):
                table[k_] = v_
            rv = SparseDiscreteTable(rv_names=rv_names,table=table)
            mi = rv.mutual_information(rv_names_0=['RV0_Ct'],
                                       rv_names_1=['RV1_Ct-k'])
            cmi = rv.conditional_mutual_information(rv_names_0=['RV0_Ct'],
                                                    rv_names_1=['RV1_Ct-k'],
                                                    rv_names_condition=['RV0_Dt','RV1_Dt-k'])
            cmi_list[i].append(cmi)
            mi_list[i].append(mi)


    cmi_grid = construct_mi_grid(cmi_list)
    mi_grid = construct_mi_grid(mi_list)

    interactive_mi_grid(mi_grid=mi_grid, crime_grid=sparse_crimes,is_conditional_mi=False)
    interactive_mi_grid(mi_grid=cmi_grid, crime_grid=sparse_crimes,is_conditional_mi=True)


if __name__ == '__main__':
    main()
