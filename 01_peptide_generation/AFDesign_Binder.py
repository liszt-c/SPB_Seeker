import numpy as np
from IPython.display import HTML
from colabdesign import mk_afdesign_model, clear_mem
import os


for i in range(10):
    model = mk_afdesign_model(protocol="binder")
    model.prep_inputs(pdb_filename="2bv6.pdb", chain="A", binder_len=7)

    print("target_length",model._target_len)
    print("binder_length",model._binder_len)

    model.design_3stage(100, 100, 10)

    model.save_pdb(f"{model.protocol}.pdb")

    filxe = open("./bind_seq.txt",'w') 
    x = model.get_seqs()
    filxe.writelines(str(x))
    filxe.close()

    os.system('mv -b ./bind_seq.txt ./result/bind_seq_'+str(i)+'.txt')
    os.system('mv -b ./binder.pdb ./result/binder_'+str(i)+'.pdb')
    print(str(i))

