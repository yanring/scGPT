from gears import PertData, GEARS
import anndata
split = "simulation"
batch_size = 64
eval_batch_size = 64
adata = anndata.read_h5ad('/home/scratch.zijiey_sw/Algorithm/scGPT/scgpt/save/new/scgpt_train/set3_train_gears.h5ad')
pert_data = PertData("./data")
pert_data.new_data_process(dataset_name = 'set3_train_gears', adata = adata)
pert_data.load(data_path = './data/set3_train_gears')
pert_data.prepare_split(split=split, seed=1)
pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)
