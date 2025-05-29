import pickle
import tqdm
ds = pickle.load(open('./cgcnn_dataset_new_fin.pkl','rb'))
oh = [d for d in ds if d['adsorbate'] == 'OH']
o = [d for d in ds if d['adsorbate'] == 'O']
ooh = [d for d in ds if d['adsorbate'] == 'OOH']
new_list = list()
for d1 in tqdm.tqdm(oh):
	flag = 0
	for d2 in ooh:
		if d1['mpid'] == d2['mpid'] and d1['miller'] == d2['miller'] and d1['shift'] == d2['shift'] and d1['top'] == d2['top']:
			for d3 in o:
				if d1['mpid'] == d3['mpid'] and d1['miller'] == d3['miller'] and d1['shift'] == d3['shift'] and d1['top'] == d3['top']:
					flag = 1
					break
		if flag == 1:
			break
	if flag == 1:
		d = dict()
		d['mpid'] = d1['mpid']
		d['miller'] = d1['miller']
		d['shift'] = d1['shift']
		d['top'] = d1['top']
		d['is_metal'] = d1['is_metal']
		d['init_struct'] = d1['init_struct']
		d['final_struct'] = d1['final_struct']
		d['delta_g_oh'] = d1['energy'] + 0.32
		d['delta_g_o'] = d3['energy'] + 0.01
		d['delta_g_ooh'] = d2['energy'] + 0.31
		d['oer_op'] = max(d['delta_g_oh'], d['delta_g_o'] - d['delta_g_oh'],d['delta_g_ooh'] - d['delta_g_o'], 4.92 - d['delta_g_ooh']) - 1.23
		new_list.append(d)
with open('./oer_op_docs.pkl','wb') as pf:
	pickle.dump(new_list,pf)
print(len(new_list))
