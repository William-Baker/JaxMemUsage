#%%

from JaxMemUsage import JaxMemUsage
JaxMemUsage.launch(interval=0.01)

print(f'MaxMem: {JaxMemUsage.max_usage_str} Mem: {JaxMemUsage.usage_str}')

#%%

from tqdm import tqdm

with tqdm(total=100, unit='batch') as tepoch:
  for i in range(100):
    # model.train()
    
    tepoch.set_postfix({'MaxMem': JaxMemUsage.max_usage_str, 'Mem': JaxMemUsage.usage_str})
    tepoch.update(1)
    
#%%
