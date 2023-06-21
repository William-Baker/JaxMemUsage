# JaxMemUsage
A simple utility to track Jax GPU memory usage within Python recoding the current and peak memory usage

```
from JaxMemUsage import JaxMemUsage
JaxMemUsage.launch(interval=0.01)
```

`print(f'MaxMem: {JaxMemUsage.max_usage_str} Mem: {JaxMemUsage.usage_str}')`
