class JaxMemUsage:
    usage = 0
    usage_unit = 'B'
    max_usage = 0
    max_usage_unit = 'B'
    usage_str = ''
    max_usage_str = ''

    import threading
    import time
    import jax
    import subprocess
    import posix
    interval = 0.01
    

    def inner():
        unit_dict = {'B': 1, 'KB': 1000, 'MB': 1000000, 'GB': 1000000000, 'TB': 1000000000000}
        while True:
            dir_prefix='/dev/shm'
            output = JaxMemUsage.subprocess.run(
                    args=['go', 'tool', 'pprof', '-tags', f'{dir_prefix}/memory.prof'],
                    stdout=JaxMemUsage.subprocess.PIPE,
                    stderr=JaxMemUsage.subprocess.DEVNULL,
                ).stdout.decode('utf-8')

            if output != '':
                output = output.split('device: Total')[1]
                JaxMemUsage.usage_str = output.split('\n')[0].split('\n')[0]
                if JaxMemUsage.usage_str.endswith('KB'):
                    output = JaxMemUsage.usage_str[:-2]
                    JaxMemUsage.usage_unit = 'KB'
                elif JaxMemUsage.usage_str.endswith('MB'):
                    output = JaxMemUsage.usage_str[:-2]
                    JaxMemUsage.usage_unit = 'MB'
                elif JaxMemUsage.usage_str.endswith('GB'):
                    output = JaxMemUsage.usage_str[:-2]
                    JaxMemUsage.usage_unit = 'GB'
                elif JaxMemUsage.usage_str.endswith('TB'):
                    output = JaxMemUsage.usage_str[:-2]
                    JaxMemUsage.usage_unit = 'TB'
                elif JaxMemUsage.usage_str.endswith('B'):
                    output = JaxMemUsage.usage_str[:-1]
                    JaxMemUsage.usage_unit = 'B'
                else:
                    print(output)
                JaxMemUsage.usage = float(output)
                if JaxMemUsage.usage * unit_dict[JaxMemUsage.usage_unit] > JaxMemUsage.max_usage * unit_dict[JaxMemUsage.max_usage_unit]:
                    JaxMemUsage.max_usage = JaxMemUsage.usage
                    JaxMemUsage.max_usage_unit = JaxMemUsage.usage_unit
                JaxMemUsage.max_usage_str = str(JaxMemUsage.max_usage) + JaxMemUsage.max_usage_unit
               
            JaxMemUsage.time.sleep(JaxMemUsage.interval)
            

    def launch(interval = 0.01):
        from jax_smi import initialise_tracking
        initialise_tracking(interval=interval)
        JaxMemUsage.interval = interval
        JaxMemUsage.time.sleep(JaxMemUsage.interval * 5)
        thread = JaxMemUsage.threading.Thread(target=JaxMemUsage.inner, daemon=True)
        thread.start()
