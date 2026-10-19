"""Sample process resources; label whole-device GPU readings separately."""
import os
import threading
import time
import psutil
import torch

class Monitor:
    def __init__(self, interval=1.0):
        self.interval=interval;self.stop_event=threading.Event();self.rows=[]
        self.process=psutil.Process(os.getpid());self.nvml=None
        try:
            import pynvml
            pynvml.nvmlInit();self.nvml=pynvml
        except Exception: pass
    def sample(self):
        try:
            procs=[self.process]+self.process.children(recursive=True)
            rss=sum(p.memory_info().rss for p in procs if p.is_running())
            row={'elapsed_seconds':time.monotonic()-self.start,'rss_process_tree_bytes':rss,
                 'cpu_process_percent':self.process.cpu_percent(None),'gpu_devices':[]}
            if self.nvml:
                for i in range(self.nvml.nvmlDeviceGetCount()):
                    h=self.nvml.nvmlDeviceGetHandleByIndex(i)
                    u=self.nvml.nvmlDeviceGetUtilizationRates(h)
                    row['gpu_devices'].append({'nvml_index':i,'uuid':self.nvml.nvmlDeviceGetUUID(h),
                      'utilization_percent':u.gpu,'memory_used_bytes':self.nvml.nvmlDeviceGetMemoryInfo(h).used})
            self.rows.append(row)
        except (psutil.Error,RuntimeError): pass
        except Exception: self.nvml=None
    def __enter__(self):
        self.start=time.monotonic();self.cpu_start=self.process.cpu_times();self.process.cpu_percent(None)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(i)
        def loop():
            while not self.stop_event.wait(self.interval): self.sample()
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop_event.set();self.thread.join();self.sample()
        elapsed=time.monotonic()-self.start;cpu=self.process.cpu_times()
        gpu_summary={}
        for row in self.rows:
            for device in row['gpu_devices']:
                entry=gpu_summary.setdefault(device['uuid'], {'utilization_percent':[], 'memory_used_bytes':[]})
                entry['utilization_percent'].append(device['utilization_percent'])
                entry['memory_used_bytes'].append(device['memory_used_bytes'])
        gpu_summary={uuid:{'peak_utilization_percent':max(v['utilization_percent']),
            'mean_utilization_percent':sum(v['utilization_percent'])/len(v['utilization_percent']),
            'peak_memory_used_bytes':max(v['memory_used_bytes'])} for uuid,v in gpu_summary.items()}
        self.result={'wall_seconds':elapsed,'whole_device_gpu_sampled':gpu_summary,'sample_interval_seconds':self.interval,
          'peak_ram_rss_process_tree_bytes':max((r['rss_process_tree_bytes'] for r in self.rows),default=None),
          'peak_cpu_process_percent_sampled':max((r['cpu_process_percent'] for r in self.rows),default=None),
          'mean_cpu_process_percent':100*(cpu.user+cpu.system-self.cpu_start.user-self.cpu_start.system)/elapsed,
          'cuda_allocator_peaks':[{'logical_device':i,'allocated_bytes':torch.cuda.max_memory_allocated(i),
             'reserved_bytes':torch.cuda.max_memory_reserved(i)} for i in range(torch.cuda.device_count())],
          'samples':self.rows,
          'note':'CPU 100%=one logical core; may exceed 100%. GPU NVML samples cover whole devices, including other processes. Sampled peaks may miss brief spikes. Null/unavailable is not zero.'}

def synchronize(device):
    if torch.device(device).type=='cuda': torch.cuda.synchronize(device)
