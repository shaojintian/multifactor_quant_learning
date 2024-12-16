内存对齐
struct MyStruct {
    int a;
    double b __attribute__((aligned(16)));
};


auto forward 
DPDK RDMA 

```c++
绑定到固定的cpu上
cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t1.native_handle(), sizeof(cpu_set_t), &cpuset);
    pthread_setaffinity_np(t2.native_handle(), sizeof(cpu_set_t), &cpuset);
```