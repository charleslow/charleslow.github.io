# Lecture 3: Database Storage 1

Class will shift gears towards how to implement a DBMS. Focus is on a single node machine first, then move to concurrency. Today's focus is on the <Disk Manager>.

Focus on <disk-based architecture>, i.e. the DBMS assumes that the primary storage location of the database is on non-volatile disk. The DBMS's components manage the movement of data between volatile (e.g. RAM) and non-volatile storage. 

Storage hierarchy. Higher rank is faster, smaller, expensive.
1. CPU Registers
2. CPU Caches
3. DRAM (i.e. memory)
4. SSD
5. HDD
6. Network Storage

Items 1-3 are volatile, i.e. random access, byte-addressable (we can fetch specific bytes). Items 4-6 are non-volatile, i.e. sequential access, block-addressable. For non-volatile storage, we need to fetch data one block at a time, and accessing data in one contiguous block sequentially is much faster than trying to access blocks scattered around. CPU refers to items 1 and 2, Memory refers to item 3, Disk refers to items 4-6.

Some new storage category: Fast network storage is in between HDD and SSD. Persistent Memory is between SSD and DRAM.

DBMSs need to exploit the sequential access on non-volatile memory to maximize speed of retrieval. Hence how we store data contiguously matters in the design of the DBMS.

Access times:
```
- 1 ns          L1 Cache Ref        1 sec
- 4 ns          L2 Cache Ref        4 sec
- 100 ns        DRAM                100 sec
- 16,000 ns     SSD                 4.4 hours
- 2,000,000 ns  HDD                 3.3 weeks
- 50,000,000 ns Network Storage     1.5 years
```

## System Design Goals

We want to allow the DBMS to manage databases that exceed the amount of memory available. 
Reading/ writing to disk is expensive, so it must be managed carefully to avoid large stalls.
Random access on disk is much slower, so DBMS wants to maximize sequential access.

