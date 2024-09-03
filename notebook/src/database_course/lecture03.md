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

The overall goal is to make it appear like we have more memory than we do:
1. We want to allow the DBMS to manage databases that exceed the amount of memory available. 
2. Reading/ writing to disk is expensive, so it must be managed carefully to avoid large stalls.
3. Random access on disk is much slower, so DBMS wants to maximize sequential access.

## Disk oriented DBMS

On disk, we have database files stored in pages. Each page is a block of data. When an execution engine makes a query, it loads a page into buffer pool in memory. The buffer pool returns a 64-bit pointer to the memory location where the page is located. The execution engine interprets the layout of the page, updates it. The new page is then written back into the database file on disk.

## Why not use the OS?

The DBMS could potentially use memory mapping `mmap` to store the contents of a file (or page) in the address space of a program. The benefit is that we can tap on the OS to decide which pages to load into physical memory and when to swap pages out. The downside is that the lack of control over memory management can lead to stalling and bad performance.

How the OS works:
1. Suppose we have Page 1, ..., Page 4 on disk. There's only enough space in physical memory to load 2 pages.
2. The OS represents this to the program as virtual memory, where Page 1, ..., Page 4 are available
3. When the program touches Page 2, the OS will load Page 2 into physical memory and return a pointer to the program
4. Suppose the program now touches Page 3 and Page 4
5. The OS needs to decide which page to evict from physical memory

So the problems with using `mmap` to handle I/O for the DBMS:
1. Transaction safety. The OS can flush dirty pages to disk at any time, even mid-write. We can get corrupted data on disk.
2. I/O stalls. The DBMS does not know which pages are in memory, so the OS could stall due to page faults (fetching a page not in memory).
3. Error handling. Difficult to validate pages. Any access can cause a `SIGBUS` that the DBMS needs to handle.
4. Performance issues. The OS has its own scheduling and data structures, and can contend with the DBMS's priorities.

There are some companies that use `mmap`:
- RavenDB
- ElasticSearch
- QuestDB
- MongoDB (moved away from `mmap`)

So DBMS almost always wants to control things itself:
- Flushing dity pages to disk in the correct order
- Specialized pre-fetching
- Buffer replacement policy
- Thread / process scheduling

Paper on this: [Are you sure you want to Use MMAP in your DBMS?](https://db.cs.cmu.edu/papers/2022/cidr2022-p13-crotty.pdf)

For database storage, we need to deal with two problems:
1. How the DBMS represents the database in files on disk
2. How the DBMS manages its memory and moves data back and forth from disk

## File Storage

The DBMS stores a database as one or more files on disk typically in a proprietary format. The OS doesn't know anything about the contents of these files. There are portable file formats (e.g. parquet etc.). Early systems in 1980s further used custom *filesystems* on raw block storage. Most newer DBMSs do not do this.

The <storage manager> is responsible for maintaining a database's files. It organizes the files as a collection of pages, and tracks the data read / written to pages, and tracks the available space in each page. A DBMS does not typically maintain multiple copies of a page on disk. This happens above or below the storage manager.

A <page> is a fixed-size block of data. It can contain tuples, meta-data, indexes, logs etc. Most systems do not mix page types. Some systems require a page to be self-contained, i.e. all the information needed to understand the data in the page is contained within the page itself (e.g. Oracle). Each page is given a unique identifier. The DBMS uses an indirection layer to map page IDs to physical locations (e.g. in memory or in S3 or whatever).

There are 3 different notions of what a page is:
1. Hardware page (typically `4KB`). A hardware page is the largest block of data that the storage device can guarantee failsafe writes (i.e. if it says it wrote `4KB`, it actually happened).
2. OS Page (usually `4KB`, but can be much larger)
3. Database Page (`512B` to `32KB`)

Why might a larger page size be a good idea? Sequential data access. If we request a page of `16KB`, it is one call to the OS and data is stored sequentially. If we request 4 pages of `4KB` each, it is potentially random access. But the downside is that writing to a larger page is slower than writing to a small one. 

## Page Storage Architecture

How do DBMSs map page IDs to physical location? Different approaches:
- Heap file organization <- most common
- Tree file organization
- Sequential / Sorted File Organization (ISAM)
- Hashing file organization

A <<heap file>> is an unordered collection of pages with tuples that are stored in random order. 