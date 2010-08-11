struct device_stats {
	unsigned char word[64];			// found word passed from GPU
	int hash_found;			// boolean if word is found
};

struct cuda_device {
	int device_id;
	struct cudaDeviceProp prop;

	int max_threads;
	int max_blocks;
	int shared_memory;

	void *device_global_memory;
	int device_global_memory_len;

	void *host_memory;

	void *device_stats_memory;
	struct device_stats stats;

	unsigned int target_hash[4];

	// to be used for debugging
	void *device_debug_memory;
};


