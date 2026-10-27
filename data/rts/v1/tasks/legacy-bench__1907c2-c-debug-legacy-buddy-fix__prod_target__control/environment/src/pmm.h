/* pmm.h - Physical Memory Manager (circa 1998)
 * Buddy allocator for physical page frames
 */

#ifndef PMM_H
#define PMM_H

#define PAGE_SIZE 4096
#define MAX_ORDER 4
#define PAGES_PER_ZONE 64
#define NUM_ZONES 2

#define ZONE_DMA    0
#define ZONE_NORMAL 1

#define REGION_USABLE    0
#define REGION_RESERVED  1
#define REGION_ACPI      2

struct region_entry {
    unsigned int base_page;
    unsigned int page_count;
    int type;
};

struct free_area {
    int count;
    int list[PAGES_PER_ZONE];
    int list_len;
};

struct zone {
    int id;
    unsigned int start_page;
    unsigned int page_count;
    unsigned char bitmap[PAGES_PER_ZONE];
    struct free_area free_areas[MAX_ORDER + 1];
    int alloc_count;
    int free_count;
    int split_count;
    int coalesce_count;
};

struct pmm {
    struct zone zones[NUM_ZONES];
    struct region_entry regions[16];
    int region_count;
    int total_pages;
    int usable_pages;
    int reserved_pages;
};

int pmm_init(struct pmm *pmm);
int pmm_add_region(struct pmm *pmm, unsigned int base, unsigned int count, int type);
int pmm_alloc_pages(struct pmm *pmm, int zone_id, int order);
int pmm_free_pages(struct pmm *pmm, int zone_id, int page_idx, int order);
int pmm_query_page(struct pmm *pmm, int zone_id, int page_idx);
void pmm_zone_stats(struct pmm *pmm, int zone_id, char *out, int out_size);
void pmm_global_stats(struct pmm *pmm, char *out, int out_size);
int pmm_freelist_count(struct pmm *pmm, int zone_id, int order);

#endif
