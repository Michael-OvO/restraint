#!/bin/bash

cat > /app/src/pmm.h << 'EOF'
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
EOF

cat > /app/src/pmm.c << 'EOF'
#include "pmm.h"
#include <stdio.h>
#include <string.h>

static int find_buddy(int pg, int ord) {
    return pg ^ (1 << ord);
}

int pmm_init(struct pmm *pmm) {
    pmm->region_count = 0;
    pmm->total_pages = 0;
    pmm->usable_pages = 0;
    pmm->reserved_pages = 0;

    for (int z = 0; z < NUM_ZONES; z++) {
        pmm->zones[z].id = z;
        pmm->zones[z].start_page = z * PAGES_PER_ZONE;
        pmm->zones[z].page_count = PAGES_PER_ZONE;
        pmm->zones[z].alloc_count = 0;
        pmm->zones[z].free_count = 0;
        pmm->zones[z].split_count = 0;
        pmm->zones[z].coalesce_count = 0;

        for (int i = 0; i < PAGES_PER_ZONE; i++) {
            pmm->zones[z].bitmap[i] = 0;
        }

        for (int o = 0; o <= MAX_ORDER; o++) {
            pmm->zones[z].free_areas[o].count = 0;
            pmm->zones[z].free_areas[o].list_len = 0;
        }

        for (int i = 0; i < PAGES_PER_ZONE; i += (1 << MAX_ORDER)) {
            int ord = MAX_ORDER;
            struct free_area *fa = &pmm->zones[z].free_areas[ord];
            fa->list[fa->list_len++] = i;
            fa->count++;
        }

        pmm->total_pages += PAGES_PER_ZONE;
        pmm->usable_pages += PAGES_PER_ZONE;
    }

    return 0;
}

int pmm_add_region(struct pmm *pmm, unsigned int base, unsigned int count, int type) {
    if (pmm->region_count >= 16) {
        return -1;
    }

    int r = pmm->region_count++;
    pmm->regions[r].base_page = base;
    pmm->regions[r].page_count = count;
    pmm->regions[r].type = type;

    if (type != REGION_USABLE) {
        for (unsigned int pg = base; pg < base + count; pg++) {
            int z = pg / PAGES_PER_ZONE;
            if (z < NUM_ZONES) {
                int i = pg % PAGES_PER_ZONE;
                if (pmm->zones[z].bitmap[i] == 0) {
                    pmm->zones[z].bitmap[i] = 2;
                    pmm->reserved_pages++;
                }
            }
        }
    }

    return 0;
}

int pmm_alloc_pages(struct pmm *pmm, int zone_id, int order) {
    if (zone_id < 0 || zone_id >= NUM_ZONES) {
        return -1;
    }
    if (order < 0 || order > MAX_ORDER) {
        return -2;
    }

    struct zone *z = &pmm->zones[zone_id];
    int block_size = 1 << order;

    for (int cur_order = order; cur_order <= MAX_ORDER; cur_order++) {
        struct free_area *fa = &z->free_areas[cur_order];
        for (int idx = 0; idx < fa->list_len; idx++) {
            int page_idx = fa->list[idx];

            int has_reserved = 0;
            for (int i = 0; i < (1 << cur_order); i++) {
                if (z->bitmap[page_idx + i] == 2) {
                    has_reserved = 1;
                    break;
                }
            }
            if (has_reserved) {
                continue;
            }

            for (int j = idx; j < fa->list_len - 1; j++) {
                fa->list[j] = fa->list[j + 1];
            }
            fa->list_len--;
            fa->count--;

            int split_order = cur_order;
            while (split_order > order) {
                split_order--;
                z->split_count++;
                int buddy = page_idx + (1 << split_order);
                struct free_area *lower = &z->free_areas[split_order];
                lower->list[lower->list_len++] = buddy;
                lower->count++;
            }

            for (int i = 0; i < block_size; i++) {
                z->bitmap[page_idx + i] = 1;
            }

            z->alloc_count++;
            return page_idx;
        }
    }

    return -3;
}

int pmm_free_pages(struct pmm *pmm, int zone_id, int page_idx, int order) {
    if (zone_id < 0 || zone_id >= NUM_ZONES) {
        return -1;
    }
    if (order < 0 || order > MAX_ORDER) {
        return -2;
    }

    struct zone *z = &pmm->zones[zone_id];
    int block_size = 1 << order;

    if (page_idx < 0 || page_idx + block_size > PAGES_PER_ZONE) {
        return -3;
    }

    if ((page_idx & ((1 << order) - 1)) != 0) {
        return -4;
    }

    for (int i = 0; i < block_size; i++) {
        if (z->bitmap[page_idx + i] == 2) {
            return -5;
        }
        if (z->bitmap[page_idx + i] == 0) {
            return -6;
        }
    }

    for (int i = 0; i < block_size; i++) {
        z->bitmap[page_idx + i] = 0;
    }
    z->free_count++;

    while (order < MAX_ORDER) {
        int buddy = find_buddy(page_idx, order);

        if (buddy < 0 || buddy >= PAGES_PER_ZONE) {
            break;
        }

        int buddy_free = 1;
        int buddy_size = 1 << order;
        for (int i = 0; i < buddy_size; i++) {
            if (z->bitmap[buddy + i] != 0) {
                buddy_free = 0;
                break;
            }
        }

        if (!buddy_free) {
            break;
        }

        struct free_area *fa = &z->free_areas[order];
        int found = 0;
        for (int i = 0; i < fa->list_len; i++) {
            if (fa->list[i] == buddy) {
                for (int j = i; j < fa->list_len - 1; j++) {
                    fa->list[j] = fa->list[j + 1];
                }
                fa->list_len--;
                fa->count--;
                found = 1;
                break;
            }
        }

        if (!found) {
            break;
        }

        z->coalesce_count++;

        if (buddy < page_idx) {
            page_idx = buddy;
        }
        order++;
    }

    struct free_area *fa = &z->free_areas[order];
    fa->list[fa->list_len++] = page_idx;
    fa->count++;

    return 0;
}

int pmm_query_page(struct pmm *pmm, int zone_id, int page_idx) {
    if (zone_id < 0 || zone_id >= NUM_ZONES) {
        return -1;
    }

    struct zone *z = &pmm->zones[zone_id];

    if (page_idx < 0 || page_idx >= PAGES_PER_ZONE) {
        return -2;
    }

    return z->bitmap[page_idx];
}

void pmm_zone_stats(struct pmm *pmm, int zone_id, char *out, int out_size) {
    if (zone_id < 0 || zone_id >= NUM_ZONES) {
        snprintf(out, out_size, "ERROR: invalid_zone");
        return;
    }

    struct zone *z = &pmm->zones[zone_id];
    int allocated = 0;
    int reserved = 0;
    int free_pages = 0;

    for (int i = 0; i < PAGES_PER_ZONE; i++) {
        if (z->bitmap[i] == 1) allocated++;
        else if (z->bitmap[i] == 2) reserved++;
        else free_pages++;
    }

    snprintf(out, out_size, "zone=%d allocs=%d frees=%d splits=%d coalesces=%d allocated=%d reserved=%d free=%d",
            zone_id, z->alloc_count, z->free_count, z->split_count, z->coalesce_count,
            allocated, reserved, free_pages);
}

void pmm_global_stats(struct pmm *pmm, char *out, int out_size) {
    int total_alloc = 0, total_free = 0, total_split = 0, total_coal = 0;
    int total_allocated = 0, total_reserved = 0;

    for (int z = 0; z < NUM_ZONES; z++) {
        total_alloc += pmm->zones[z].alloc_count;
        total_free += pmm->zones[z].free_count;
        total_split += pmm->zones[z].split_count;
        total_coal += pmm->zones[z].coalesce_count;

        for (int i = 0; i < PAGES_PER_ZONE; i++) {
            if (pmm->zones[z].bitmap[i] == 1) total_allocated++;
            else if (pmm->zones[z].bitmap[i] == 2) total_reserved++;
        }
    }

    int usable = pmm->total_pages - total_reserved;

    snprintf(out, out_size, "total=%d usable=%d reserved=%d allocs=%d frees=%d splits=%d coalesces=%d",
            pmm->total_pages, usable, total_reserved,
            total_alloc, total_free, total_split, total_coal);
}

int pmm_freelist_count(struct pmm *pmm, int zone_id, int order) {
    if (zone_id < 0 || zone_id >= NUM_ZONES) {
        return -1;
    }
    if (order < 0 || order > MAX_ORDER) {
        return -2;
    }

    return pmm->zones[zone_id].free_areas[order].count;
}
EOF

cat > /app/src/main.c << 'EOF'
#include "pmm.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_LINE 256

int main(void) {
    struct pmm pmm;
    char line[MAX_LINE];
    char cmd[32];
    int arg1, arg2, arg3;
    int result;
    char stats_buf[256];

    pmm_init(&pmm);

    while (fgets(line, sizeof(line), stdin) != nullptr) {
        int nargs = sscanf(line, "%31s %d %d %d", cmd, &arg1, &arg2, &arg3);

        if (nargs < 1) {
            continue;
        }

        if (strcmp(cmd, "region") == 0 && nargs >= 4) {
            result = pmm_add_region(&pmm, arg1, arg2, arg3);
            if (result == 0) {
                printf("OK\n");
            } else {
                printf("ERROR: region_limit\n");
            }
        } else if (strcmp(cmd, "alloc") == 0 && nargs >= 3) {
            result = pmm_alloc_pages(&pmm, arg1, arg2);
            if (result >= 0) {
                printf("PAGE=%d\n", result);
            } else if (result == -1) {
                printf("ERROR: invalid_zone\n");
            } else if (result == -2) {
                printf("ERROR: invalid_order\n");
            } else {
                printf("ERROR: no_memory\n");
            }
        } else if (strcmp(cmd, "free") == 0 && nargs >= 4) {
            result = pmm_free_pages(&pmm, arg1, arg2, arg3);
            if (result == 0) {
                printf("OK\n");
            } else if (result == -1) {
                printf("ERROR: invalid_zone\n");
            } else if (result == -2) {
                printf("ERROR: invalid_order\n");
            } else if (result == -3) {
                printf("ERROR: invalid_page\n");
            } else if (result == -4) {
                printf("ERROR: unaligned\n");
            } else if (result == -5) {
                printf("ERROR: reserved\n");
            } else if (result == -6) {
                printf("ERROR: double_free\n");
            } else {
                printf("ERROR: unknown\n");
            }
        } else if (strcmp(cmd, "query") == 0 && nargs >= 3) {
            result = pmm_query_page(&pmm, arg1, arg2);
            if (result == -1) {
                printf("ERROR: invalid_zone\n");
            } else if (result == -2) {
                printf("ERROR: invalid_page\n");
            } else if (result == 0) {
                printf("FREE\n");
            } else if (result == 1) {
                printf("ALLOCATED\n");
            } else if (result == 2) {
                printf("RESERVED\n");
            } else {
                printf("UNKNOWN\n");
            }
        } else if (strcmp(cmd, "zstats") == 0 && nargs >= 2) {
            pmm_zone_stats(&pmm, arg1, stats_buf, sizeof(stats_buf));
            printf("%s\n", stats_buf);
        } else if (strcmp(cmd, "stats") == 0) {
            pmm_global_stats(&pmm, stats_buf, sizeof(stats_buf));
            printf("%s\n", stats_buf);
        } else if (strcmp(cmd, "freelist") == 0 && nargs >= 3) {
            result = pmm_freelist_count(&pmm, arg1, arg2);
            if (result == -1) {
                printf("ERROR: invalid_zone\n");
            } else if (result == -2) {
                printf("ERROR: invalid_order\n");
            } else {
                printf("COUNT=%d\n", result);
            }
        }

        fflush(stdout);
    }

    return 0;
}
EOF
