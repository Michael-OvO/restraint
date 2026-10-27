/* pmm.c - Physical Memory Manager implementation
 * Original author: K. Werner, 1998
 * Last modified: 2003
 */

#include "pmm.h"
#include <stdio.h>
#include <string.h>

static int find_buddy(pg, ord)
    int pg;
    int ord;
{
    return pg ^ (1 << ord);
}

int pmm_init(pmm)
    struct pmm *pmm;
{
    int z, i, o;

    pmm->region_count = 0;
    pmm->total_pages = 0;
    pmm->usable_pages = 0;
    pmm->reserved_pages = 0;

    for (z = 0; z < NUM_ZONES; z++) {
        pmm->zones[z].id = z;
        pmm->zones[z].start_page = z * PAGES_PER_ZONE;
        pmm->zones[z].page_count = PAGES_PER_ZONE;
        pmm->zones[z].alloc_count = 0;
        pmm->zones[z].free_count = 0;
        pmm->zones[z].split_count = 0;
        pmm->zones[z].coalesce_count = 0;

        for (i = 0; i < PAGES_PER_ZONE; i++) {
            pmm->zones[z].bitmap[i] = 0;
        }

        for (o = 0; o <= MAX_ORDER; o++) {
            pmm->zones[z].free_areas[o].count = 0;
            pmm->zones[z].free_areas[o].list_len = 0;
        }

        for (i = 0; i < PAGES_PER_ZONE; i += (1 << MAX_ORDER)) {
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

int pmm_add_region(pmm, base, count, type)
    struct pmm *pmm;
    unsigned int base;
    unsigned int count;
    int type;
{
    int r, z, i, pg;

    if (pmm->region_count >= 16) {
        return -1;
    }

    r = pmm->region_count++;
    pmm->regions[r].base_page = base;
    pmm->regions[r].page_count = count;
    pmm->regions[r].type = type;

    if (type != REGION_USABLE) {
        for (pg = base; pg < base + count; pg++) {
            z = pg / PAGES_PER_ZONE;
            if (z < NUM_ZONES) {
                i = pg % PAGES_PER_ZONE;
                pmm->zones[z].bitmap[i] = 2;
            }
        }
        pmm->reserved_pages += count;
    }

    return 0;
}

int pmm_alloc_pages(pmm, zone_id, order)
    struct pmm *pmm;
    int zone_id;
    int order;
{
    struct zone *z;
    struct free_area *fa;
    int cur_order, page_idx, i, j;
    int block_size;

    if (zone_id < 0 || zone_id >= NUM_ZONES) {
        return -1;
    }
    if (order < 0 || order > MAX_ORDER) {
        return -2;
    }

    z = &pmm->zones[zone_id];

    for (cur_order = order; cur_order <= MAX_ORDER; cur_order++) {
        fa = &z->free_areas[cur_order];
        if (fa->count > 0) {
            page_idx = fa->list[0];

            for (j = 0; j < fa->list_len - 1; j++) {
                fa->list[j] = fa->list[j + 1];
            }
            fa->list_len--;
            fa->count--;

            while (cur_order > order) {
                cur_order--;
                z->split_count++;

                int buddy = page_idx + (1 << cur_order);
                struct free_area *lower = &z->free_areas[cur_order];
                lower->list[lower->list_len++] = buddy;
            }

            block_size = 1 << order;
            for (i = 0; i < block_size; i++) {
                z->bitmap[page_idx + i] = 1;
            }

            z->alloc_count++;
            return page_idx;
        }
    }

    return -3;
}

int pmm_free_pages(pmm, zone_id, page_idx, order)
    struct pmm *pmm;
    int zone_id;
    int page_idx;
    int order;
{
    struct zone *z;
    int block_size, i, buddy, found;
    struct free_area *fa;

    if (zone_id < 0 || zone_id >= NUM_ZONES) {
        return -1;
    }
    if (order < 0 || order > MAX_ORDER) {
        return -2;
    }

    z = &pmm->zones[zone_id];
    block_size = 1 << order;

    if (page_idx < 0 || page_idx + block_size > PAGES_PER_ZONE) {
        return -3;
    }

    if ((page_idx & ((1 << order) - 1)) != 0) {
        return -4;
    }

    for (i = 0; i < block_size; i++) {
        if (z->bitmap[page_idx + i] == 2) {
            return -5;
        }
        if (z->bitmap[page_idx + i] == 0) {
            return -6;
        }
    }

    for (i = 0; i < block_size; i++) {
        z->bitmap[page_idx + i] = 0;
    }
    z->free_count++;

    while (order < MAX_ORDER) {
        buddy = find_buddy(page_idx, order);

        if (buddy < 0 || buddy >= PAGES_PER_ZONE) {
            break;
        }

        int buddy_free = 1;
        int buddy_size = 1 << order;
        for (i = 0; i < buddy_size; i++) {
            if (z->bitmap[buddy + i] != 0) {
                buddy_free = 0;
                break;
            }
        }

        if (!buddy_free) {
            break;
        }

        fa = &z->free_areas[order];
        found = 0;
        for (i = 0; i < fa->list_len; i++) {
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

        if (buddy < page_idx) {
            page_idx = buddy;
        }
        order++;
    }

    fa = &z->free_areas[order];
    fa->list[fa->list_len++] = page_idx;
    fa->count++;

    return 0;
}

int pmm_query_page(pmm, zone_id, page_idx)
    struct pmm *pmm;
    int zone_id;
    int page_idx;
{
    struct zone *z;

    if (zone_id < 0 || zone_id >= NUM_ZONES) {
        return -1;
    }

    z = &pmm->zones[zone_id];

    if (page_idx < 0 || page_idx >= PAGES_PER_ZONE) {
        return -2;
    }

    return z->bitmap[page_idx];
}

void pmm_zone_stats(pmm, zone_id, out, out_size)
    struct pmm *pmm;
    int zone_id;
    char *out;
    int out_size;
{
    struct zone *z;
    int i, allocated, reserved, free_pages;

    if (zone_id < 0 || zone_id >= NUM_ZONES) {
        sprintf(out, "ERROR: invalid_zone");
        return;
    }

    z = &pmm->zones[zone_id];
    allocated = 0;
    reserved = 0;
    free_pages = 0;

    for (i = 0; i < PAGES_PER_ZONE; i++) {
        if (z->bitmap[i] == 1) allocated++;
        else if (z->bitmap[i] == 2) reserved++;
        else free_pages++;
    }

    sprintf(out, "zone=%d allocs=%d frees=%d splits=%d coalesces=%d allocated=%d reserved=%d free=%d",
            zone_id, z->alloc_count, z->free_count, z->split_count, z->coalesce_count,
            allocated, reserved, free_pages);
}

void pmm_global_stats(pmm, out, out_size)
    struct pmm *pmm;
    char *out;
    int out_size;
{
    int z, total_alloc, total_free, total_split, total_coal;
    int total_allocated, total_reserved, total_free_pages;

    total_alloc = total_free = total_split = total_coal = 0;
    total_allocated = total_reserved = total_free_pages = 0;

    for (z = 0; z < NUM_ZONES; z++) {
        total_alloc += pmm->zones[z].alloc_count;
        total_free += pmm->zones[z].free_count;
        total_split += pmm->zones[z].split_count;
        total_coal += pmm->zones[z].coalesce_count;

        for (int i = 0; i < PAGES_PER_ZONE; i++) {
            if (pmm->zones[z].bitmap[i] == 1) total_allocated++;
            else if (pmm->zones[z].bitmap[i] == 2) total_reserved++;
            else total_free_pages++;
        }
    }

    sprintf(out, "total=%d usable=%d reserved=%d allocs=%d frees=%d splits=%d coalesces=%d",
            pmm->total_pages, total_free_pages + total_allocated, total_reserved,
            total_alloc, total_free, total_split, total_coal);
}

int pmm_freelist_count(pmm, zone_id, order)
    struct pmm *pmm;
    int zone_id;
    int order;
{
    if (zone_id < 0 || zone_id >= NUM_ZONES) {
        return -1;
    }
    if (order < 0 || order > MAX_ORDER) {
        return -2;
    }

    return pmm->zones[zone_id].free_areas[order].count;
}
