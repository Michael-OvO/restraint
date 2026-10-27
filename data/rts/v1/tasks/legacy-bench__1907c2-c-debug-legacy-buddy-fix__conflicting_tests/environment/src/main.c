/* main.c - Physical Memory Manager command interface
 * Test driver for PMM subsystem
 */

#include "pmm.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_LINE 256

int main(void)
{
    struct pmm pmm;
    char line[MAX_LINE];
    char cmd[32];
    int arg1, arg2, arg3;
    int result;
    char stats_buf[256];

    pmm_init(&pmm);

    while (fgets(line, sizeof(line), stdin) != NULL) {
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
