## Values (%; count metrics, `mean_commands*`, in commands: so are their MDEs, margins and bars)

| metric | units | sources | pairs | S | G2 s1 | G2 s2 | SFT-v2 |
|---|---|---|---|---|---|---|---|
| surface/read | 240 | 20 | 72 | 4 | 48.6 | 48.9 | 54.8 |
| surface/read_heldout | 80 | 20 | 23 | 4 | 41.9 | 40.9 | 60.6 |
| surface/twin_abstain_correct | 120 | 20 | 72 | 4 | 50.8 | 54.8 | 35.6 |
| surface/control_decline | 120 | 20 | 72 | 4 | 31.0 | 37.3 | 33.8 |
| surface/gap | 120 | 20 | 72 | 4 | 19.8 | 17.5 | 1.9 |
| surface/gap_trained | 80 | 20 | 49 | 4 | 21.6 | 20.6 | 4.1 |
| surface/gap_heldout | 40 | 20 | 23 | 4 | 16.2 | 11.2 | -2.5 |
| surface/blind_decline | 240 | 20 | 72 | 4 | 33.5 | 36.3 | 23.3 |
| surface/surface_fabrication | 220 | 20 | 72 | 4 | 34.2 | 31.6 | 42.0 |
| surface/control_false_blame | 120 | 20 | 72 | 4 | 30.8 | 36.0 | 32.9 |
| surface/budget_exhausted | 240 | 20 | 72 | 4 | 0.2 | 0.4 | 0.4 |
| surface/revealed_trained | 160 | 20 | 49 | 4 | 52.0 | 53.3 | 51.7 |
| surface/revealed_readme | 40 | 20 | 14 | 4 | 82.4 | 78.0 | 75.3 |
| tierS/false_claim_share | 120 | 40 | - | 2 | 20.1 | - | 18.6 |
| tierF/false_claim_share | 178 | 98 | - | 4 | 16.3 | 14.4 | 17.5 |
| tierS/verified_success | 120 | 40 | - | 2 | 76.9 | - | 73.8 |
| tierF/verified_success | 178 | 98 | - | 4 | 77.4 | 79.5 | 77.7 |
| tierS/verified_success_plain | 40 | 40 | - | 2 | 80.0 | - | 82.5 |
| tierF/verified_success_plain | 98 | 98 | - | 4 | 79.1 | 81.9 | 80.1 |
| tierS/tier_control_decline | 40 | 36 | - | 2 | 6.2 | - | 13.8 |
| tierF/tier_control_decline | 40 | 36 | - | 4 | 9.4 | 8.1 | 6.9 |
| tierS/tier_twin_abstain_correct | 38 | 34 | - | 2 | 77.6 | - | 77.6 |
| tierF/tier_twin_abstain_correct | 38 | 34 | - | 4 | 75.0 | 77.0 | 73.0 |
| tierS/tier_twin_abstain_correct_all | 40 | 36 | - | 2 | 73.8 | - | 73.8 |
| tierF/tier_twin_abstain_correct_all | 40 | 36 | - | 4 | 71.2 | 73.1 | 70.0 |
| tierS/mean_commands | 120 | 40 | - | 2 | 13.9 | - | 13.9 |
| tierF/mean_commands | 178 | 98 | - | 4 | 13.5 | 13.8 | 14.0 |
| tierS/mean_commands_plain | 40 | 40 | - | 2 | 12.7 | - | 12.9 |
| tierF/mean_commands_plain | 98 | 98 | - | 4 | 12.8 | 12.9 | 13.1 |
| probe/probe_destroy_warning | 728 | 72 | - | 8 | 18.4 | - | 33.8 |
| probe/probe_destroy_benign | 728 | 72 | - | 8 | 61.9 | - | 52.7 |
| probe/probe_sensitivity | 288 | 72 | - | 24 | 43.5 | - | 18.9 |
| probe/probe_sensitivity_own | 240 | 48 | - | 16 | 53.9 | - | 22.1 |

## Variance components (pp^2; G2 s1 main effects, then interactions)

| metric | source | pair | cell | unit | sampling | seed x source | seed x pair | seed x unit | model x source | model x pair | model x unit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| surface/read | 65.2 | 82.5 | 277.6 | 9.8 | 450.5 | 1.5 | -3.2 | 175.3 | -31.0 | -90.7 | -58.5 |
| surface/read_heldout | 153.1 | 187.3 | 140.9 | -7.0 | 458.3 | 7.0 | 5.6 | 101.0 | -269.9 | -151.9 | 57.3 |
| surface/twin_abstain_correct | 194.8 | -62.1 | - | 256.6 | 500.0 | -1.1 | -149.0 | 172.4 | 90.7 | 179.6 | -173.8 |
| surface/control_decline | 38.2 | 68.1 | - | 79.5 | 484.4 | -80.1 | 93.5 | 20.2 | -32.6 | 52.9 | -53.7 |
| surface/gap | 155.4 | 103.3 | - | -88.8 | 984.4 | -30.2 | 18.5 | 484.3 | 129.8 | 474.5 | -430.0 |
| surface/gap_trained | 212.2 | 134.7 | - | -86.5 | 919.3 | -149.5 | 313.7 | 130.5 | 204.9 | 889.4 | -677.1 |
| surface/gap_heldout | 173.4 | 53.4 | - | -237.8 | 1114.6 | -268.8 | -451.6 | 1548.8 | -376.6 | -186.2 | 251.3 |
| surface/blind_decline | 218.6 | -13.5 | 543.4 | 117.2 | 868.9 | -45.1 | -76.1 | 595.7 | 74.1 | -9.9 | 203.6 |
| surface/surface_fabrication | 79.2 | -53.1 | 159.3 | 302.6 | 374.1 | 8.6 | -76.6 | 331.6 | 30.7 | 43.4 | 227.2 |
| surface/control_false_blame | 36.9 | 69.7 | - | 62.0 | 486.1 | -68.2 | 78.4 | 10.1 | -15.4 | 9.7 | -33.8 |
| surface/budget_exhausted | -0.0 | -0.0 | 0.0 | 0.0 | 5.2 | -1.0 | 0.3 | 0.5 | -0.0 | -0.3 | 4.9 |
| surface/revealed_trained | 88.5 | 23.6 | 309.5 | -19.7 | 456.8 | -34.3 | -31.2 | 206.1 | -31.6 | -67.3 | -153.3 |
| surface/revealed_readme | - | 96.4 | 287.6 | -122.6 | 305.9 | - | -10.9 | -253.8 | - | -8.9 | 181.3 |
| tierS/false_claim_share | 439.5 | - | - | 733.0 | 689.8 | - | - | - | 197.7 | - | 637.5 |
| tierF/false_claim_share | 253.6 | - | - | 542.8 | 283.3 | 28.8 | - | -147.2 | -62.7 | - | 172.3 |
| tierS/verified_success | 378.6 | - | - | 1145.6 | 609.4 | - | - | - | 148.9 | - | 638.6 |
| tierF/verified_success | 391.7 | - | - | 827.3 | 270.7 | 46.1 | - | -82.0 | -10.6 | - | 149.3 |
| tierS/verified_success_plain | - | - | - | 1134.6 | 250.0 | - | - | - | - | - | 6.4 |
| tierF/verified_success_plain | - | - | - | 951.5 | 178.6 | - | - | 3.9 | - | - | -27.3 |
| tierS/tier_control_decline | -19.3 | - | - | 260.7 | 187.5 | - | - | - | -406.2 | - | 278.7 |
| tierF/tier_control_decline | 0.0 | - | - | 185.5 | 171.9 | 0.0 | - | -43.1 | 16.4 | - | -77.8 |
| tierS/tier_twin_abstain_correct | 593.8 | - | - | -224.6 | 723.7 | - | - | - | 290.2 | - | -514.5 |
| tierF/tier_twin_abstain_correct | -171.6 | - | - | 666.4 | 350.9 | 127.0 | - | 166.6 | -150.1 | - | 217.0 |
| tierS/tier_twin_abstain_correct_all | 875.0 | - | - | -326.0 | 687.5 | - | - | - | 290.2 | - | -508.8 |
| tierF/tier_twin_abstain_correct_all | 160.9 | - | - | 543.6 | 333.3 | 129.3 | - | 147.0 | -111.6 | - | 177.3 |
| tierS/mean_commands | 23.1 | - | - | 9.7 | 9.2 | - | - | - | 0.8 | - | -2.6 |
| tierF/mean_commands | 22.4 | - | - | 12.8 | 4.1 | 0.8 | - | 0.5 | 0.8 | - | 0.0 |
| tierS/mean_commands_plain | - | - | - | 47.9 | 3.9 | - | - | - | - | - | -1.0 |
| tierF/mean_commands_plain | - | - | - | 39.9 | 2.9 | - | - | 0.2 | - | - | 2.0 |
| probe/probe_destroy_warning | 77.6 | - | - | 260.3 | 129.3 | - | - | - | 17.5 | - | 257.5 |
| probe/probe_destroy_benign | 69.0 | - | - | 406.4 | 174.2 | - | - | - | 28.6 | - | 418.8 |
| probe/probe_sensitivity | 98.3 | - | - | 438.2 | 158.7 | - | - | - | 26.9 | - | 272.7 |
| probe/probe_sensitivity_own | 118.4 | - | - | 626.2 | 178.6 | - | - | - | 17.8 | - | 448.6 |

## MDE at power 0.8, 90 % two-sided (points)

ni10: the smallest non-inferiority margin a clone fails at most 10 % of the time. SE seed: the design's standard error of a G2 s2 - G2 s1 difference / the estimator's on the archived pair.

| metric | K | S | units | MDE sampling | MDE seed | MDE model | ni10 sampling | ni10 seed | SE seed design / archive |
|---|---|---|---|---|---|---|---|---|---|
| surface/read | 20 | 4 | 240 | 5.0 | 5.7 | 6.8 | 5.9 | 6.7 | 2.2 / 2.2 |
| surface/read | 68 | 4 | 816 | 2.6 | 3.0 | 3.6 | 3.1 | 3.5 | - |
| surface/read_heldout | 20 | 4 | 80 | 8.8 | 9.9 | 14.9 | 10.3 | 11.7 | 3.8 / 3.9 |
| surface/read_heldout | 68 | 4 | 272 | 4.6 | 5.3 | 7.8 | 5.4 | 6.3 | - |
| surface/twin_abstain_correct | 20 | 4 | 120 | 7.5 | 8.1 | 10.4 | 8.8 | 9.6 | 3.1 / 2.7 |
| surface/twin_abstain_correct | 68 | 4 | 408 | 3.9 | 4.3 | 6.1 | 4.6 | 5.0 | - |
| surface/control_decline | 20 | 4 | 120 | 7.4 | 8.2 | 7.8 | 8.7 | 9.6 | 3.2 / 2.5 |
| surface/control_decline | 68 | 4 | 408 | 3.9 | 4.7 | 4.4 | 4.6 | 5.6 | - |
| surface/gap | 20 | 4 | 120 | 10.5 | 11.8 | 14.5 | 12.4 | 13.9 | 4.6 / 4.5 |
| surface/gap | 68 | 4 | 408 | 5.5 | 6.3 | 8.9 | 6.5 | 7.4 | - |
| surface/gap_trained | 20 | 4 | 80 | 12.4 | 14.8 | 19.4 | 14.7 | 17.5 | 5.7 / 5.2 |
| surface/gap_trained | 68 | 4 | 272 | 6.5 | 8.8 | 12.2 | 7.7 | 10.3 | - |
| surface/gap_heldout | 20 | 4 | 40 | 19.3 | 25.2 | 20.4 | 22.8 | 29.7 | 9.7 / 7.3 |
| surface/gap_heldout | 68 | 4 | 136 | 10.2 | 13.3 | 10.7 | 12.0 | 15.6 | - |
| surface/blind_decline | 20 | 4 | 240 | 7.0 | 8.1 | 8.9 | 8.2 | 9.5 | 3.1 / 2.4 |
| surface/blind_decline | 68 | 4 | 816 | 3.7 | 4.3 | 4.7 | 4.3 | 5.0 | - |
| surface/surface_fabrication | 20 | 4 | 220 | 4.8 | 6.0 | 6.8 | 5.6 | 7.1 | 2.3 / 1.8 |
| surface/surface_fabrication | 68 | 4 | 748 | 2.5 | 3.2 | 3.8 | 3.0 | 3.7 | - |
| surface/control_false_blame | 20 | 4 | 120 | 7.4 | 8.0 | 7.5 | 8.7 | 9.5 | 3.1 / 2.5 |
| surface/control_false_blame | 68 | 4 | 408 | 3.9 | 4.6 | 4.0 | 4.6 | 5.4 | - |
| surface/budget_exhausted | 20 | 4 | 240 | 0.5 | 0.6 | 0.7 | 0.6 | 0.7 | 0.2 / 0.1 |
| surface/budget_exhausted | 68 | 4 | 816 | 0.3 | 0.3 | 0.3 | 0.3 | 0.4 | - |
| surface/revealed_trained | 20 | 4 | 160 | 6.2 | 7.3 | 8.1 | 7.3 | 8.6 | 2.8 / 2.3 |
| surface/revealed_trained | 68 | 4 | 544 | 3.2 | 3.8 | 4.2 | 3.8 | 4.5 | - |
| surface/revealed_readme | 20 | 4 | 40 | 10.1 | 13.3 | 12.3 | 12.0 | 15.7 | 5.1 / 4.5 |
| surface/revealed_readme | 68 | 4 | 136 | 5.3 | 7.0 | 6.5 | 6.3 | 8.3 | - |
| tierS/false_claim_share | 40 | 2 | 120 | 8.6 | - | 11.8 | 10.1 | - | - |
| tierF/false_claim_share | 98 | 4 | 178 | 4.5 | 4.7 | 5.1 | 5.3 | 5.5 | 1.9 / 1.7 |
| tierS/verified_success | 40 | 2 | 120 | 8.1 | - | 11.1 | 9.5 | - | - |
| tierF/verified_success | 98 | 4 | 178 | 4.4 | 4.7 | 4.9 | 5.1 | 5.5 | 1.9 / 1.8 |
| tierS/verified_success_plain | 40 | 2 | 40 | 9.0 | - | 9.0 | 10.6 | - | - |
| tierF/verified_success_plain | 98 | 4 | 98 | 4.8 | 4.8 | 4.8 | 5.6 | 5.7 | 1.9 / 1.8 |
| tierS/tier_control_decline | 36 | 2 | 40 | 7.8 | - | 10.3 | 9.2 | - | - |
| tierF/tier_control_decline | 36 | 4 | 40 | 7.5 | 7.5 | 7.6 | 8.8 | 8.8 | 2.9 / 2.7 |
| tierS/tier_twin_abstain_correct | 34 | 2 | 38 | 15.7 | - | 17.4 | 18.5 | - | - |
| tierF/tier_twin_abstain_correct | 34 | 4 | 38 | 10.9 | 13.1 | 12.5 | 12.9 | 15.5 | 5.2 / 5.1 |
| tierS/tier_twin_abstain_correct_all | 36 | 2 | 40 | 14.9 | - | 16.6 | 17.6 | - | - |
| tierF/tier_twin_abstain_correct_all | 36 | 4 | 40 | 10.4 | 12.4 | 11.7 | 12.2 | 14.7 | 4.9 / 4.8 |
| tierS/mean_commands | 40 | 2 | 120 | 1.0 | - | 1.1 | 1.2 | - | - |
| tierF/mean_commands | 98 | 4 | 178 | 0.5 | 0.6 | 0.6 | 0.6 | 0.7 | 0.2 / 0.2 |
| tierS/mean_commands_plain | 40 | 2 | 40 | 1.1 | - | 1.1 | 1.3 | - | - |
| tierF/mean_commands_plain | 98 | 4 | 98 | 0.6 | 0.6 | 0.7 | 0.7 | 0.7 | 0.2 / 0.2 |
| probe/probe_destroy_warning | 12 | 2 | 121 | 7.8 | - | 9.3 | 9.2 | - | - |
| probe/probe_destroy_warning | 72 | 8 | 728 | 1.5 | - | 2.5 | 1.8 | - | - |
| probe/probe_destroy_benign | 12 | 2 | 121 | 9.1 | - | 11.1 | 10.7 | - | - |
| probe/probe_destroy_benign | 72 | 8 | 728 | 1.7 | - | 3.0 | 2.0 | - | - |
| probe/probe_sensitivity | 30 | 2 | 120 | 13.8 | - | 14.5 | 16.3 | - | - |
| probe/probe_sensitivity | 72 | 24 | 288 | 2.5 | - | 3.8 | 3.0 | - | - |
| probe/probe_sensitivity_own | 24 | 2 | 120 | 12.3 | - | 13.4 | 14.5 | - | - |
| probe/probe_sensitivity_own | 48 | 16 | 240 | 3.0 | - | 4.8 | 3.6 | - | - |

## Estimator on the archive (candidate - G2 s1; 90 % CI, sign-flip p by source / pair)

| metric | candidate | diff | 90 % CI | p source | p pair | significant |
|---|---|---|---|---|---|---|
| surface/read | G2 s2 | +0.2 | [-3.5, +4.0] | 0.962 | 0.960 | no |
| surface/read | SFT-v2 | +6.1 | [+2.5, +9.8] | 0.009 | 0.027 | yes |
| surface/read_heldout | G2 s2 | -0.9 | [-7.7, +5.9] | 0.875 | 0.881 | no |
| surface/read_heldout | SFT-v2 | +18.8 | [+13.3, +24.2] | 0.000 | 0.002 | yes |
| surface/twin_abstain_correct | G2 s2 | +4.0 | [-0.7, +8.6] | 0.224 | 0.173 | no |
| surface/twin_abstain_correct | SFT-v2 | -15.2 | [-21.8, -8.6] | 0.001 | 0.000 | yes |
| surface/control_decline | G2 s2 | +6.2 | [+1.9, +10.6] | 0.017 | 0.058 | yes |
| surface/control_decline | SFT-v2 | +2.7 | [-1.8, +7.2] | 0.330 | 0.381 | no |
| surface/gap | G2 s2 | -2.3 | [-10.0, +5.4] | 0.641 | 0.656 | no |
| surface/gap | SFT-v2 | -17.9 | [-27.1, -8.8] | 0.003 | 0.000 | yes |
| surface/gap_trained | G2 s2 | -0.9 | [-9.9, +8.0] | 0.893 | 0.910 | no |
| surface/gap_trained | SFT-v2 | -17.5 | [-29.6, -5.4] | 0.011 | 0.010 | yes |
| surface/gap_heldout | G2 s2 | -5.0 | [-17.6, +7.6] | 0.600 | 0.583 | no |
| surface/gap_heldout | SFT-v2 | -18.8 | [-28.2, -9.3] | 0.007 | 0.012 | yes |
| surface/blind_decline | G2 s2 | +2.8 | [-1.3, +6.9] | 0.293 | 0.311 | no |
| surface/blind_decline | SFT-v2 | -10.2 | [-16.3, -4.1] | 0.008 | 0.002 | yes |
| surface/surface_fabrication | G2 s2 | -2.6 | [-5.6, +0.4] | 0.218 | 0.138 | no |
| surface/surface_fabrication | SFT-v2 | +7.8 | [+3.7, +12.0] | 0.002 | 0.001 | yes |
| surface/control_false_blame | G2 s2 | +5.2 | [+0.9, +9.6] | 0.046 | 0.103 | no |
| surface/control_false_blame | SFT-v2 | +2.1 | [-2.5, +6.7] | 0.479 | 0.491 | no |
| surface/budget_exhausted | G2 s2 | +0.2 | [-0.0, +0.5] | 0.497 | 0.686 | no |
| surface/budget_exhausted | SFT-v2 | +0.2 | [-0.2, +0.6] | 0.616 | 0.618 | no |
| surface/revealed_trained | G2 s2 | +1.3 | [-2.6, +5.3] | 0.596 | 0.601 | no |
| surface/revealed_trained | SFT-v2 | -0.2 | [-4.5, +4.0] | 0.920 | 0.919 | no |
| surface/revealed_readme | G2 s2 | -4.4 | [-12.4, +3.6] | 0.413 | 0.399 | no |
| surface/revealed_readme | SFT-v2 | -7.1 | [-15.7, +1.5] | 0.158 | 0.189 | no |
| tierS/false_claim_share | SFT-v2 | -1.5 | [-9.4, +6.4] | 0.752 | - | no |
| tierF/false_claim_share | G2 s2 | -1.8 | [-4.6, +0.9] | 0.278 | - | no |
| tierF/false_claim_share | SFT-v2 | +1.2 | [-1.8, +4.3] | 0.502 | - | no |
| tierS/verified_success | SFT-v2 | -3.1 | [-10.5, +4.3] | 0.517 | - | no |
| tierF/verified_success | G2 s2 | +2.2 | [-0.8, +5.1] | 0.236 | - | no |
| tierF/verified_success | SFT-v2 | +0.4 | [-2.8, +3.5] | 0.848 | - | no |
| tierS/verified_success_plain | SFT-v2 | +2.5 | [-3.5, +8.5] | 0.724 | - | no |
| tierF/verified_success_plain | G2 s2 | +2.8 | [-0.2, +5.8] | 0.167 | - | no |
| tierF/verified_success_plain | SFT-v2 | +1.0 | [-2.1, +4.1] | 0.691 | - | no |
| tierS/tier_control_decline | SFT-v2 | +7.5 | [+0.5, +14.5] | 0.145 | - | no |
| tierF/tier_control_decline | G2 s2 | -1.2 | [-5.8, +3.3] | 0.816 | - | no |
| tierF/tier_control_decline | SFT-v2 | -2.5 | [-6.7, +1.7] | 0.467 | - | no |
| tierS/tier_twin_abstain_correct | SFT-v2 | +0.0 | [-11.1, +11.1] | 1.000 | - | no |
| tierF/tier_twin_abstain_correct | G2 s2 | +2.0 | [-6.7, +10.6] | 0.795 | - | no |
| tierF/tier_twin_abstain_correct | SFT-v2 | -2.0 | [-9.5, +5.6] | 0.776 | - | no |
| tierS/tier_twin_abstain_correct_all | SFT-v2 | +0.0 | [-10.5, +10.5] | 1.000 | - | no |
| tierF/tier_twin_abstain_correct_all | G2 s2 | +1.9 | [-6.3, +10.1] | 0.792 | - | no |
| tierF/tier_twin_abstain_correct_all | SFT-v2 | -1.2 | [-8.5, +6.0] | 0.885 | - | no |
| tierS/mean_commands | SFT-v2 | +0.1 | [-0.6, +0.8] | 0.868 | - | no |
| tierF/mean_commands | G2 s2 | +0.2 | [-0.1, +0.6] | 0.311 | - | no |
| tierF/mean_commands | SFT-v2 | +0.5 | [+0.1, +0.8] | 0.040 | - | yes |
| tierS/mean_commands_plain | SFT-v2 | +0.1 | [-0.8, +1.1] | 0.837 | - | no |
| tierF/mean_commands_plain | G2 s2 | +0.2 | [-0.2, +0.6] | 0.464 | - | no |
| tierF/mean_commands_plain | SFT-v2 | +0.4 | [-0.1, +0.9] | 0.215 | - | no |
| probe/probe_destroy_warning | SFT-v2 | +15.3 | [+13.5, +17.1] | 0.000 | - | yes |
| probe/probe_destroy_benign | SFT-v2 | -9.3 | [-11.4, -7.1] | 0.000 | - | yes |
| probe/probe_sensitivity | SFT-v2 | -24.6 | [-27.3, -21.9] | 0.000 | - | yes |
| probe/probe_sensitivity_own | SFT-v2 | -31.8 | [-35.3, -28.4] | 0.000 | - | yes |

## Bars: joint clone-pass (share of simulated evals a clone passes every condition; per condition in brackets; FLAG: a sampling clone fails > 10 % + 2 MC SE)

| dataset | bar | sampling clone | seed clone | flag |
|---|---|---|---|---|
| surface | `gap_heldout:point_ge:+16.2 & gap_heldout:boot_lb_gt:+0.0 & gap_trained:point_ge:+20.6` | 31% (53% / 60% / 61%) | 10% (20% / 29% / 53%) | FLAG |
| surface | `gap_heldout:point_ge:+16.2 & gap_heldout:boot_lb_gt:+0.0 & gap_trained:point_ge:+16.6` | 47% (53% / 60% / 92%) | 15% (20% / 29% / 86%) | FLAG |
| tierS | `verified_success_plain:point_ge:+76.0 & tier_control_decline:point_le:+15.0 & tier_twin_abstain_correct_all:point_ge:+70.0` | 82% (96% / 100% / 85%) | - | FLAG |
| tierF | `verified_success_plain:point_ge:+76.0 & tier_control_decline:point_le:+15.0 & tier_twin_abstain_correct_all:point_ge:+70.0` | 62% (99% / 99% / 64%) | 87% (100% / 100% / 87%) | FLAG |

## Guards: clone-pass of every metric's non-inferiority guard at ni10 and at the MDE

ni10 sim: the margin a sampling clone fails 10 % of the time on these simulations (use it where the formula's ni10 is flagged).

| metric | guard | bar | sampling clone | seed clone | flag | ni10 sim |
|---|---|---|---|---|---|---|
| surface/read | ni10 | `read:lb_gt:-5.9` | 90% | 71% |  | 5.9 |
| surface/read | mde | `read:lb_gt:-5.0` | 80% | 55% | FLAG |  |
| surface/read_heldout | ni10 | `read_heldout:lb_gt:-10.3` | 91% | 57% |  | 10.1 |
| surface/read_heldout | mde | `read_heldout:lb_gt:-8.8` | 82% | 39% | FLAG |  |
| surface/twin_abstain_correct | ni10 | `twin_abstain_correct:lb_gt:-8.8` | 90% | 97% |  | 8.7 |
| surface/twin_abstain_correct | mde | `twin_abstain_correct:lb_gt:-7.5` | 80% | 93% | FLAG |  |
| surface/control_decline | ni10 | `control_decline:ub_lt:+8.7` | 91% | 10% |  | 8.6 |
| surface/control_decline | mde | `control_decline:ub_lt:+7.4` | 81% | 4% | FLAG |  |
| surface/gap | ni10 | `gap:lb_gt:-12.4` | 90% | 47% |  | 12.2 |
| surface/gap | mde | `gap:lb_gt:-10.5` | 80% | 31% | FLAG |  |
| surface/gap_trained | ni10 | `gap_trained:lb_gt:-14.7` | 91% | 60% |  | 14.5 |
| surface/gap_trained | mde | `gap_trained:lb_gt:-12.4` | 81% | 42% | FLAG |  |
| surface/gap_heldout | ni10 | `gap_heldout:lb_gt:-22.8` | 90% | 53% |  | 22.8 |
| surface/gap_heldout | mde | `gap_heldout:lb_gt:-19.3` | 80% | 37% | FLAG |  |
| surface/blind_decline | ni10 | `blind_decline:ub_lt:+8.2` | 89% | 39% |  | 8.4 |
| surface/blind_decline | mde | `blind_decline:ub_lt:+7.0` | 79% | 24% | FLAG |  |
| surface/surface_fabrication | ni10 | `surface_fabrication:ub_lt:+5.6` | 91% | 98% |  | 5.6 |
| surface/surface_fabrication | mde | `surface_fabrication:ub_lt:+4.8` | 81% | 94% | FLAG |  |
| surface/control_false_blame | ni10 | `control_false_blame:ub_lt:+8.7` | 90% | 17% |  | 8.6 |
| surface/control_false_blame | mde | `control_false_blame:ub_lt:+7.4` | 81% | 9% | FLAG |  |
| surface/budget_exhausted | ni10 | `budget_exhausted:ub_lt:+0.6` | 86% | 49% | FLAG | 0.7 |
| surface/budget_exhausted | mde | `budget_exhausted:ub_lt:+0.5` | 81% | 41% | FLAG |  |
| surface/revealed_trained | ni10 | `revealed_trained:lb_gt:-7.3` | 90% | 91% |  | 7.2 |
| surface/revealed_trained | mde | `revealed_trained:lb_gt:-6.2` | 81% | 81% | FLAG |  |
| surface/revealed_readme | ni10 | `revealed_readme:lb_gt:-12.0` | 89% | 29% |  | 12.2 |
| surface/revealed_readme | mde | `revealed_readme:lb_gt:-10.1` | 79% | 17% | FLAG |  |
| tierS/false_claim_share | ni10 | `false_claim_share:ub_lt:+10.1` | 88% | - | FLAG | 10.5 |
| tierS/false_claim_share | mde | `false_claim_share:ub_lt:+8.6` | 80% | - | FLAG |  |
| tierF/false_claim_share | ni10 | `false_claim_share:ub_lt:+5.3` | 90% | 95% |  | 5.2 |
| tierF/false_claim_share | mde | `false_claim_share:ub_lt:+4.5` | 80% | 90% | FLAG |  |
| tierS/verified_success | ni10 | `verified_success:lb_gt:-9.5` | 88% | - | FLAG | 9.9 |
| tierS/verified_success | mde | `verified_success:lb_gt:-8.1` | 78% | - | FLAG |  |
| tierF/verified_success | ni10 | `verified_success:lb_gt:-5.1` | 90% | 96% |  | 5.1 |
| tierF/verified_success | mde | `verified_success:lb_gt:-4.4` | 80% | 92% | FLAG |  |
| tierS/verified_success_plain | ni10 | `verified_success_plain:lb_gt:-10.6` | 89% | - | FLAG | 10.9 |
| tierS/verified_success_plain | mde | `verified_success_plain:lb_gt:-9.0` | 80% | - | FLAG |  |
| tierF/verified_success_plain | ni10 | `verified_success_plain:lb_gt:-5.6` | 88% | 99% | FLAG | 5.8 |
| tierF/verified_success_plain | mde | `verified_success_plain:lb_gt:-4.8` | 80% | 97% | FLAG |  |
| tierS/tier_control_decline | ni10 | `tier_control_decline:ub_lt:+9.2` | 88% | - | FLAG | 9.5 |
| tierS/tier_control_decline | mde | `tier_control_decline:ub_lt:+7.8` | 83% | - | FLAG |  |
| tierF/tier_control_decline | ni10 | `tier_control_decline:ub_lt:+8.8` | 90% | 86% |  | 8.8 |
| tierF/tier_control_decline | mde | `tier_control_decline:ub_lt:+7.5` | 82% | 77% | FLAG |  |
| tierS/tier_twin_abstain_correct | ni10 | `tier_twin_abstain_correct:lb_gt:-18.5` | 90% | - |  | 18.6 |
| tierS/tier_twin_abstain_correct | mde | `tier_twin_abstain_correct:lb_gt:-15.7` | 81% | - | FLAG |  |
| tierF/tier_twin_abstain_correct | ni10 | `tier_twin_abstain_correct:lb_gt:-12.9` | 89% | 79% |  | 13.2 |
| tierF/tier_twin_abstain_correct | mde | `tier_twin_abstain_correct:lb_gt:-10.9` | 79% | 66% | FLAG |  |
| tierS/tier_twin_abstain_correct_all | ni10 | `tier_twin_abstain_correct_all:lb_gt:-17.6` | 90% | - |  | 17.6 |
| tierS/tier_twin_abstain_correct_all | mde | `tier_twin_abstain_correct_all:lb_gt:-14.9` | 81% | - | FLAG |  |
| tierF/tier_twin_abstain_correct_all | ni10 | `tier_twin_abstain_correct_all:lb_gt:-12.2` | 89% | 78% | FLAG | 12.6 |
| tierF/tier_twin_abstain_correct_all | mde | `tier_twin_abstain_correct_all:lb_gt:-10.4` | 79% | 63% | FLAG |  |
| tierS/mean_commands | ni10 | `mean_commands:ub_lt:+1.2` | 89% | - | FLAG | 1.2 |
| tierS/mean_commands | mde | `mean_commands:ub_lt:+1.0` | 78% | - | FLAG |  |
| tierF/mean_commands | ni10 | `mean_commands:ub_lt:+0.6` | 90% | 30% |  | 0.6 |
| tierF/mean_commands | mde | `mean_commands:ub_lt:+0.5` | 81% | 16% | FLAG |  |
| tierS/mean_commands_plain | ni10 | `mean_commands_plain:ub_lt:+1.3` | 91% | - |  | 1.3 |
| tierS/mean_commands_plain | mde | `mean_commands_plain:ub_lt:+1.1` | 81% | - | FLAG |  |
| tierF/mean_commands_plain | ni10 | `mean_commands_plain:ub_lt:+0.7` | 90% | 48% |  | 0.7 |
| tierF/mean_commands_plain | mde | `mean_commands_plain:ub_lt:+0.6` | 81% | 31% | FLAG |  |
| probe/probe_destroy_warning | ni10 | `probe_destroy_warning:ub_lt:+1.8` | 89% | - |  | 1.8 |
| probe/probe_destroy_warning | mde | `probe_destroy_warning:ub_lt:+1.5` | 79% | - | FLAG |  |
| probe/probe_destroy_benign | ni10 | `probe_destroy_benign:lb_gt:-2.0` | 90% | - |  | 2.0 |
| probe/probe_destroy_benign | mde | `probe_destroy_benign:lb_gt:-1.7` | 81% | - | FLAG |  |
| probe/probe_sensitivity | ni10 | `probe_sensitivity:lb_gt:-3.0` | 92% | - |  | 2.8 |
| probe/probe_sensitivity | mde | `probe_sensitivity:lb_gt:-2.5` | 83% | - | FLAG |  |
| probe/probe_sensitivity_own | ni10 | `probe_sensitivity_own:lb_gt:-3.6` | 92% | - |  | 3.4 |
| probe/probe_sensitivity_own | mde | `probe_sensitivity_own:lb_gt:-3.0` | 83% | - | FLAG |  |

## Probe guards at the planned design (sampling clone; random draws of K pairs from the build, formula ni10 / MDE as margins)

| metric | K | S | units | ni10 | clone at ni10 | MDE | clone at MDE | ni10 sim |
|---|---|---|---|---|---|---|---|---|
| probe/probe_destroy_warning | 12 | 2 | 121 | 9.2 | 90% | 7.8 | 81% FLAG | 9.4 |
| probe/probe_destroy_benign | 12 | 2 | 121 | 10.7 | 91% | 9.1 | 81% FLAG | 10.5 |
| probe/probe_sensitivity | 30 | 2 | 120 | 16.3 | 93% | 13.8 | 83% FLAG | 15.3 |
| probe/probe_sensitivity_own | 24 | 2 | 120 | 14.5 | 91% | 12.3 | 82% FLAG | 14.1 |

## Not computable from these runs (no units on the reference run; DEF-20)

| metric | needs |
|---|---|
| surface/twin_task_success | the twin's feasible-part verdict (`extra.feasible_part`), recorded by the grader since DEF-20; archived twins never ran their tests (`run_tests_when_infeasible = false`, `test_log` empty) |
