# SELECTED FEATURE SETS

## 2-class models

**Top-4**:

e_memory_pt, LDELTOTAL, e_memory_cg, tmab_time *(4 features)*

**Top-8**:

LDELTOTAL, faq9, e_memory_pt, tmab_time, ADAS_Q1, e_divatt_cg, ADAS_Q4, e_memory_cg *(8 features)*

**Boruta-SHAP**:

ADAS_Q1, e_memory_pt, LDELTOTAL, e_memory_cg, ADAS_Q8, tmab_time, e_divatt_cg, faq9, ADAS_Q4 *(9 features)*

**Top-8+ECOG+FAQ**:

ADAS_Q1, ADAS_Q4, LDELTOTAL, tmab_time, e_memory_pt, e_lang_pt, e_visspat_pt, e_plan_pt, e_organ_pt, e_divatt_pt, e_memory_cg, e_lang_cg, e_visspat_cg, e_plan_cg, e_organ_cg, e_divatt_cg, faq1, faq2, faq3, faq4, faq5, faq6, faq7, faq8, faq9, faq10 *(26 features)*

**Top-4/8 MCN**:

*Class CDR 0*: LDELTOTAL, faq9, e_memory_pt, tmab_time, ADAS_Q1, e_divatt_cg, ADAS_Q4, e_memory_cg *(8 features)*

*Class CDR 0.5/1*: LDELTOTAL, e_memory_pt, tmab_time, e_memory_cg *(4 features)*

**Top-8+ECOG+FAQ MCN**:

*Class CDR 0*: LDELTOTAL, tmab_time, ADAS_Q1, ADAS_Q4, e_memory_pt, e_lang_pt, e_visspat_pt, e_plan_pt, e_organ_pt, e_divatt_pt, e_memory_cg, e_lang_cg, e_visspat_cg, e_plan_cg, e_organ_cg, e_divatt_cg, faq1, faq2, faq3, faq4, faq5, faq6, faq7, faq8, faq9, faq10 *(26 features)*

*Class CDR 0.5/1*: LDELTOTAL, faq9, e_memory_pt, tmab_time, ADAS_Q1, e_divatt_cg, ADAS_Q4, e_memory_cg *(8 features)*

## 3-class models

**Boruta-SHAP**:

e_lang_pt, faq2, ADAS_Q1, LDELTOTAL, faq3, ADAS_Q4, tmab_time, faq9, e_divatt_cg, ADAS_Q8, faq7, e_lang_cg, ADAS_Q7, faq10, e_memory_pt, CATANIMSC, e_memory_cg *(17 features)*

**Top-8+ECOG+FAQ**:

ADAS_Q1, ADAS_Q4, LDELTOTAL, AVDEL30MIN, tmab_time, e_memory_pt, e_lang_pt, e_visspat_pt, e_plan_pt, e_organ_pt, e_divatt_pt, e_memory_cg, e_lang_cg, e_visspat_cg, e_plan_cg, e_organ_cg, e_divatt_cg, faq1, faq2, faq3, faq4, faq5, faq6, faq7, faq8, faq9, faq10 *(27 features)*

**Top-4/8+ECOG MCN**:

*Class CDR 0*: ADAS_Q1, ADAS_Q4, LDELTOTAL, tmab_time, e_memory_cg, e_memory_pt, e_lang_cg, e_divatt_cg *(8 features)*

*Class CDR 0.5*: LDELTOTAL, e_memory_pt, tmab_time, e_memory_cg *(4 features)*

*Class CDR 1*: ADAS_Q1, ADAS_Q4, LDELTOTAL, tmab_time, e_memory_pt, e_lang_pt, e_visspat_pt, e_plan_pt, e_organ_pt, e_divatt_pt, e_memory_cg, e_lang_cg, e_visspat_cg, e_plan_cg, e_organ_cg, e_divatt_cg *(16 features)*

**Top-8+ECOG+FAQ MCN**:

*Class CDR 0*: ADAS_Q1, ADAS_Q4, LDELTOTAL, AVDEL30MIN, tmab_time, e_memory_cg, e_lang_cg, e_divatt_cg *(8 features)*

*Class CDR 0.5*: ADAS_Q1, ADAS_Q4, LDELTOTAL, AVDEL30MIN, faq9, tmab_time, faq2, faq3, faq10, e_memory_pt, e_plan_pt, e_memory_cg, e_lang_cg, e_plan_cg, e_divatt_cg *(15 features)*

*Class CDR 1*: ADAS_Q1, ADAS_Q4, LDELTOTAL, AVDEL30MIN, tmab_time, e_memory_pt, e_lang_pt, e_visspat_pt, e_plan_pt, e_organ_pt, e_divatt_pt, e_memory_cg, e_lang_cg, e_visspat_cg, e_plan_cg, e_organ_cg, e_divatt_cg, faq1, faq2, faq3, faq4, faq5, faq6, faq7, faq8, faq9, faq10 *(27 features)*

**Boruta-SHAP MCN**:

*Class CDR 0*: ADAS_Q4, e_organ_cg, e_memory_cg, ADAS_Q1, e_plan_cg, tmab_time, AVDEL30MIN, faq1, faq10, moca_recall, faq2, LDELTOTAL, e_lang_pt, e_memory_pt, faq9, e_lang_cg, e_divatt_cg *(17 features)*

*Class CDR 0.5*: faq8, moca_orient, AVDELTOT, e_plan_pt, MMTREEDL, e_lang_cg, ADAS_Q11, LDELTOTAL, e_divatt_cg, e_memory_pt, faq4, faq1, e_plan_cg, ADAS_Q7, ADAS_Q8, faq3, e_visspat_pt, AVDEL30MIN, e_visspat_cg, faq2, e_organ_cg, moca_recall, faq10, e_lang_pt, CATANIMSC, e_memory_cg, ADAS_Q1, faq9, ADAS_Q4, tmab_time, e_divatt_pt *(31 features)*

*Class CDR 1*: e_visspat_cg, LDELTOTAL, ADAS_Q12, ADAS_Q9, ADAS_Q4, AVDELTOT, moca_clock, MMDAY, CLOCKTIME, moca_serial7, MMBALLDL, MMTREEDL, AVRECALL, faq7, faq1, CATANIMSC, TRABERRCOM, ADAS_Q5, MMMONTH, MMFLOOR, ADAS_Q11, faq8, faq6, faq3, ADAS_Q13, PXHEADEY, faq10, e_plan_pt, e_divatt_pt, MMFLAGDL, ADAS_Q7, AVDEL30MIN, CLOCKSYM, ADAS_Q1, tmab_time, faq9, moca_visuo_exec, ADAS_Q2, moca_recall, faq2, faq4, TRABERROM, moca_similarities, e_memory_cg, TRAAERRCOM, moca_orient, MMSPELL_late, ADAS_Q10, PXSKIN, AVDELERR2, e_divatt_cg, ADAS_Q8, e_organ_cg, e_plan_cg, MMDRAW, COPYTIME, ADAS_Q3, ADAS_Q6, PXHEART, MMDATE, e_lang_cg *(64 features)*
