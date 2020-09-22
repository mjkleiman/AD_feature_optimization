'''
This file combines the available neuropsychological, demographic, and classification data
from the separate ADNI data files

Outputs write to the parent directory's data > interim folder
'''

import numpy as np
import pandas as pd
import os
from sklearn.impute import SimpleImputer

na_values = [-1,-4,np.nan]
df_adni_dx = pd.read_csv('../data/interim/DX.csv', na_values=na_values)
df_adni_adas = pd.read_csv('../data/raw/ADAS_norm.csv', na_values=na_values)
df_adni_moca = pd.read_csv('../data/raw/MOCA.csv', na_values=na_values)
df_adni_faq = pd.read_csv('../data/raw/FAQ.csv', na_values=na_values)
df_adni_mmse = pd.read_csv('../data/raw/MMSE.csv', na_values=na_values)
df_adni_cdr = pd.read_csv('../data/raw/CDR.csv', na_values=na_values)
df_adni_medhist = pd.read_csv('../data/raw/MEDHIST.csv', na_values=na_values)
df_adni_phys = pd.read_csv('../data/raw/PHYSICAL.csv', na_values=na_values)
df_adni_demog = pd.read_csv('../data/raw/PTDEMOG.csv', na_values=na_values)
df_adni_neurobat = pd.read_csv('../data/raw/NEUROBAT.csv', na_values=na_values)
df_adni_ecog_pt = pd.read_csv('../data/raw/ECOGPT.csv', na_values=9)
df_adni_ecog_cg = pd.read_csv('../data/raw/ECOGSP.csv', na_values=9)

df_adni_faq = df_adni_faq.loc[:,['RID','VISCODE2','FAQFINAN','FAQFORM','FAQSHOP','FAQGAME','FAQBEVG',
                                 'FAQMEAL','FAQEVENT','FAQTV','FAQREM','FAQTRAVL']
                              ].rename(columns={'VISCODE2':'VISCODE','FAQFINAN':'faq1','FAQFORM':'faq2','FAQSHOP':'faq3',
                                                'FAQGAME':'faq4','FAQBEVG':'faq5','FAQMEAL':'faq6','FAQEVENT':'faq7',
                                                'FAQTV':'faq8','FAQREM':'faq9','FAQTRAVL':'faq10'})
                                                                            
df_adni_mmse = df_adni_mmse.loc[:,['RID','VISCODE2','MMDATE','MMYEAR','MMMONTH','MMDAY','MMSEASON','MMHOSPIT','MMFLOOR','MMCITY',
                                   'MMAREA','MMSTATE','MMBALL','MMFLAG','MMTREE','MMD','MML','MMR','MMO','MMW','MMBALLDL','MMFLAGDL','MMTREEDL',
                                  'MMWATCH', 'MMPENCIL','MMREPEAT', 'MMHAND', 'MMFOLD', 'MMONFLR', 'MMREAD', 'MMWRITE','MMDRAW']
                              ].rename(columns={'VISCODE2':'VISCODE'})
df_adni_medhist = df_adni_medhist.loc[:,['RID', 'VISCODE2','MHPSYCH', 'MH2NEURL', 'MH3HEAD', 'MH4CARD', 'MH5RESP', 'MH6HEPAT', 'MH7DERM', 'MH8MUSCL', 'MH9ENDO', 'MH10GAST', 'MH11HEMA',
                                          'MH12RENA', 'MH13ALLE', 'MH14ALCH', 'MH15DRUG', 'MH16SMOK', 'MH17MALI', 'MH18SURG', 'MH19OTHR']
                              ].rename(columns={'VISCODE2':'VISCODE'})
df_adni_phys = df_adni_phys.loc[:,['RID', 'VISCODE2','PXGENAPP', 'PXHEADEY', 'PXNECK', 'PXCHEST', 'PXHEART', 'PXABDOM', 'PXEXTREM', 'PXPERIPH', 'PXSKIN', 'PXMUSCUL']
                              ].rename(columns={'VISCODE2':'VISCODE'})
df_adni_demog = df_adni_demog.loc[:,['RID', 'VISCODE2', 'PTGENDER', 'PTHAND', 'PTMARRY', 'PTEDUCAT', 'PTNOTRT', 'PTHOME', 'PTETHCAT', 'PTRACCAT']
                              ].rename(columns={'VISCODE2':'VISCODE'})
df_adni_neurobat = df_adni_neurobat.loc[:,['RID', 'VISCODE2','CLOCKCIRC', 'CLOCKSYM', 'CLOCKNUM', 'CLOCKHAND', 'CLOCKTIME','COPYCIRC', 'COPYSYM', 'COPYNUM', 'COPYHAND', 'COPYTIME',
                                           'AVTOT1', 'AVTOT2', 'AVTOT3', 'AVTOT4', 'AVTOT5', 'AVTOT6', 'DSPANFOR','DSPANBAC','CATANIMSC','CATVEGESC','TRAASCOR','TRABSCOR',
                                           'TRAAERRCOM', 'TRAAERROM','TRABERRCOM','TRABERROM','LIMMTOTAL','LDELTOTAL','AVDEL30MIN','AVDELTOT','AVDELERR2']
                              ].rename(columns={'VISCODE2':'VISCODE','TRAASCOR':'tma_time','TRABSCOR':'tmb_time','DSPANFOR':'nbspan_forward','DSPANBAC':'nbspan_backward'})
df_adni_ecog_pt = df_adni_ecog_pt.loc[:,['RID','VISCODE2','MEMORY1', 'MEMORY2', 'MEMORY3', 'MEMORY4',
       'MEMORY5', 'MEMORY6', 'MEMORY7', 'MEMORY8', 'LANG1', 'LANG2',
       'LANG3', 'LANG4', 'LANG5', 'LANG6', 'LANG7', 'LANG8', 'LANG9',
       'VISSPAT1', 'VISSPAT2', 'VISSPAT3', 'VISSPAT4', 'VISSPAT5',
       'VISSPAT6', 'VISSPAT7', 'VISSPAT8', 'PLAN1', 'PLAN2', 'PLAN3',
       'PLAN4', 'PLAN5', 'ORGAN1', 'ORGAN2', 'ORGAN3', 'ORGAN4', 'ORGAN5',
       'ORGAN6', 'DIVATT1', 'DIVATT2', 'DIVATT3', 'DIVATT4']].rename(columns={'VISCODE2':'VISCODE','MEMORY1':'MEM1_pt','MEMORY2':'MEM2_pt','MEMORY3':'MEM3_pt','MEMORY4':'MEM4_pt',
                                                                     'MEMORY5':'MEM5_pt','MEMORY6':'MEM6_pt','MEMORY7':'MEM7_pt','MEMORY8':'MEM8_pt',
                                                                     'LANG1':'LANG1_pt','LANG2':'LANG2_pt','LANG3':'LANG3_pt','LANG4':'LANG4_pt',
                                                                     'LANG5':'LANG5_pt','LANG6':'LANG6_pt','LANG7':'LANG7_pt','LANG8':'LANG8_pt',
                                                                     'LANG9':'LANG9_pt','VISSPAT1':'VISSPAT1_pt','VISSPAT2':'VISSPAT2_pt','VISSPAT3':'VISSPAT3_pt',
                                                                     'VISSPAT4':'VISSPAT4_pt','VISSPAT5':'VISSPAT5_pt','VISSPAT6':'VISSPAT6_pt','VISSPAT7':'VISSPAT7_pt',
                                                                     'VISSPAT8':'VISSPAT8_pt','PLAN1':'PLAN1_pt','PLAN2':'PLAN2_pt','PLAN3':'PLAN3_pt','PLAN4':'PLAN4_pt',
                                                                     'PLAN5':'PLAN5_pt','ORGAN1':'ORGAN1_pt','ORGAN2':'ORGAN2_pt','ORGAN3':'ORGAN3_pt','ORGAN4':'ORGAN4_pt',
                                                                     'ORGAN5':'ORGAN5_pt','ORGAN6':'ORGAN6_pt','DIVATT1':'DIVATT1_pt','DIVATT2':'DIVATT2_pt',
                                                                     'DIVATT3':'DIVATT3_pt','DIVATT4':'DIVATT4_pt'})

df_adni_ecog_cg = df_adni_ecog_cg.loc[:,['RID','VISCODE2','MEMORY1', 'MEMORY2', 'MEMORY3', 'MEMORY4',
       'MEMORY5', 'MEMORY6', 'MEMORY7', 'MEMORY8', 'LANG1', 'LANG2',
       'LANG3', 'LANG4', 'LANG5', 'LANG6', 'LANG7', 'LANG8', 'LANG9',
       'VISSPAT1', 'VISSPAT2', 'VISSPAT3', 'VISSPAT4', 'VISSPAT5',
       'VISSPAT6', 'VISSPAT7', 'VISSPAT8', 'PLAN1', 'PLAN2', 'PLAN3',
       'PLAN4', 'PLAN5', 'ORGAN1', 'ORGAN2', 'ORGAN3', 'ORGAN4', 'ORGAN5',
       'ORGAN6', 'DIVATT1', 'DIVATT2', 'DIVATT3', 'DIVATT4']].rename(columns={'VISCODE2':'VISCODE','MEMORY1':'MEM1_cg','MEMORY2':'MEM2_cg','MEMORY3':'MEM3_cg','MEMORY4':'MEM4_cg',
                                                                     'MEMORY5':'MEM5_cg','MEMORY6':'MEM6_cg','MEMORY7':'MEM7_cg','MEMORY8':'MEM8_cg',
                                                                     'LANG1':'LANG1_cg','LANG2':'LANG2_cg','LANG3':'LANG3_cg','LANG4':'LANG4_cg',
                                                                     'LANG5':'LANG5_cg','LANG6':'LANG6_cg','LANG7':'LANG7_cg','LANG8':'LANG8_cg',
                                                                     'LANG9':'LANG9_cg','VISSPAT1':'VISSPAT1_cg','VISSPAT2':'VISSPAT2_cg','VISSPAT3':'VISSPAT3_cg',
                                                                     'VISSPAT4':'VISSPAT4_cg','VISSPAT5':'VISSPAT5_cg','VISSPAT6':'VISSPAT6_cg','VISSPAT7':'VISSPAT7_cg',
                                                                     'VISSPAT8':'VISSPAT8_cg','PLAN1':'PLAN1_cg','PLAN2':'PLAN2_cg','PLAN3':'PLAN3_cg','PLAN4':'PLAN4_cg',
                                                                     'PLAN5':'PLAN5_cg','ORGAN1':'ORGAN1_cg','ORGAN2':'ORGAN2_cg','ORGAN3':'ORGAN3_cg','ORGAN4':'ORGAN4_cg',
                                                                     'ORGAN5':'ORGAN5_cg','ORGAN6':'ORGAN6_cg','DIVATT1':'DIVATT1_cg','DIVATT2':'DIVATT2_cg',
                                                                     'DIVATT3':'DIVATT3_cg','DIVATT4':'DIVATT4_cg'})

df_adni_cdr = df_adni_cdr.loc[:,['RID','VISCODE2','CDSOURCE', 'CDVERSION', 'CDMEMORY',
       'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE', 'CDGLOBAL']].rename(columns={'VISCODE2':'VISCODE'})

df_adni_ecog_p = df_adni_ecog_pt.copy()
df_adni_ecog_p.replace({np.nan:0}, inplace=True)
df_adni_ecog_pt['e_memory_pt'] = df_adni_ecog_p[['MEM1_pt', 'MEM2_pt', 'MEM3_pt', 'MEM4_pt',
       'MEM5_pt', 'MEM6_pt', 'MEM7_pt', 'MEM8_pt']].mean(axis=1).round()
df_adni_ecog_pt['e_lang_pt'] = df_adni_ecog_p[['LANG1_pt', 'LANG2_pt',
       'LANG3_pt', 'LANG4_pt', 'LANG5_pt', 'LANG6_pt', 'LANG7_pt',
       'LANG8_pt']].mean(axis=1).round()
df_adni_ecog_pt['e_visspat_pt'] = df_adni_ecog_p[['VISSPAT1_pt', 'VISSPAT2_pt',
       'VISSPAT3_pt', 'VISSPAT4_pt', 'VISSPAT5_pt', 'VISSPAT6_pt',
       'VISSPAT7_pt', 'VISSPAT8_pt']].mean(axis=1).round()
df_adni_ecog_pt['e_plan_pt'] = df_adni_ecog_p[['PLAN1_pt', 'PLAN2_pt', 'PLAN3_pt',
       'PLAN4_pt', 'PLAN5_pt']].mean(axis=1).round()
df_adni_ecog_pt['e_organ_pt'] = df_adni_ecog_p[['ORGAN1_pt', 'ORGAN2_pt', 'ORGAN3_pt',
       'ORGAN4_pt', 'ORGAN5_pt', 'ORGAN6_pt']].mean(axis=1).round()
df_adni_ecog_pt['e_divatt_pt'] = df_adni_ecog_p[['DIVATT1_pt', 'DIVATT2_pt',
       'DIVATT3_pt', 'DIVATT4_pt']].mean(axis=1).round()
df_adni_ecog_c = df_adni_ecog_cg
df_adni_ecog_c.replace({np.nan:0}, inplace=True)
df_adni_ecog_cg['e_memory_cg'] = df_adni_ecog_c[['MEM1_cg', 'MEM2_cg', 'MEM3_cg', 'MEM4_cg',
       'MEM5_cg', 'MEM6_cg', 'MEM7_cg', 'MEM8_cg']].mean(axis=1).round()
df_adni_ecog_cg['e_lang_cg'] = df_adni_ecog_c[['LANG1_cg', 'LANG2_cg',
       'LANG3_cg', 'LANG4_cg', 'LANG5_cg', 'LANG6_cg', 'LANG7_cg',
       'LANG8_cg']].mean(axis=1).round()
df_adni_ecog_cg['e_visspat_cg'] = df_adni_ecog_c[['VISSPAT1_cg', 'VISSPAT2_cg',
       'VISSPAT3_cg', 'VISSPAT4_cg', 'VISSPAT5_cg', 'VISSPAT6_cg',
       'VISSPAT7_cg', 'VISSPAT8_cg']].mean(axis=1).round()
df_adni_ecog_cg['e_plan_cg'] = df_adni_ecog_c[['PLAN1_cg', 'PLAN2_cg', 'PLAN3_cg',
       'PLAN4_cg', 'PLAN5_cg']].mean(axis=1).round()
df_adni_ecog_cg['e_organ_cg'] = df_adni_ecog_c[['ORGAN1_cg', 'ORGAN2_cg', 'ORGAN3_cg',
       'ORGAN4_cg', 'ORGAN5_cg', 'ORGAN6_cg']].mean(axis=1).round()
df_adni_ecog_cg['e_divatt_cg'] = df_adni_ecog_c[['DIVATT1_cg', 'DIVATT2_cg',
       'DIVATT3_cg', 'DIVATT4_cg']].mean(axis=1).round()
df_adni_ecog_pt = df_adni_ecog_pt.loc[:,['RID','VISCODE','q_memory_pt','q_orient_pt','q_judgmt_pt','q_outsideact_pt','q_homeact_pt','q_language_pt','q_attention_pt',
                                         'e_memory_pt','e_lang_pt','e_visspat_pt','e_plan_pt','e_organ_pt','e_divatt_pt']]
df_adni_ecog_cg = df_adni_ecog_cg.loc[:,['RID','VISCODE','q_memory_cg','q_orient_cg','q_judgmt_cg','q_outsideact_cg','q_homeact_cg','q_language_cg','q_attention_cg',
                                         'e_memory_cg','e_lang_cg','e_visspat_cg','e_plan_cg','e_organ_cg','e_divatt_cg']]

viscode2dict = {'v01':'sc','v02':'scmri','v03':'bl','v04':'m03','v05':'m06','v06':'m24','v07':'m30','v11':'m36','v12':'m42','v21':'m48','v22':'m54','v31':'m60','v32':'m66','v41':'m72','v42':'m78'}
viscode2num = {'sc':0,'scmri':0,'bl':1,'m03':2,'m06':2,'m12':2,'m18':3,'m24':3,'m30':4,'m36':4,'m42':5,'m48':5,'m54':6,'m60':6,'m66':7,'m72':7,'m78':8,'m84':8,'m90':9,'m96':9,'m102':10,'m108':10,'m114':10,'m120':11,'m126':11,'m132':12,'m144':13,'m156':14,'m168':15,'uns1':-1,'init':-1,'f':-1,'y1':-1}

# Merge onto a single dataframe
df_all = df_adni_dx.merge(df_adni_moca, how='outer', on=['RID','VISCODE']).set_index(['RID','VISCODE'])
df_all = df_all.merge(df_adni_adas, how='outer', on=['RID','VISCODE']).set_index(['RID','VISCODE']) 
df_all = df_all.merge(df_adni_faq, how='outer', on=['RID','VISCODE']).set_index(['RID','VISCODE']) 
df_all = df_all.merge(df_adni_mmse, how='outer', on=['RID','VISCODE']).set_index(['RID','VISCODE']) 
df_all = df_all.merge(df_adni_phys, how='outer', on=['RID','VISCODE']).set_index(['RID','VISCODE'])
df_all = df_all.merge(df_adni_cdr, how='outer', on=['RID','VISCODE']).set_index(['RID','VISCODE'])
df_all = df_all.merge(df_adni_neurobat, how='outer', on=['RID','VISCODE']).set_index(['RID','VISCODE'])
df_all = df_all.merge(df_adni_ecog_pt, how='outer', on=['RID','VISCODE']).set_index(['RID','VISCODE'])
df_all = df_all.merge(df_adni_ecog_cg, how='outer', on=['RID','VISCODE']).set_index(['RID','VISCODE'])
df_all.reset_index(inplace=True)
df_all['VISCODE'].replace(viscode2num, inplace=True)

df_all.sort_values(['RID','VISCODE'], inplace=True)
df_2 = pd.DataFrame(columns=df_all.columns.values)

# Imputation
for i in df_all['RID'].unique():
  df_1 = df_all.loc[df_all['RID']==i]
  df_1 = df_1.loc[df_all['RID']==i].fillna(method='ffill')
  df_1 = df_1.loc[df_all['RID']==i].fillna(method='bfill')
  df_2 = df_2.append(df_1)

df_2 = df_2.drop_duplicates(subset=['RID','VISCODE'])
colnum = df_2.columns.nunique()
df_2 = df_2.dropna(thresh=(colnum-15)) # Select rows only where # of NaNs is less than 15
df_2 = df_2.dropna(subset=['DX'])
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
df_imp = imp_median.fit_transform(df_2)
df_imp = pd.DataFrame(df_imp, columns=df_2.columns.values).astype(float).round(2)

df_adni = df_imp.loc[df_imp['VISCODE']==0]

df_adni = df_adni[df_adni.CDGLOBAL != 2] # Remove single subject where CDR = 2
df_adni['MMSPELL_early'] = df_adni.loc[:,['MMD','MML']].sum(axis=1)
df_adni['MMSPELL_late'] = df_adni.loc[:,['MMW','MMO','MMR']].sum(axis=1)
df_adni['AVRECALL'] = df_adni['AVTOT5'] - df_adni['AVTOT6'] 
df_adni['AVRECALL'] = df_adni['AVRECALL'].clip(lower=0)
df_adni['tmab_time'] = df_adni.loc[:,['tma_time','tmb_time']].sum(axis=1)
df_adni.drop(columns=['MMW','MMO','MMR','MML','MMD','AVTOT1','AVTOT2','AVTOT3','AVTOT4','AVTOT5','AVTOT6',
                   'tma_time','tmb_time','LIMMTOTAL'], axis=1, inplace=True)

df_adni.to_csv('../data/interim/df_adni.csv', index=False)