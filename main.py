# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 22:08:07 2025

@author: Elsa
"""

import numpy as np
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")

import functions as func
from model_1 import Model_1

def single_run(in_path, fit, elec_price, out_path,
               md_level=np.inf, ud_penalty=0.001, re_level=0, 
               voll=0.7, total_budget=np.inf, interest=0.1,
               re_start=0, pros_perc = None):
    
    global model
    os.makedirs(out_path, exist_ok=True)
    
    model = Model_1(_file_name=in_path)
    model.load_data()
    model.solve(fit=fit, elec_price=elec_price, 
                md_level = md_level, ud_penalty=ud_penalty, 
                re_level=re_level, voll=voll, total_budget=total_budget,
                interest=interest, re_start=re_start, pros_perc=pros_perc)
    # func.output_data(model, 7)
    func.to_xlsx(model, round(fit * 100), round(elec_price * 100), 
                 out_path, multi=0)

def multi_run(in_path, fits, elec_prices, out_path,
              md_level=np.inf, ud_penalty=0.001, re_level=0, 
              voll=0.7, total_budget=np.inf, index='budget', interest=0.1,
               re_start=0, pros_perc=None):
    
    global model
    os.makedirs(out_path, exist_ok=True)
    
    for elec_price in elec_prices:
        for fit in fits:
            model = Model_1(_file_name=in_path)
            model.load_data()
            model.solve(fit=fit, elec_price=elec_price, 
                        md_level = md_level, ud_penalty=ud_penalty, 
                        re_level=re_level, voll=voll, 
                        total_budget=total_budget, interest=interest,
                        re_start=re_start, pros_perc=pros_perc)
            # func.output_data(model, 2)
            func.to_xlsx(model, round(fit * 100), round(elec_price * 100), 
                         out_path, 1, index)    
            print(f'Save to: {out_path}')
            
def fit_search(in_path, out_path, prices,
               md_level=np.inf, ud_penalty=0.001, re_level=0, voll=0.7,
               total_budget=np.inf, search='budget', interest=0.1,
               re_start=0, pros_perc=None, base_npv=None):
    
    os.makedirs(out_path, exist_ok=True)
    index = search
    
    # Initialize model
    model = Model_1(_file_name=in_path)
    model.load_data()
    
    
    # Define base case
    if base_npv == None:
        base_path = os.path.join(os.getcwd(), "Outputs",
                                 "0. Current Case", "Output_0_40.xlsx")
        file = pd.read_excel(base_path, sheet_name='Summary')
        file.set_index('Unnamed: 0', inplace=True)
        base_npv = file.loc['NPV', 0]

    # Grid search
    try:
        output_files = os.listdir(out_path)
        
        if "Summary.xlsx" in output_files:
            prev_summary = pd.read_excel(os.path.join(out_path, "Summary.xlsx"),
                                         sheet_name=None)
            last_re = list(prev_summary.keys())[-1]
            
            if index == 're':
                if float(last_re) == re_level:
                    last_re_summary = prev_summary[
                        last_re
                        ].set_index('Unnamed: 0')
                    fits = last_re_summary.loc['Feed-in Tariffs'].dropna()
                    fits = list(fits)
                    objs = last_re_summary.loc['NPV'].dropna()
                    objs = list(objs)
                    
                elif float(last_re) < re_level:
                    fits = []
                    objs = []
                    
                else:
                    return
                
            else:
                fits = []
                objs = []
                
        else:
            fits = []
            objs = []
                
    except FileNotFoundError:
        fits = []
        objs = []

    for el_price in prices[len(fits)::]:
        # Check if there is a positive solution
        fit = 0
        model.solve(fit=fit, elec_price=el_price,
                    ud_penalty=ud_penalty, md_level=md_level,
                    re_level=re_level, voll=voll, 
                    total_budget=total_budget, interest=interest,
                        re_start=re_start, pros_perc=pros_perc)
        ud_correction = sum([model.ud[y, d, h].X
                             * model.d_weights[d]
                             * model.ud_penalty
                             * (1 / (1 + model.i) ** y)
                             for y in range(model.years)
                             for d in range(model.days)
                             for h in range(model.hours)])
        
        if float(model.m.ObjVal + ud_correction) < base_npv:
            print(f'No positive solution for {el_price}')
            fits.append(0)
            objs.append(base_npv)
            
            summary = pd.DataFrame((prices, fits, [base_npv]*len(fits), objs), 
                                   index=['Prices', 'Feed-in Tariffs', 
                                          'Base NPV', 'NPV'])
            try:
                with pd.ExcelWriter(os.path.join(out_path, 'Summary.xlsx'), 
                                    mode='a', engine='openpyxl', 
                                    if_sheet_exists='replace') as writer:
                    if index == 're':
                        summary.to_excel(writer, sheet_name=str(re_level))
                        
                    elif index == 'budget':
                        summary.to_excel(writer, 
                                         sheet_name=str(total_budget))
                    elif index == 'voll':
                        summary.to_excel(writer, 
                                         sheet_name=str(voll))
                    elif index == 'pros':
                        summary.to_excel(writer, 
                                         sheet_name=str(pros_perc))
                
            except FileNotFoundError:
                with pd.ExcelWriter(os.path.join(out_path, 'Summary.xlsx'), 
                                    mode='w', engine='openpyxl') as writer:
                    if index == 're':
                        summary.to_excel(writer, sheet_name=str(re_level))
                        
                    elif index == 'budget':
                        summary.to_excel(writer, 
                                         sheet_name=str(total_budget))
                    elif index == 'voll':
                        summary.to_excel(writer, 
                                         sheet_name=str(voll))
                    elif index == 'pros':
                        summary.to_excel(writer, 
                                         sheet_name=str(pros_perc))
                        
        # If yes, run a binary grid search to find it
        elif len(fits) != 0:
            fit_left = fits[-1]
            fit_right = 4
            fit_mid = (fit_left + fit_right) / 2
            model.m.reset()
            worse = True
            print(f'-----FiT: {fit_mid}, price: {el_price}-----')
            model.solve(fit=fit_mid, elec_price=el_price,
                        ud_penalty=ud_penalty, md_level=md_level, 
                        re_level=re_level, voll=voll, 
                        total_budget=total_budget, interest=interest,
                        re_start=re_start, pros_perc=pros_perc)
            ud_correction = sum([model.ud[y, d, h].X
                                 * model.d_weights[d]
                                 * model.ud_penalty
                                 * (1 / (1 + model.i) ** y)
                                 for y in range(model.years)
                                 for d in range(model.days)
                                 for h in range(model.hours)])
    
            while worse:
                model_feed_in = sum(model.feed_in[i, y, d, h].X 
                                    for (i, y, d, h) in model.feed_in.keys())
                model_obj = float(model.m.ObjVal + ud_correction)
                
                if model_feed_in == 0 and model_obj >= base_npv:
                    fit_mid = 'inf'
                    break
                
                # if (fit_right - fit_mid) < 0.01 and model_obj >= base_npv:
                #     print("------------breaking-----------------")
                #     break
                
                if model_obj >= base_npv:
                    fit_left = fit_mid
                else:
                    fit_right = fit_mid
                fit_mid = (fit_right + fit_left) / 2
                model.m.reset()
                print(f'-----FiT: {fit_mid}, price: {el_price}-----')
                model.solve(fit=fit_mid, elec_price=el_price,
                            ud_penalty=ud_penalty, md_level=md_level, 
                            re_level=re_level, voll=voll,
                            total_budget=total_budget, interest=interest,
                            re_start=re_start, pros_perc=pros_perc)
                ud_correction = sum([model.ud[y, d, h].X
                                     * model.d_weights[d]
                                     * model.ud_penalty
                                     * (1 / (1 + model.i) ** y)
                                     for y in range(model.years)
                                     for d in range(model.days)
                                     for h in range(model.hours)])
                model_obj = float(model.m.ObjVal + ud_correction)
                
                if (abs(model_obj - base_npv) <= 10000
                    and model_obj >= base_npv):
                    worse = False
                
            if fit_mid == 'inf':
                break
            
            fits.append(fit_mid)
            objs.append(model_obj)
            func.output_data(model, 2)
            func.to_xlsx(model, round(fit_mid * 100), round(el_price * 100), 
                         os.path.join(out_path), index=search)
            
            summary = pd.DataFrame((prices, fits, [base_npv]*len(fits), objs), 
                                   index=['Prices', 'Feed-in Tariffs', 
                                          'Base NPV', 'NPV'])
            try:
                with pd.ExcelWriter(os.path.join(out_path, 'Summary.xlsx'), 
                                    mode='a', engine='openpyxl', 
                                    if_sheet_exists='replace') as writer:
                    if index == 're':
                        summary.to_excel(writer, sheet_name=str(re_level))
                        
                    elif index == 'budget':
                        summary.to_excel(writer, 
                                         sheet_name=str(total_budget))
                    elif index == 'voll':
                        summary.to_excel(writer, 
                                         sheet_name=str(voll))
                    elif index == 'pros':
                        summary.to_excel(writer, 
                                         sheet_name=str(pros_perc))
                        
                
            except FileNotFoundError:
                with pd.ExcelWriter(os.path.join(out_path, 'Summary.xlsx'), 
                                    mode='w', engine='openpyxl') as writer:
                    if index == 're':
                        summary.to_excel(writer, sheet_name=str(re_level))
                        
                    elif index == 'budget':
                        summary.to_excel(writer, 
                                         sheet_name=str(total_budget))
                    elif index == 'voll':
                        summary.to_excel(writer, 
                                         sheet_name=str(voll))
                    elif index == 'pros':
                        summary.to_excel(writer, 
                                         sheet_name=str(pros_perc))
            
        else:
            fit_left = 0
            fit_right = 4
            fit_mid = (fit_left + fit_right) / 2
            model.m.reset()
            worse = True
            print(f'-----FiT: {fit_mid}, price: {el_price}-----')
            model.solve(fit=fit_mid, elec_price=el_price,
                        ud_penalty=ud_penalty, md_level=md_level,
                        re_level = re_level, voll=voll,
                        total_budget=total_budget, interest=interest,
                        re_start=re_start, pros_perc=pros_perc)
            ud_correction = sum([model.ud[y, d, h].X
                                 * model.d_weights[d]
                                 * model.ud_penalty
                                 * (1 / (1 + model.i) ** y)
                                 for y in range(model.years)
                                 for d in range(model.days)
                                 for h in range(model.hours)])
            
            while worse:
                model_feed_in = sum(model.feed_in[i, y, d, h].X 
                                    for (i, y, d, h) in model.feed_in.keys())
                model_obj = float(model.m.ObjVal + ud_correction)
                
                if model_feed_in == 0 and model_obj >= base_npv:
                    fit_mid = 'inf'
                    break
                
                # if (fit_right - fit_mid) < 0.01 and model_obj >= base_npv:
                #     break
                
                if model_obj >= base_npv:
                    fit_left = fit_mid
                else:
                    fit_right = fit_mid
                fit_mid = (fit_right + fit_left) / 2
                model.m.reset()
                print(f'-----FiT: {fit_mid}, price: {el_price}-----')
                model.solve(fit=fit_mid, elec_price=el_price,
                            ud_penalty=ud_penalty, md_level=md_level, 
                            re_level=re_level, voll=voll,
                            total_budget=total_budget, interest=interest,
                            pros_perc=pros_perc)
                ud_correction = sum([model.ud[y, d, h].X
                                     * model.d_weights[d]
                                     * model.ud_penalty
                                     * (1 / (1 + model.i) ** y)
                                     for y in range(model.years)
                                     for d in range(model.days)
                                     for h in range(model.hours)])
                model_obj = float(model.m.ObjVal + ud_correction)
                
                if (abs(model_obj - base_npv) <= 10000
                    and model_obj >= base_npv):
                    worse = False
                
            if fit_mid == 'inf':
                break
                    
            fits.append(fit_mid)
            objs.append(model_obj)
            func.output_data(model, 2)
            func.to_xlsx(model, round(fit_mid * 100), round(el_price * 100), 
                         out_path, index=search)
        
            
            summary = pd.DataFrame((prices, fits, [base_npv]*len(fits), objs), 
                                   index=['Prices', 'Feed-in Tariffs', 
                                          'Base NPV', 'NPV'])
            try:
                with pd.ExcelWriter(os.path.join(out_path, 'Summary.xlsx'), 
                                    mode='a', engine='openpyxl', 
                                    if_sheet_exists='replace') as writer:
                    summary.to_excel(writer, sheet_name=str(re_level))
                
            except FileNotFoundError:
                with pd.ExcelWriter(os.path.join(out_path, 'Summary.xlsx'), 
                                    mode='w', engine='openpyxl') as writer:
                    if index == 're':
                        summary.to_excel(writer, sheet_name=str(re_level))
                        
                    elif index == 'budget':
                        summary.to_excel(writer, 
                                         sheet_name=str(total_budget))
                    elif index == 'voll':
                        summary.to_excel(writer, 
                                         sheet_name=str(voll))
                    elif index == 'pros':
                        summary.to_excel(writer, 
                                         sheet_name=str(pros_perc))
                        
def min_fi(casePath, keys, max_fit, tot_d, perc, day_weights=[199, 106, 60]):
    
    sumPath = os.path.join(casePath, "Summary.xlsx")
    sumFile = pd.read_excel(sumPath, sheet_name = None)
    filePaths = os.path.join(casePath, "Grid Search", "Output Files")
    output_path = os.path.join(casePath, "Summary feed-in.xlsx")
        
    all_results = {}
    result_keys = []
    for key in keys:
        out = sumFile[str(key)].set_index("Unnamed: 0")
        fit_row = out.loc["Feed-in Tariffs"]
        if not np.isnan(fit_row.tolist()[-1]):
            min_fit = None
        else:
            non_nan_idx = np.where(~np.isnan(fit_row))[0]
            col_idx = non_nan_idx[-3]
            min_fit = out.loc["Feed-in Tariffs", col_idx]
            min_p = out.loc["Prices", col_idx]
        
        if min_fit != None:
            results = {}
            result_keys.append(key)
            for p in np.arange(round(min_p, 2) + 0.01, 0.41, 0.01):
                max_fit_p = None
                for fit in np.arange(round(min_fit, 2), max_fit, 0.01):
                    outFile = pd.read_excel(
                        os.path.join(filePaths,
                                     str(key),
                                     f"Output_{int(fit * 100)}_{int(p * 100)}.xlsx"),
                        sheet_name = None
                        )
                    out_fi = outFile['Fed-in Capacity']
                    out_fi.set_index("Unnamed: 0", inplace=True)
                    fi = 0
                    for _, row in out_fi.iterrows():
                        day = int(str(_).split('.')[1])
                        fi += sum(row) * day_weights[day]
                    out_ud = outFile["Summary"].set_index("Unnamed: 0")
                    ud = out_ud.loc["Unmet Demand", 0]
                    threshold = fi / (tot_d - ud)
                    if threshold <= perc:
                        max_fit_p = fit
                        break
                    if fit == max_fit - 0.01:
                        max_fit_p = np.inf
                        
                results[round(p, 2)] = max_fit_p
        
            all_results[key] = results
            
    with pd.ExcelWriter(output_path, 
                        engine='openpyxl') as writer:
        for key in result_keys:
            df_out = pd.DataFrame(all_results[key], index=["max_fit"])
            df_out.to_excel(writer, sheet_name=str(key))
            
            

cwd = os.getcwd()
# day_weights = [199, 106, 60]
# --> OR (after running the model at least once) ####
# list(model.d_weights)
####################################################


outFile_sum = os.path.join(cwd, 'Outputs')

# Current Case ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
in_path = os.path.join(cwd, 'Inputs', 'inputs.xlsx')
out_path = os.path.join(cwd, 'Outputs', '0. Current Case_250')
single_run(in_path=in_path, fit=0, elec_price=0.4, out_path=out_path,
           total_budget=250000, ud_penalty = 0)

# Current budget
current_path = os.path.join(cwd, 
                            'Outputs', 
                            '0. Current Case', 
                            'Output_0_40.xlsx')
current_df = pd.read_excel(current_path, sheet_name=None)
current_cfs = current_df['Costs and Revenues'].set_index('Unnamed: 0')
interest = current_df['Summary'].set_index('Unnamed: 0').loc['Interest Rate'][0]

current_budget = 0
for y in range(len(current_cfs.columns)):
    capex_y = current_cfs.loc['Total Capital Costs'][y]
    current_budget += capex_y * (1 / (1 + interest) ** y)
    
current_md = current_df['Summary'].set_index('Unnamed: 0').loc['Unmet Demand', 0]

# Budget sensitivity ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
in_path = os.path.join(cwd, 'Inputs', 'inputs_RE.xlsx')
out_path = os.path.join(cwd, 'Outputs', '1. Budget')
out_path_gs = os.path.join(out_path, 'Grid Search')

#   Creating Bugdet Range
budgets = np.arange(250000, 2000001, 250000)
budgets = budgets.tolist()
budgets.insert(1, int(current_budget // 1e5 * 1e5))
budgets.remove(500000)
budgets.insert(0, 100000)
budgets.insert(3, 420000)

prices = np.arange(0, 0.41, 0.01)
prices_gs = np.arange(0, 0.41, 0.01)
fits = np.arange(0, 0.26, 0.01)
'''
for budget in budgets:
    #fit_search(in_path, out_path, prices, re_level=0,
    #            total_budget=budget, search='budget')
    multi_run(in_path=in_path, fits=fits, elec_prices=prices_gs, 
              out_path=out_path_gs, re_level=0, 
              total_budget=budget)
'''
#   Finding total prosumer generation
inFile = pd.read_excel(in_path, sheet_name=None)
day_weights = inFile['day_weights']['Weight'].tolist()
pros_cap = sum(inFile['rent_cap'].set_index('Unnamed: 0').iloc[1])
pros_gen = 0
i = 0
for _, row in inFile['cap_factors'].set_index('Unnamed: 0').iterrows():
    pros_gen += sum(row) * pros_cap * day_weights[i] * 250 * 15
    i += 1

# summary_path_1 = os.path.join(outFile_sum, '1. Budget', 'Summary.xlsx')
# func.eval_summary(os.path.join(cwd, 'Outputs', '1. Budget', 
#                                 'Grid Search', 'Output Files'), pros_gen,
#                   max_fits = summary_path_1)
budgets = [250000, 420000, 750000, 1000000, 1250000, 1500000, 1750000, 2000000]
# min_fi(out_path, budgets, 0.26, 30789066.4, 0.05)

emFile_1 = pd.read_excel(os.path.join(out_path,
                                      'Grid Search',
                                      'Evaluation Metrics.xlsx'))
emFile_1.set_index('Budget', inplace=True)
weights = [199, 106, 60]
fi_perc = {}
for _, row in emFile_1.iterrows():
    p = int(row['Price'] * 100)
    fit = int(row['FiT'] * 100)
    outfile = pd.read_excel(os.path.join(out_path,
                                         'Grid Search',
                                         'Output Files',
                                         str(_),
                                         f'Output_{fit}_{p}.xlsx'),
                            sheet_name = None)
    pv_df = outfile["PV Dispatch"].set_index('Unnamed: 0')
    dg_df = outfile["DG Dispatch"].set_index('Unnamed: 0')
    fi_df = outfile['Fed-in Capacity'].set_index('Unnamed: 0')
    ud_df = outfile['Unmet Demand'].set_index('Unnamed: 0')
    
    tot_pv = 0
    tot_dg = 0
    tot_fi = 0
    tot_ud = 0
    
    for j, val in pv_df.iterrows():
        day = int(str(j).split('.')[1])
        tot_pv += sum(val) * weights[day]
    for j, val in dg_df.iterrows():
        day = int(str(j).split('.')[1])
        tot_dg += sum(val) * weights[day]
    for j, val in ud_df.iterrows():
        day = int(str(j).split('.')[1])
        tot_ud += sum(val) * weights[day]
    for j, val in fi_df.iterrows():
        day = int(str(j).split('.')[1])
        tot_fi += sum(val) * weights[day]
        
    fi_perc[_] = (tot_fi / (tot_pv + tot_dg + tot_fi + tot_ud))

fi_perc_df = pd.DataFrame(fi_perc, index=['feed-in perc'])
fi_perc_df.to_excel(os.path.join(out_path, "Fi perc.xlsx"))
'''
# RE Sensitivity ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
in_path = os.path.join(cwd, 'Inputs', 'inputs_RE.xlsx')
out_path = os.path.join(cwd, 'Outputs', '2. RE sensitivity')

re_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
prices = np.arange(0, 0.41, 0.01)
prices_gs = np.arange(0, 0.41, 0.01)
fits = np.arange(0, 0.3, 0.01)
current_budget = 400000

out_path_gs = os.path.join(out_path, 'Grid Search')

for re_level in re_levels:
    fit_search(in_path, out_path, prices, re_level=re_level,
               total_budget=current_budget, search='re')
    multi_run(in_path=in_path, fits=fits, elec_prices=prices_gs, 
              out_path=out_path_gs, re_level=re_level, 
              total_budget=current_budget, index='re')

inFile = pd.read_excel(in_path, sheet_name=None)
day_weights = inFile['day_weights']['Weight'].tolist()
pros_cap = sum(inFile['rent_cap'].set_index('Unnamed: 0').iloc[1])
pros_gen = 0
i = 0
for _, row in inFile['cap_factors'].set_index('Unnamed: 0').iterrows():
    pros_gen += sum(row) * pros_cap * day_weights[i] * 250 * 15
    i += 1
    
summary_path_2_b = os.path.join(outFile_sum, '2. RE sensitivity', 'Summary.xlsx')

func.eval_summary(os.path.join(outFile_sum, '2. RE sensitivity',
                               'Grid Search', 'Output Files'),
                  pros_gen,
                  max_fits = summary_path_2_b, index='re')

# Prosumer % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
in_path = os.path.join(cwd, 'Inputs', 'inputs.xlsx')
out_path = os.path.join(cwd, 'Outputs', '3. Prosumer percentage')

pros_percs = [0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1]
#pros_percs = [0, 0.1, 0.25, 0.4, 0.5]
#pros_percs = [0.6, 0.75, 0.9, 1]
prices = np.arange(0, 0.41, 0.01)
prices_gs = np.arange(0, 0.41, 0.01)
fits = np.arange(0, 0.26, 0.01)

out_path_gs = os.path.join(cwd, 'Outputs', '3. Prosumer percentage', 
                           'Grid Search')

pros_gens = {}
inFile = pd.read_excel(in_path, sheet_name=None)
day_weights = inFile['day_weights']['Weight'].tolist()
pros_cap = sum(inFile['rent_cap'].set_index('Unnamed: 0').iloc[1])

#   Current base cases:
for pros_perc in pros_percs:
    out_path_bc = os.path.join(out_path, 'Base Cases', 
                                str(int(pros_perc * 100)))
    single_run(in_path=in_path, fit=0, elec_price=0.4, ud_penalty=0, out_path=out_path_bc,
                total_budget=np.inf, pros_perc = pros_perc)
    pros_gen = 0
    i = 0
    for _, row in inFile['cap_factors'].set_index('Unnamed: 0').iterrows():
        pros_gen += sum(row) * pros_cap * day_weights[i] * model.max_house[1] * 15
        i += 1
    pros_gens[pros_perc] = pros_gen

base_npvs = []
current_budgets = []

for pros_perc in pros_percs:
    base_path = os.path.join(out_path, "Base Cases", 
                              str(int(pros_perc * 100)), "Output_0_40.xlsx")
    file = pd.read_excel(base_path, sheet_name=None)
    file['Summary'].set_index('Unnamed: 0', inplace=True)
    base_npvs.append(file['Summary'].loc['NPV', 0])
    
    current_cfs = file['Costs and Revenues'].set_index('Unnamed: 0')
    interest = file['Summary'].loc['Interest Rate'][0]
    current_b = 0
    for y in range(len(current_cfs.columns)):
        capex_y = current_cfs.loc['Total Capital Costs'][y]
        current_b += capex_y * (1 / (1 + interest) ** y)
    current_budgets.append(current_b)

in_path = os.path.join(cwd, 'Inputs', 'inputs_RE.xlsx')

# for i, pros_perc in enumerate(pros_percs):
#     fit_search(in_path, out_path, prices, total_budget=current_budgets[i], 
#                pros_perc=pros_perc, search='pros', base_npv = base_npvs[i],
#                md_level=np.inf)
    # multi_run(in_path=in_path, fits=fits, elec_prices=prices_gs, 
    #           out_path=out_path_gs, total_budget=current_budgets[i], 
    #           pros_perc=pros_perc, index='pros', md_level=np.inf)
    
summary_path_5 = os.path.join(out_path, 'Summary.xlsx')
func.eval_summary(os.path.join(cwd, 'Outputs', '3. Prosumer percentage', 
                                'Grid Search', 'Output Files'),
                  pros_gens,
                  max_fits = summary_path_5, index='pros')

npvs = []
hess = []
salvages = []
budgets = []

to_check = [f"Output_0_{int(p * 100)}.xlsx" for p in prices]
inFile = pd.read_excel(in_path, sheet_name = None)
for f in to_check:
    out = pd.read_excel(os.path.join(out_path_gs, 'Output Files', '0', f), 
                        sheet_name = None)
    npv = out['Summary'].set_index("Unnamed: 0").loc['NPV']
    npvs.append(npv)
    hes = out['Summary'].set_index("Unnamed: 0").loc["Household Surplus"]
    hess.append(hes)
    capex = inFile['capex'].set_index("Unnamed: 0")
    budget = 0
    d_factors = [1/(1.1)** y for y in range(15)]
    for _, row in out['Added Capacities'].set_index('Unnamed: 0').iterrows():
        expenses = np.multiply(capex[_], row)
        budget += np.dot(expenses, d_factors)
    budgets.append(budget)


# Constrained Prosumer % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
in_path = os.path.join(cwd, 'Inputs', 'inputs.xlsx')
out_path_b = os.path.join(cwd, 'Outputs', '4. Prosumer percentage constrained')
out_path_gs_b = os.path.join(out_path_b, 'Grid Search')
out_path = os.path.join(cwd, 'Outputs', '3. Prosumer percentage')

prices = np.arange(0, 0.41, 0.01)
prices_gs = np.arange(0, 0.41, 0.01)
fits = np.arange(0, 0.26, 0.01)

md_levels = []
interest_reg = 0.04
pros_percs = [0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1]

inFile = pd.read_excel(in_path, sheet_name=None)
day_weights = inFile['day_weights']['Weight'].tolist()

for i, pros_perc in enumerate(pros_percs):
    base_path = os.path.join(out_path, "Base Cases", 
                              str(int(pros_perc * 100)), "Output_0_40.xlsx")
    file = pd.read_excel(base_path, sheet_name=None)
    summ = file['Summary'].set_index('Unnamed: 0')
    md_level = summ.loc["Unmet Demand", 0]
    md_levels.append(md_level)

in_path = os.path.join(cwd, 'Inputs', 'inputs_RE.xlsx')

for i, pros_perc in enumerate(pros_percs):
    fit_search(in_path, out_path_b, prices, total_budget=current_budgets[i], 
                 pros_perc=pros_perc, search='pros', base_npv = base_npvs[i],
                 md_level=md_levels[i])
    multi_run(in_path=in_path, fits=fits, elec_prices=prices_gs, 
              out_path=out_path_gs_b, total_budget=current_budgets[i], 
              pros_perc=pros_perc, index='pros', md_level=md_levels[i])
    
summary_path_5 = os.path.join(out_path_b, 'Summary.xlsx')
func.eval_summary(os.path.join(out_path_gs_b, 'Output Files'),
                  pros_gens,
                  max_fits = summary_path_5, index='pros')

# Constrained Budget ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
in_path = os.path.join(cwd, 'Inputs', 'inputs_RE.xlsx')
out_path_b = os.path.join(cwd, 'Outputs', '5. Budget constrained')
out_path_gs_b = os.path.join(out_path_b, 'Grid Search')
out_path = os.path.join(cwd, 'Outputs', '5. Budget constrained')

#   Creating Bugdet Range
budgets = np.arange(250000, 2000001, 250000)
budgets = budgets.tolist()
budgets.insert(1, int(current_budget // 1e5 * 1e5))
budgets.remove(500000)
budgets.insert(0, 100000)
budgets.insert(3, 420000)
budgets = [1000000, 1250000, 1500000, 1750000, 2000000]
prices = np.arange(0, 0.41, 0.01)
prices_gs = np.arange(0, 0.41, 0.01)
fits = np.arange(0, 0.26, 0.01)

for budget in budgets:
    fit_search(in_path, out_path_b, prices, re_level=0,
                total_budget=budget, search='budget', md_level = current_md)
    multi_run(in_path=in_path, fits=fits, elec_prices=prices_gs, 
              out_path=out_path_gs_b, re_level=0, 
              total_budget=budget, md_level = current_md)

#   Finding total prosumer generation
inFile = pd.read_excel(in_path, sheet_name=None)
day_weights = inFile['day_weights']['Weight'].tolist()
pros_cap = sum(inFile['rent_cap'].set_index('Unnamed: 0').iloc[1])
pros_gen = 0
i = 0
for _, row in inFile['cap_factors'].set_index('Unnamed: 0').iterrows():
    pros_gen += sum(row) * pros_cap * day_weights[i] * 250 * 15
    i += 1

summary_path_1 = os.path.join(outFile_sum, '5. Budget constrained', 'Summary.xlsx')
func.eval_summary(os.path.join(cwd, 'Outputs', '5. Budget constrained', 
                                'Grid Search', 'Output Files'), pros_gen,
                  max_fits = summary_path_1)
'''