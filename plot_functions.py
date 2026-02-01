# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 22:18:29 2025

@author: Elsa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import PathPatch
from scipy.interpolate import griddata
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patches
from collections import defaultdict

import os

def rep_day(outFile, year, day, multi=1):
    '''Plot some output data and save to folder'''
    
    sns.set(font_scale=1)
    sns.set_style("whitegrid")
    
    new_plots_folder = os.path.join(outFile, "..")

    if multi == 1:
        re_folder = os.path.basename(os.path.dirname(outFile))
        new_plots_folder = os.path.join(new_plots_folder, "..", "..",
                                        "Daily Generation", re_folder)
        os.makedirs(new_plots_folder, exist_ok=True)

    out = pd.read_excel(outFile, sheet_name=None)

    file = os.path.basename(outFile)
    fit = int(file.split('_')[1])
    el_price = int(file.split('_')[2].split('.')[0])

    disp_dg = out['DG Dispatch'].set_index('Unnamed: 0')
    disp_pv = out['PV Dispatch'].set_index('Unnamed: 0')
    bat_in = out['Battery Input '].set_index('Unnamed: 0')
    bat_out = out['Battery Output'].set_index('Unnamed: 0')
    feed_in = out['Fed-in Capacity'].set_index('Unnamed: 0')
    tot_dem = out['Yearly demand'].set_index('Unnamed: 0')
    net_sur = out['Net surplus'].set_index('Unnamed: 0')
    ud = out['Unmet Demand'].set_index('Unnamed: 0')
    
    sur = net_sur - feed_in

    fig, ax = plt.subplots(figsize=(12, 8))

    # Representative day
    index = float(f'{year}' + '.' + f'{day}')
    ax.bar(np.arange(24), bat_in.loc[index] * -1,
           width=0.5, label='Battery Input', color='#b1b1b1')
    ax.bar(np.arange(24), disp_dg.loc[index],
           bottom = feed_in.loc[index],
           width=0.5, label='DG', color='#d14b4b')
    ax.bar(np.arange(24), disp_pv.loc[index],
           bottom=disp_dg.loc[index] + feed_in.loc[index],
           width=0.5, label='PV', color='#f9e395')
    ax.bar(np.arange(24), feed_in.loc[index],
           bottom= 0,
           width=0.5, label='Feed in', color='#dbe4ed')
    ax.bar(np.arange(24), bat_out.loc[index],
           bottom=(disp_dg.loc[index]
                   + disp_pv.loc[index] + feed_in.loc[index]),
           width=0.5, label='Battery Output', color='#c2deaf')
    ax.bar(np.arange(24), ud.loc[index],
           bottom=(disp_dg.loc[index]
                   + disp_pv.loc[index] + feed_in.loc[index]
                   + bat_out.loc[index]),
           width=0.5, label='Unmet Demand', color='#f2b382')

    ax.plot(np.arange(24), tot_dem.loc[index].to_numpy() * -1,
            label='Total Demand', color='#595755')
    ax.plot(np.arange(24), sur.loc[index].to_numpy(),
            label='Surplus', linestyle='dashed',
            color='#595755')

    ax.set_xlabel('Hour')
    ax.set_ylabel('Energy')
    '''
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=4,
              bbox_transform=fig.transFigure,
              frameon=False)
    '''
    if multi == 0:
        ax.set_yticks([i for i in range (0, 901, 100)])
    
    plt.subplots_adjust(top=.85)
    plot_path = os.path.join(new_plots_folder, 
                             f"Daily_gen_{fit}_{el_price}_{year}-{day}.png")
    plt.savefig(plot_path)
    plt.close()
    

def inst_cap(outFile, multi=1):
    '''Plot installed capacities and save to folder'''
    
    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")
    new_plots_folder = os.path.join(outFile, "..")

    if multi == 1:
        re_folder = os.path.basename(os.path.dirname(outFile))
        new_plots_folder = os.path.join(new_plots_folder, "..", "..",
                                        "Installed capacities", re_folder)
        os.makedirs(new_plots_folder, exist_ok=True)

    inst = pd.read_excel(outFile, sheet_name='Installed Capacities')
    inst.set_index('Unnamed: 0', inplace=True)

    file = os.path.basename(outFile)
    fit = int(file.split('_')[1])
    el_price = int(file.split('_')[2].split('.')[0])


    fig, ax = plt.subplots(figsize=(8,6))

    ax.bar(np.arange(15), inst.loc['Diesel Generator'],
           width=0.5, label='Diesel generator', color="#d14b4b")
    ax.bar(np.arange(15), inst.loc['Owned PV'],
           width=0.5, label='DGC PV',
           bottom=inst.loc['Diesel Generator'], color="#f9e395")
    ax.bar(np.arange(15), inst.loc['Owned Batteries'],
           width=0.5, label='DGC batteries',
           bottom=inst.loc['Owned PV'] + inst.loc['Diesel Generator'],
           color="#c2deaf")

    ax.set_xlabel('Year')
    ax.set_ylabel('Capacity installed (kW)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, .94), ncol=3,
              bbox_transform=fig.transFigure, frameon=False)
    
    if multi == 1:
        ax.set_yticks([i for i in range (0, 1001, 250)])
      
    plt.subplots_adjust(top=.85)
    plot_path = os.path.join(new_plots_folder, 
                             f"Installed_Capacities_{fit}_{el_price}.png")
    plt.savefig(plot_path)
    plt.close()
    

def gen_year(outFile, multi=1):
    
    sns.set(font_scale=1)
    sns.set_style("whitegrid")
    new_plots_folder = os.path.join(outFile, "..")

    if multi == 1:
        re_folder = os.path.basename(os.path.dirname(outFile))
        new_plots_folder = os.path.join(new_plots_folder, "..", "..",
                                        "Yearly generation", re_folder)
        os.makedirs(new_plots_folder, exist_ok=True)

    out = pd.read_excel(outFile, sheet_name=None)

    file = os.path.basename(outFile)
    fit = int(file.split('_')[1])
    el_price = int(file.split('_')[2].split('.')[0])
    
    disp_dg = out['DG Dispatch'].set_index('Unnamed: 0')
    disp_pv = out['PV Dispatch'].set_index('Unnamed: 0')
    bat_out = out['Battery Output'].set_index('Unnamed: 0')
    bat_in = out['Battery Input '].set_index('Unnamed: 0')
    feed_in = out['Fed-in Capacity'].set_index('Unnamed: 0')
    unmet = out['Unmet Demand'].set_index('Unnamed: 0')
    
    weights = {0: 199,
               1: 106,
               2: 60}
    
    year_dg = []
    for y in range(15):
        tot_dg_y = 0
        for d in range(3):
            tot_dg_y += sum(disp_dg.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
        year_dg.append(tot_dg_y / 1e6)
        
    year_pv = []
    for y in range(15):
        tot_pv_y = 0
        for d in range(3):
            tot_pv_y += sum(disp_pv.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
        year_pv.append(tot_pv_y / 1e6)
    
    year_bat_in = []
    for y in range(15):
        tot_bat_in_y = 0
        for d in range(3):
            tot_bat_in_y += sum(bat_in.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
        year_bat_in.append(- tot_bat_in_y / 1e6)

    year_bat_out = []
    for y in range(15):
        tot_bat_out_y = 0
        for d in range(3):
            tot_bat_out_y += sum(bat_out.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
        year_bat_out.append(tot_bat_out_y / 1e6)

    year_fed_in = []
    for y in range(15):
        tot_fed_in_y = 0
        for d in range(3):
            tot_fed_in_y += sum(feed_in.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
        year_fed_in.append(tot_fed_in_y / 1e6)

    year_ud = []
    for y in range(15):
        tot_ud_y = 0
        for d in range(3):
            tot_ud_y += sum(unmet.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
        year_ud.append(tot_ud_y / 1e6)
        
        
    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.bar(np.arange(15), year_bat_in,
           width=0.5, label='Charge', color='#b1b1b1')
    ax.bar(np.arange(15), year_dg,
           bottom = year_fed_in,
           width=0.5, label='Diesel Generator', color='#d14b4b')
    ax.bar(np.arange(15), year_pv,
           bottom=np.add(year_dg, year_fed_in),
           width=0.5, label='DGC PV', color='#f9e395')
    ax.bar(np.arange(15), year_fed_in,
           bottom= 0,
           width=0.5, label='Feed in', color='#dbe4ed')
    ax.bar(np.arange(15), year_bat_out,
           bottom=np.add(np.add(year_dg, year_fed_in), year_pv),
           width=0.5, label='Discharge', color='#c2deaf')
    ax.bar(np.arange(15), year_ud,
           bottom=np.add(np.add(np.add(year_dg, year_fed_in), year_pv),
                         year_bat_out),
           width=0.5, label='Unmet Demand', color='#f2b382')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Energy (MWh)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3,
              bbox_transform=fig.transFigure, frameon=False)
    
    if multi == 1:
       ax.set_yticks([i / 1e6 for i in range (-1500000, 3000001, 500000)]) 
    
    plt.subplots_adjust(top=.75)
    plot_path = os.path.join(new_plots_folder, 
                             f"Yearly_gen_{fit}_{el_price}.png")
    plt.savefig(plot_path)
    plt.close()
        
def add_ret(outFile, multi=1):

    sns.set(font_scale=1.5)
    #sns.set_style("whitegrid")
    new_plots_folder = os.path.join(outFile, "..")

    if multi == 1:
        re_folder = os.path.basename(os.path.dirname(outFile))
        new_plots_folder = os.path.join(new_plots_folder, "..", "..",
                                        "Added and retired capacities",
                                        re_folder)
        os.makedirs(new_plots_folder, exist_ok=True)

    out = pd.read_excel(outFile, sheet_name=None)
    added = out['Added Capacities'].set_index("Unnamed: 0")
    ret = out['Retired Capacities'].set_index("Unnamed: 0")

    file = os.path.basename(outFile)
    fit = int(file.split('_')[1])
    el_price = int(file.split('_')[2].split('.')[0])
    
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(np.arange(15), added.loc['Diesel Generator'],
           width=0.5, label='Added diesel generator', color="#d14b4b")
    ax.bar(np.arange(15), added.loc['Owned PV'],
           width=0.5, label='Added PV',
           bottom=added.loc['Diesel Generator'], color="#f9e395")
    ax.bar(np.arange(15), added.loc['Owned Batteries'],
           width=0.5, label='Added batteries',
           bottom=added.loc['Owned PV'] + added.loc['Diesel Generator'],
           color="#c2deaf")
    
    ax.bar(np.arange(15), ret.loc['Diesel Generator'] * -1,
           width=0.5, label='Retired diesel generator', color="#d14b4b",
           hatch='\\')
    ax.bar(np.arange(15), ret.loc['Owned PV'] * -1,
           width=0.5, label='Retired PV',
           bottom=added.loc['Diesel Generator'] * -1, color="#f9e395",
           hatch='\\')
    ax.bar(np.arange(15), ret.loc['Owned Batteries'] * -1,
           width=0.5, label='Retired batteries',
           bottom=-ret.loc['Owned PV'] - ret.loc['Diesel Generator'],
           color="#c2deaf", hatch='\\')
    
    ax.plot(np.arange(15), [0]*15, color='#595755')

    ax.set_xlabel('Year')
    ax.set_ylabel('Capacity (kW)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2,
              bbox_transform=fig.transFigure,
              frameon=False)

    if multi == 1:
        ax.set_yticks([i for i in range (-500, 551, 100)]) 
     
    plt.subplots_adjust(top=0.8)
    plt.tight_layout(pad=1)
    plot_path = os.path.join(new_plots_folder, f"Add_Ret_Capacities_{fit}_{el_price}.png")
    plt.savefig(plot_path)
    plt.close()
    
def get_npv(casePath):
    
    new_plots_folder = os.path.join(casePath, "NPV.png")
    outFile = os.path.join(casePath, "Output Files")

    files = os.listdir(outFile)

    npvs = []
    fits = []
    prices = []
    
    for file in files:
        fit = int(file.split('_')[1])
        price = int(file.split('_')[2].split('.')[0])
        summary = pd.read_excel(os.path.join(outFile, file), 
                                sheet_name='Summary')
        summary.set_index("Unnamed: 0", inplace=True)
        npv = summary.loc["Household Surplus"]
        npvs.append(npv)
        fits.append(fit)
        prices.append(price)
    
    x_grid = np.linspace(min(prices), max(prices), 5)  # Adjust resolution (50x50 grid)
    y_grid = np.linspace(min(fits), max(fits), 5)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Interpolate Z values onto the structured grid
    Z_grid = griddata((prices, fits), npvs, (X_grid, Y_grid), method='cubic')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis')
    
    ax.set_xlabel("Price")
    ax.set_ylabel("FiT")
    ax.set_zlabel("NPV")
    
    plt.show()
    plt.savefig(new_plots_folder, dpi=300, bbox_inches='tight')
    
    
def fit_v_price(casePath, search='re', keys=None, colors = None, 
                max_sol = None):
    sns.set(font_scale=1.2)
    sns.set_style('whitegrid')
    
    name_map = {'pros': 'Prosumer percentage',
                'budget': 'Budget'}
    new_plots_folder = os.path.join(casePath, "FiTs v Prices.png")
    if len(keys) == 1:
        new_plots_folder = os.path.join(casePath, f"FiTs v Prices_{keys[0]}.png")
    outFile = os.path.join(casePath, "Summary.xlsx")
    
    out = pd.read_excel(outFile, sheet_name=None)
    summary_fi_path = os.path.join(casePath, "Summary feed-in.xlsx")
    fi_sheets = pd.read_excel(summary_fi_path, sheet_name=None)

    fig, ax = plt.subplots(figsize=(7,5))
    if colors == None:
        colors = ["#595755", "#6d597a", "#DA4167" ,
                  "#f2b382", "#f4d35e", "#85a4c4", 
                  "#c2deaf" ]
    i = 0
    show_infeasible_label = True
    
    if keys == None:
        keys = list(out.keys())
        if search == 're':
            if len(keys) >= 7:
                keys = keys[:7]
                
        elif search == 'budget':
            if len(keys) >= 7:
                keys = keys[:7]
                keys[::-1]
                
    for i, key in enumerate(keys):
        try:
            out[key].set_index('Unnamed: 0', inplace=True)
        except KeyError:
            key = str(key)
            out[key].set_index('Unnamed: 0', inplace=True)
        fits = out[key].loc['Feed-in Tariffs'].to_list()
        unfeas_fits = [fit for fit in fits if fit == 0]
        prices = out[key].loc['Prices'].to_list()
        
        if search == 're' or search == 'pros':
            label = f'{int(float(key) * 100)}%'
        elif search == 'budget':
            label = f'USD {float(key)/ 1e6} M'
        if len(keys) == 1:
            label = 'Policies with status-quo DGC NPV'
        
        full_fits = np.array(fits)
        price_arr = np.array(prices)
        
        mask = np.isnan(full_fits)
        full_fits[mask] = price_arr[mask]
        
        price_plot = prices[len(unfeas_fits) - 1 ::]
        fit_plot = full_fits.tolist()[len(unfeas_fits) - 1 ::]
        
        ax.plot(price_plot, 
                fit_plot, 
                linestyle='-', color=colors[i], 
                zorder=2 if show_infeasible_label else 1,
                label=label)
        
        
        mask_nonan = ~np.isnan(np.array(fits))
        index = np.where(mask_nonan)[0][-1]

        P_n = np.array(prices)[mask]
        P_n = P_n.tolist()
        P_n.insert(0, prices[index])
        FiT_last = full_fits[mask]
        FiT_last = FiT_last.tolist()
        FiT_last.insert(0, fits[index])

        ax.fill_between(price_plot, 
                        fit_plot,
                        alpha=0.3, color=colors[i])
        
        if str(key) in fi_sheets:
            df_fi = fi_sheets[str(key)].set_index("Unnamed: 0")
            lower_bound = df_fi.iloc[0].tolist()
            price_vals = [float(p) for p in df_fi.columns]
            price_vals.sort()
        
            poly = ax.fill_between(price_vals,
                            lower_bound,
                            fit_plot[-len(price_vals):],
                            alpha=1, facecolor='none', hatch='//',
                            zorder=len(keys) - i,
                            edgecolor=colors[i],
                            linewidth=0.5)
                            # label='Policies with limited feed-in' if i==0 else None)
            path = poly.get_paths()[0]
            clip_patch = PathPatch(path, transform=ax.transData)
            poly.set_clip_path(clip_patch)
        show_infeasible_label = False
    
    plt.axvline(x=0.4, color='red', linestyle='--', linewidth=1,
                label= "Current price")
    
    i = 0
    if max_sol != None:
        max_sol_df = pd.read_excel(max_sol).set_index(name_map[search])
        for _, row in max_sol_df.iterrows():
            if _ in keys:
                opt_p = row['Price']
                opt_fit = row['FiT']
                ax.scatter([opt_p], [opt_fit], color=colors[i])
                i += 1
            
    ax.set_xlabel('Price (USD)')
    ax.set_ylabel('Maximum Feed-in Tariff (USD)')

    if search == 're' or search == 'pros':
        ax.legend(title = 'Minimum Renewable Energy Generation Target',
                  loc='upper center',
                  bbox_to_anchor=(0.5, 1.25),
                  ncol=3 if len(keys) > 1 else 2,
                  frameon=False)
    elif search == 'budget':
        handles, labels = ax.get_legend_handles_labels()
        hatch_proxy = mpatches.Patch(facecolor='none', edgecolor='grey', hatch='///', label='Policies with limited feed-in')
        highlight_proxy = mpatches.Patch(facecolor='lightgrey', alpha=0.5, label='Policies with feed-in')
        
        handles, labels = ax.get_legend_handles_labels()
        handles += [highlight_proxy, hatch_proxy]
        labels += ['Policies with feed-in', 'Policies with limited feed-in']
        ax.legend(handles[::-1], labels[::-1],
                  title = 'Budget Available (USD)' if len(keys) > 1 else "",
                  loc='upper center',
                  bbox_to_anchor=(0.5, 1.35),
                  ncol=3 if len(keys) > 1 else 2,
                  frameon=False)

    ax.set_xticks(np.arange(0.25, 0.41, 0.02))
    
    sns.set_style("whitegrid")
    
    plt.subplots_adjust(top=0.65 if len(keys)>1 else 0.85)
    plt.tight_layout(pad=1)

    plt.savefig(new_plots_folder)
    plt.close()

def fi_gradient(casePath, key, day_weights=[199, 106, 60]):
    sns.set(font_scale=1.2)
    sns.set_style('whitegrid')
    
    new_plots_folder = os.path.join(casePath, "Grid Search", f"Feed-in perc {key}.png")
    
    filePaths = os.path.join(casePath, "Grid Search", 
                             "Output Files", str(key))
    summaryFile = os.path.join(casePath, "Summary (initial).xlsx")
    files = os.listdir(filePaths)
    
    data = {'Prices ($)': [],
            'FiTs ($)': [],
            'Feed-in': []}

    for file in files:
        price = int(file.split('_')[2].split('.')[0]) / 100
        fit = int(file.split('_')[1]) / 100
        if price >= 0.26:
            data['Prices ($)'].append(price)
            data['FiTs ($)'].append(fit)
            out = pd.read_excel(os.path.join(filePaths, file), 
                                sheet_name=None)
            out_fi = out['Fed-in Capacity']
            out_fi.set_index("Unnamed: 0", inplace=True)
            fi = 0
            for _, row in out_fi.iterrows():
                day = int(str(_).split('.')[1])
                fi += sum(row) * day_weights[day]
            ud = out['Summary'].set_index("Unnamed: 0").loc["Unmet Demand", 0]
            data['Feed-in'].append(fi / (30789066.4 - ud) * 100)

    
    price_index = list(dict.fromkeys(data['Prices ($)']))
    price_index.sort()
    fit_index = list(dict.fromkeys(data['FiTs ($)']))
    fit_index.sort()
    
    df = pd.DataFrame(data)
    
    
    heatmap_data = df.pivot_table(index='FiTs ($)', 
                                  columns='Prices ($)', 
                                  values='Feed-in')
    
    mask = np.zeros_like(heatmap_data, dtype=bool)
    

    max_fits = pd.read_excel(summaryFile, sheet_name = str(key))
    max_fits.set_index("Unnamed: 0", inplace=True)
    
    for i, price in enumerate(price_index):
        row = max_fits.loc['Prices']
        price_col = row[row == price].index.tolist()
        
        try:
            max_fit = max_fits[price_col[0]]['Feed-in Tariffs']
            if np.isnan(max_fit):
                max_fit = price
            # max_fit = round(max_fit, 2)
            for j, fit in enumerate(fit_index):
                if fit > max_fit or max_fit == 0:
                    mask[j, i] = True
                    
        except IndexError:
            print(f'WAA {max_fit}, {price}')
                
    plt.figure(figsize=(8, 6))

    
    sns.heatmap(heatmap_data, fmt=".1f", 
                mask=mask, annot=False)


    plt.gca().invert_yaxis()
    plt.xlabel("Prices (USD)")
    plt.ylabel("FiTs (USD)")
    plt.tight_layout()
    plt.savefig(new_plots_folder)
    plt.close()        
    
def surp_heatmap(casePath, key, index = 're', max_fits=None, # summary file
                 p_lb=0, p_ub=np.inf, fit_lb=0 , fit_ub=np.inf): 
    
    global data
    global files
    sns.set(font_scale=1.3)
    assert (index == 're'
            or index == 'budget'
            or index == 'pros')
    
    new_plots_folder = os.path.join(casePath, f"Surplus heatmap {key}.png")
    
    if index == 're' or index == 'pros':
        endFile = str(int(key * 100))
    else:
        endFile = str(key)
    filesPath = os.path.join(casePath, 'Output Files', endFile)
    files = os.listdir(filesPath)
    
    data = {'Prices ($)': [],
            'FiTs ($)': [],
            'Surpluses': []}

    for file in files:
        price = int(file.split('_')[2].split('.')[0]) / 100
        fit = int(file.split('_')[1]) / 100
        if price > p_lb and price < p_ub and fit > fit_lb and fit < fit_ub:
            data['Prices ($)'].append(price)
            data['FiTs ($)'].append(fit)
            out = pd.read_excel(os.path.join(filesPath, file), sheet_name='Summary')
            out.set_index("Unnamed: 0", inplace=True)
            data['Surpluses'].append(out.loc["Household Surplus"][0])

    
    price_index = list(dict.fromkeys(data['Prices ($)']))
    price_index.sort()
    fit_index = list(dict.fromkeys(data['FiTs ($)']))
    fit_index.sort()
    
    df = pd.DataFrame(data)
    
    
    heatmap_data = df.pivot_table(index='FiTs ($)', 
                                  columns='Prices ($)', 
                                  values='Surpluses')
    
    mask = np.zeros_like(heatmap_data, dtype=bool)
    
    if max_fits is not None:
        max_fits = pd.read_excel(max_fits, sheet_name = str(key))
        max_fits.set_index("Unnamed: 0", inplace=True)
        
        for i, price in enumerate(price_index):
            row = max_fits.loc['Prices']
            price_col = row[row == price].index.tolist()
            
            try:
                max_fit = max_fits[price_col[0]]['Feed-in Tariffs']
                if np.isnan(max_fit):
                    max_fit = price
                # max_fit = round(max_fit, 2)
                for j, fit in enumerate(fit_index):
                    if fit > max_fit or max_fit == 0:
                        mask[j, i] = True
                        
            except IndexError:
                print(f'{index} key={key}, price={price}')
    
    visible_data = heatmap_data.mask(mask)
    max_row, max_col = np.unravel_index(np.nanargmax(visible_data.values),
                                        visible_data.shape)
    
    heatmap_cat = pd.cut(heatmap_data.stack(),
                         bins=[-np.inf, 6.5 * 1e6, 7.5 * 1e6, 
                               8.5 * 1e6, 9.5 * 1e6, 
                               np.inf],
                         labels=[0, 1, 2, 3, 4],
                         include_lowest=True
                         ).unstack().astype(int)
        
    plt.figure(figsize=(8, 6))

    max_val = round(heatmap_data.max().max() / 1e6, 1) 

    if max_val > 9.5 :
        last_label = f"9.5–{max_val:.1f}"
    else:
        last_label = ">9.5"

    
    cmap = ListedColormap(["#f7f0cc", "#fcdbc3", "#7bdbb1", "#64b985", "#2f847e"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    ax = sns.heatmap(heatmap_cat, fmt=".1f", 
                     cmap=cmap, 
                     cbar_kws={'label': 'Household Economic Surplus (M USD)',
                               'ticks':[0, 1, 2, 3, 4]},
                     mask=mask, annot=False,
                     norm=norm)
    
    ax.add_patch(patches.Rectangle(
        (max_col, max_row),  
        1, 1,
        fill=False,
        edgecolor="red",
        linewidth=2,
        zorder=5            
    ))
    
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticklabels(["<6.5", "6.5–7.5", "7.5-8.5",
                             "8.5-9.5", last_label])

    plt.gca().invert_yaxis()
    plt.xlabel("Prices (USD)")
    plt.ylabel("FiTs (USD)")
    plt.tight_layout()
    plt.savefig(new_plots_folder)
    plt.close()
    
def ud_comp(casePaths, re_levels): # re_levels in %
    
    assert type(casePaths)== list, "casePaths should be a list"
    
    new_plots_folder = os.path.join(casePaths[0], '..', 'UD comparison.png')
    
    cases = []
    sns.set(font_scale = 2)
    colors = ["#f4d35e", "#85a4c4", "#c2deaf" ]
    i = 0
    bar_data = {}
    
    name_mapping = {
    'No PV': 'Feed-in Only',
    'No PV w Bat': 'Feed-in + Batteries',
    'With PV': 'Feed-in + PV + Batteries'
    }
    
    bar_data['RE target'] = re_levels
    for casePath in casePaths:
        original_name = os.path.basename(casePath)
        readable_name = name_mapping.get(original_name, original_name)
        cases.append(readable_name)
        case_summary = pd.read_excel(os.path.join(casePath, 
                                                   'Evaluation Metrics.xlsx'))
            
        if len(case_summary['Unmet Demand']) < len(re_levels):
            temp = case_summary['Unmet Demand'] * 100
            temp = list(temp)
            temp += [np.nan] * (len(re_levels) 
                               - len(case_summary['Unmet Demand']))
            bar_data[cases[-1]] = temp
        elif len(case_summary['Unmet Demand']) == len(re_levels):
            bar_data[cases[-1]] = case_summary['Unmet Demand'] * 100
        i += 1
        
    df_data = pd.DataFrame(bar_data)
    df_data_melted = df_data.melt('RE target', var_name='Unmet Demand', 
                                    value_name='Value')
    plt.figure(figsize=(15, 7))
    hue_order = ['Feed-in Only', 
                 'Feed-in + Batteries', 
                 'Feed-in + PV + Batteries']
    color_map = dict(zip(hue_order, colors))
    
    ax = sns.barplot(x='RE target', y='Value', hue='Unmet Demand', 
                     data=df_data_melted, palette=colors, hue_order=hue_order)
    
    re_targets = df_data['RE target'].values
    x_locs = ax.get_xticks()
    width_per_bar = 0.8 / len(hue_order) 


    for i, re_target in enumerate(re_targets):
        for j, case in enumerate(hue_order):
            y_val = df_data[case].iloc[i]
            if pd.isna(y_val):
                x_pos = x_locs[i] - 0.4 + width_per_bar * (j + 0.5)
                ax.scatter(x_pos, -0.1, marker='x', color=color_map[case], 
                           s=100, linewidths=2)
    
    handles, labels = ax.get_legend_handles_labels()
    black_x = Line2D([], [], color='black', marker='x', linestyle='None',
                     markersize=10, label='Infeasible', mew=2)
    handles.append(black_x)
    labels.append('Infeasible')
                
    plt.xlabel('RE Target (%)')
    plt.ylabel('Unmet Demand (%)')
    ax.legend(handles=handles, labels=labels,
              title='Case',
              loc='upper center',
              bbox_to_anchor=(0.5, 1.23),
              ncol=4,
              frameon=False)
    plt.subplots_adjust(top=.85) 
    plt.subplots_adjust(bottom = .13)
    plt.yticks(np.arange(0, 81, 20))
    
    plt.savefig(new_plots_folder)
    plt.close()
    
def ws_comp(casePaths, re_levels): 
    
    assert type(casePaths)== list, "casePaths should be a list"
    
    new_plots_folder = os.path.join(casePaths[0], '..', 'WS comparison.png')
    
    cases = []
    sns.set(font_scale = 2)
    colors = ["#f4d35e", "#85a4c4", "#c2deaf" ]
    i = 1
    bar_data = {}
    
    name_mapping = {
    'No PV': 'Feed-in Only',
    'No PV w Bat': 'Feed-in + Batteries',
    'With PV': 'Feed-in + PV + Batteries'
    }
    
    bar_data['RE target'] = re_levels
    
    for casePath in casePaths:
        original_name = os.path.basename(casePath)
        readable_name = name_mapping.get(original_name, original_name)
        cases.append(readable_name)
        case_summary = pd.read_excel(os.path.join(casePath, 
                                                   'Evaluation Metrics.xlsx'))
        if len(case_summary['Wasted Surplus']) <= len(re_levels):
            temp = case_summary['Wasted Surplus'] * 100
            temp = list(temp)
            temp += [np.nan] * (len(re_levels) 
                               - len(case_summary['Wasted Surplus']))
            bar_data[cases[-1]] = temp
        elif len(case_summary['Wasted Surplus']) == len(re_levels):
            bar_data[cases[-1]] = case_summary['Wasted Surplus'] * 100
            
        bar_data[cases[-1]] = case_summary['Wasted Surplus'] * 100
        i += 1
        
    df_data = pd.DataFrame(bar_data)
    df_data_melted = df_data.melt('RE target', var_name='Wasted Surplus', 
                                    value_name='Value')
    plt.figure(figsize=(15, 7))
    
    hue_order = ['Feed-in Only', 
                 'Feed-in + Batteries', 
                 'Feed-in + PV + Batteries']
    color_map = dict(zip(hue_order, colors))
    
    ax = sns.barplot(x='RE target', y='Value', hue='Wasted Surplus', 
                     data=df_data_melted, palette=colors, hue_order=hue_order)
    
    re_targets = df_data['RE target'].values
    x_locs = ax.get_xticks()
    width_per_bar = 0.8 / len(hue_order) 

    for i, re_target in enumerate(re_targets):
        for j, case in enumerate(hue_order):
            y_val = df_data[case].iloc[i]
            if pd.isna(y_val):
                x_pos = x_locs[i] - 0.4 + width_per_bar * (j + 0.5)
                ax.scatter(x_pos, -0.1, marker='x', color=color_map[case], 
                           s=100, linewidths=2)
    
    handles, labels = ax.get_legend_handles_labels()
    black_x = Line2D([], [], color='black', marker='x', linestyle='None',
                     markersize=10, label='Infeasible', mew=2)
    handles.append(black_x)
    labels.append('Infeasible')

    plt.xlabel('RE Target (%)')
    plt.ylabel('Wasted Surplus (%)')
    ax.legend(handles=handles, labels=labels,
              title='Case',
              loc='upper center',
              bbox_to_anchor=(0.5, 1.23),
              ncol=4,
              frameon=False)
    #plt.subplots_adjust(top=.85) 
    #plt.subplots_adjust(bottom = .13)
    
    plt.yticks(np.arange(0, 81, 20))
    
    plt.savefig(new_plots_folder)
    plt.close()
    
def re_comp(casePath, index='re', addCurrent=None):
    
    sns.set(font_scale=1.65)
    
    new_plots_folder_ud = os.path.join(casePath, f'UD+WS+{index} comparison.png')
    new_plots_folder_hs = os.path.join(casePath, f'HS+{index} comparison.png')
    new_plots_folder_pf = os.path.join(casePath, f'P+FiT+{index} comparison.png')
    
    fig_ud, ax_ud = plt.subplots(figsize=(10, 6))

    
    # Get summary from evaluation metrics files
    try:
        eval_sum = pd.read_excel(os.path.join(casePath, 
                                              'Grid Search',
                                              'Evaluation Metrics.xlsx'))
    except FileNotFoundError:
        eval_sum = pd.read_excel(os.path.join(casePath,
                                              'Evaluation Metrics.xlsx'))
    
    indices = {'re': 'RE target',
               'budget': 'Budget',
               'pros': 'Prosumer percentage'}
    eval_sum = eval_sum.sort_values(by=indices[index])
    
    # Plot
    if index == 'budget':
        width = 0.3
        x = np.array(eval_sum[indices[index]]) / 1e6
        #x = np.delete(x, 1)
        y_ud = np.array(eval_sum['Unmet Demand']) * 100
        y_ws = np.array(eval_sum['Wasted generation (from total)']) * 100
        y_hs = np.array(eval_sum['Household Surplus']) / 1e6
        y_p = np.array(eval_sum['Price'])
        y_fit = np.array(eval_sum['FiT'])
    else:
        width = 0.3
        x = np.array(eval_sum[indices[index]])
        y_ud = np.array(eval_sum['Unmet Demand']) * 100
        y_ws = np.array(eval_sum['Wasted generation (from total)']) * 100
        y_hs = np.array(eval_sum['Household Surplus']) / 1e6
        y_p = np.array(eval_sum['Price'])
        y_fit = np.array(eval_sum['FiT'])
        
    ax_ud.bar(np.arange(len(x)) - width/2, y_ud,
              width=width,
              color='#f18a75',
              label='Unmet demand')

    ax_ud.bar(np.arange(len(x)) + width/2, y_ws,
              width=width,
              color='#3a737d',
              label='Wasted PV-owner excess generation')
    
    #y_min, y_max = ax_ud.get_ylim()
    #ax_ud.set_ylim(y_min, y_max * 1.5)
    
    if index == 're':
        ax_ud.set_xlabel('RE target (%)')
    elif index == 'budget':
        ax_ud.set_xlabel('Budget (M USD)')
    elif index == 'pros':
        ax_ud.set_xlabel('PV-owners (%)')
    ax_ud.set_ylabel('(%)')
    ax_ud.set_xticks(np.arange(len(x)))
    ax_ud.set_xticklabels(x)
    
    handles, labels = ax_ud.get_legend_handles_labels()
    ax_ud.legend(handles, labels,
                 loc='upper center',
                 bbox_to_anchor=(0.5, 1.1),
                 ncol=2,
                 frameon=False)
    
    #fig_ud.subplots_adjust(top=0.8, left=0.15) 
    
    fig_ud.savefig(new_plots_folder_ud)
    plt.close(fig_ud)
    
    sns.set(font_scale=1.3)
    
    fig_hs, ax_hs = plt.subplots(figsize=(7,5))
    fig_pf, ax_pf = plt.subplots(figsize=(7,5))
    
    label_hs = None
    ncol = 1
    
    if addCurrent != None:
        label_hs = 'Household economic surplus'
        ncol = 1
        if type(addCurrent) == str:
            curr_df = pd.read_excel(addCurrent, sheet_name='Summary')
            curr_df.set_index("Unnamed: 0", inplace=True)
            curr_hs = curr_df.loc["Household Surplus"][0]
            
            ax_hs.plot(x,
                       [curr_hs / 1e6] * len(x),
                       linewidth = 2,
                       linestyle= '--',
                       color = "#595755",
                       label='Status-quo household economic surplus')
        
        elif type(addCurrent) == list:
            curr_hss = []
            for curr in addCurrent:
                curr_df = pd.read_excel(curr, sheet_name='Summary')
                curr_df.set_index("Unnamed: 0", inplace=True)
                curr_hs = curr_df.loc["Household Surplus"][0]
                curr_hss.append(curr_hs)
            
            curr_hss = np.array(curr_hss)
            ax_hs.plot(x,
                       curr_hss /1e6,
                       linewidth = 2,
                       linestyle= '--',
                       color = "#595755",
                       label='Status-quo household economic surplus')
        
    y_new = y_hs.tolist()
    y_new.insert(1,7)
    x_new = x.tolist()
    x_new.insert(1,0.175)
    ax_hs.plot(x_new,
               y_new,
               linewidth = 3,
               color = '#64b985',
               label=label_hs)
    x_budgets = [420000 / 1e6, 175000 / 1e6]

    for x_budget in x_budgets:
        ax_hs.axvline(x=x_budget,
                      linestyle='--',
                      color='grey',
                      linewidth=1,
                      ymin=-0.05,
                      ymax=1.0,
                      clip_on=False)
        
        ax_hs.text(x_budget, 
                   ax_hs.get_ylim()[0] - 0.5,          
                   f'{x_budget}',
                   ha='center',
                   va='top',
                   fontsize=15,
                   rotation=0,
                   clip_on = False)
    
    if index == 'budget':
        ax_hs.set_xlabel(f'{indices[index]} (M USD)')
    elif index == 're':
        ax_hs.set_xlabel(f'{indices[index]} (%)')
    else:
        ax_hs.set_xlabel('PV-owners (%)')
    ax_hs.set_ylabel('Household economic surplus (M USD)')
    ax_hs.legend(loc='upper center',
                 bbox_to_anchor=(0.5, 1.2),
                 ncol=ncol,
                 frameon=False)
    #fig_hs.subplots_adjust(top=0.8, left=0.15) 
    fig_hs.tight_layout(pad=1)
    fig_hs.savefig(new_plots_folder_hs)
    plt.close(fig_hs)
    
    ax_pf.plot(x,
               y_p,
               linewidth = 3,
               color = '#6C3428',
               label = 'Price')
    ax_pf.plot(x,
               y_fit,
               linewidth = 3,
               color = '#6DC5D1',
               label = 'FiT')
    if index == 'budget':
        ax_pf.set_xlabel(f'{indices[index]} (M USD)')
    elif index == 're':
        ax_pf.set_xlabel(f'{indices[index]} (%)')
    else:
        ax_pf.set_xlabel('PV-owners (%)')
    ax_pf.set_ylabel('USD')
    ax_pf.legend(loc='upper center',
                 bbox_to_anchor=(0.5, 1.1),
                 ncol=2,
                 frameon=False)
    
    #fig_pf.subplots_adjust(top=0.8, left=0.15) 
    plt.tight_layout(pad=1)
    fig_pf.savefig(new_plots_folder_pf)
    plt.close(fig_pf)
    
    

def energy_sensitivity(casePath, way_1, s_range_1, 
                       way_2=None, s_2=None, 
                       show_demand=0):

    assert (way_1 == 'budget' 
            or way_1 == 'i'
            or way_1 == 're'
            or way_1 == 'pros'), 'Wrong first sensitivity parameter'
    assert (way_2 == 'budget' 
            or way_2 == 'i'
            or way_2 == 're'
            or way_2 == 'pros'
            or way_2 == None), 'Wrong second sensitivity parameter'
    
    try:
        s_range_1 = s_range_1.tolist()
    except AttributeError:
        pass
    
    sns.set(font_scale=1.75)
    labels = {'budget': 'Budget',
              'i': 'Interest',
              're': 'RE target',
              'pros': 'Prosumer percentage'}

    # Create new figure
    if way_2 == None:
        new_plots_folder = os.path.join(casePath, 'Energy.png')
    elif way_2 == 're' or way_2 == 'pros':
        assert (s_2 != None), "Missing second sensitivity value"
        new_plots_folder = os.path.join(casePath, 
                                        f'Energy {str(int(s_2 * 100))}.png')
    elif way_2 == 'budget':
        assert (s_2 != None), "Missing second sensitivity value"
        new_plots_folder = os.path.join(casePath, 
                                        f'Energy {str(s_2)}.png')
        
    fig, ax = plt.subplots(figsize=(10,8))
    ax2 = ax.twinx()
    
    tot_dg = []
    tot_pv = []
    tot_fi = []
    tot_ud = []
    tot_bat = []
    tot_bat_c = []
    tot_dem = []
    
    for s in s_range_1:        
        # find summary of evaluation metrics
        if way_2 != None:
            summary_df = pd.read_excel(os.path.join(casePath, 
                                                    'Sensitivity',
                                                    str(s),
                                                    'Evaluation Metrics.xlsx'))
            
            summary_df.set_index(labels[way_2], inplace=True)
            
            if way_2 == 're' or way_2 == 'i':
                fit_b = summary_df['FiT'].loc[int(s_2 * 100)]
                fit_b = round(fit_b, 2)
                p_b = round(summary_df['Price'].loc[int(s_2 * 100)], 2)
            else:
                fit_b = summary_df['FiT'].loc[s_2]
                fit_b = round(fit_b, 2)
                p_b = round(summary_df['Price'].loc[s_2], 2)
                
            
            out = pd.read_excel(os.path.join(casePath,
                                         'Sensitivity',
                                         str(s),
                                         'Output Files',
                                         str(int(s_2 * 100)),
                                         f'Output_{int(fit_b*100)}_{int(p_b*100)}.xlsx'),
                            sheet_name=None)
            
        else:
            try:
                summary_df = pd.read_excel(os.path.join(casePath,
                                                        'Grid Search',
                                                    'Evaluation Metrics.xlsx'))
            except:
                summary_df = pd.read_excel(os.path.join(casePath,
                                                    'Evaluation Metrics.xlsx'))
            summary_df.set_index(labels[way_1], inplace=True)
            if way_1 == 're' or way_1 == 'i':
                fit_b = summary_df['FiT'].loc[int(s * 100)]
                fit_b = round(fit_b, 2)
                p_b = round(summary_df['Price'].loc[int(s * 100)], 2)
                s = int(s * 100)
            else:
                fit_b = summary_df['FiT'].loc[s]
                fit_b = round(fit_b, 2)
                p_b = round(summary_df['Price'].loc[s], 2)
            
            try:
                out = pd.read_excel(os.path.join(casePath,
                                             'Grid Search',
                                             'Output Files',
                                             str(s),
                                             f'Output_{int(fit_b*100)}_{int(p_b*100)}.xlsx'),
                                sheet_name=None)
            except:
                out = pd.read_excel(os.path.join(casePath,
                                             'Output Files',
                                             str(s),
                                             f'Output_{int(fit_b*100)}_{int(p_b*100)}.xlsx'),
                                sheet_name=None)
        
        
        disp_dg = out['DG Dispatch'].set_index('Unnamed: 0')
        disp_pv = out['PV Dispatch'].set_index('Unnamed: 0')
        feed_in = out['Fed-in Capacity'].set_index('Unnamed: 0')
        unmet = out['Unmet Demand'].set_index('Unnamed: 0')
        bat = out['Battery Output'].set_index('Unnamed: 0')
        bat_c = out['Battery Input '].set_index('Unnamed: 0')
        demand = out['Yearly demand'].set_index('Unnamed: 0')
        
        weights_df = out['Summary'].set_index('Unnamed: 0')
        weights = weights_df[0].loc['Day Weights'].split(',')
        weights = [int(w) for w in weights]
        
        year_dg = []
        for y in range(15):
            tot_dg_y = 0
            for d in range(3):
                tot_dg_y += sum(disp_dg.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_dg.append(tot_dg_y)
        tot_dg.append(sum(year_dg) / 1e6)
        
        year_pv = []
        for y in range(15):
            tot_pv_y = 0
            for d in range(3):
                tot_pv_y += sum(disp_pv.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_pv.append(tot_pv_y)
        tot_pv.append(sum(year_pv) / 1e6)
        
        year_fed_in = []
        for y in range(15):
            tot_fed_in_y = 0
            for d in range(3):
                tot_fed_in_y += sum(feed_in.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_fed_in.append(tot_fed_in_y)
        tot_fi.append(sum(year_fed_in) / 1e6)
    
        year_ud = []
        for y in range(15):
            tot_ud_y = 0
            for d in range(3):
                tot_ud_y += sum(unmet.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_ud.append(tot_ud_y)
        tot_ud.append(sum(year_ud) / 1e6) 
        
        year_bat_out = []
        for y in range(15):
            tot_bat_out_y = 0
            for d in range(3):
                tot_bat_out_y += sum(bat.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_bat_out.append(tot_bat_out_y)
        tot_bat.append(sum(year_bat_out) / 1e6)
        
        year_bat_in = []
        for y in range(15):
            tot_bat_in_y = 0
            for d in range(3):
                tot_bat_in_y += sum(bat_c.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_bat_in.append(tot_bat_in_y)
        tot_bat_c.append(sum(year_bat_in) / 1e6)
        
        demand_l = []
        for y in range(15):
            demand_y = 0
            for d in range(3):
                demand_y += sum(demand.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            demand_l.append(demand_y)
        tot_dem.append(sum(demand_l) / 1e6)
     
        
    if way_1 == 'budget':
        s_range_1 = [i / 1e6 for i in s_range_1]
    tot_dg = np.array(tot_dg)
    tot_pv = np.array(tot_pv)
    tot_fi = np.array(tot_fi)
    tot_ud = np.array(tot_ud)
    tot_bat = np.array(tot_bat)
    tot_bat_c = np.array(tot_bat_c)
    tot_dem = np.array(tot_dem)
    
    ax.bar(np.arange(len(s_range_1)),
           tot_dg,
           label = 'Diesel Generator',
           color = "#d14b4b", width = 0.5, zorder=3
           )
    
    ax.bar(np.arange(len(s_range_1)),
           tot_pv,
           label = 'DGC PV',
           color = "#f9e395", width = 0.5,
           bottom = tot_dg, zorder=3)
    
    ax.bar(np.arange(len(s_range_1)),
           tot_fi,
           label = 'Fed-in Capacity',
           color = "#c2deaf",  width = 0.5,
           bottom = tot_dg + tot_pv, zorder=3)
    
    ax.bar(np.arange(len(s_range_1)),
           tot_ud,
           label = 'Unmet Demand',
           color = "#f2b382", width = 0.5,
           bottom = tot_dg + tot_pv + tot_fi, zorder=3)
    
    ax.bar(np.arange(len(s_range_1)),
           tot_bat,
           label = 'Discharge',
           color = "#b1b1b1", width = 0.5,
           bottom = tot_dg + tot_pv + tot_fi + tot_ud, zorder=3)
    
    ax.bar(np.arange(len(s_range_1)),
           tot_bat_c * -1,
           label = 'Charge',
           color = "#828282", width = 0.5, zorder=3)
    
    if show_demand == 1:
        ax.plot(tot_dem * -1,
                label = 'Net demand',
                color = 'black', zorder=4)
    
    
    summary_df = summary_df.sort_index()
    ax2.plot(np.arange(len(s_range_1)),
             np.array(summary_df['Wasted generation (from total)']) * 100,
             '--',
             label = 'Wasted excess',
             color = "#3a737d",
             linewidth = 3,
             zorder = 5)
    

    # Second axis ticks
    y1_min, y1_max = ax.get_ylim()
    y2_max = np.ceil(max(summary_df['Wasted generation (from total)']) * 100)
    # y2_max = y2_max * 100
    ax2.set_ylim(0, y2_max)
    ax2.set_ylim(bottom=0,
                 top=y2_max * (y1_max / y1_max)
                 )
    plt.xticks(np.arange(len(s_range_1)), s_range_1)
    
    if way_1 == 'budget':
        ax.set_xlabel('Budget (USD M)')
    elif way_1 == 'i':
        ax.set_xlabel('Interest Rate')
    elif way_1 == 're':
        ax.set_xlabel('RE target')
    elif way_1 == 'pros':
        ax.set_xlabel('PV-owner %')
    ax.set_ylabel('Energy (GWh)')
    ax2.set_ylabel('Wasted household PV excess generation potential %', 
                   rotation=270, 
                   va='center',
                   labelpad = 20)

    
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    
    ax2.grid(False) #, linestyle='--', alpha=0, zorder=0)
    
    ax.legend(h1 + h2, l1 + l2,
              loc='upper center',
              bbox_to_anchor=(0.5, 1.25),
              ncol=3,
              frameon=False
              )

    
    plt.subplots_adjust(top=.82)
    # plt.tight_layout(pad=1) 
    
    plt.savefig(new_plots_folder)
    plt.close()


def capacity_sensitivity(casePath, way_1, s_range_1, way_2=None, s_2=None):

    
    assert (way_1 == 'budget' 
            or way_1 == 'pros'
            or way_1 == 're'), 'Wrong first sensitivity parameter'
    assert (way_2 == 'budget' 
            or way_2 == 'pros'
            or way_2 == 're'
            or way_2 == None), 'Wrong second sensitivity parameter'
    
    try:
        s_range_1 = s_range_1.tolist()
    except AttributeError:
        pass
    
    sns.set(font_scale=1.5)
    labels = {'budget': 'Budget',
              'pros': 'Prosumer percentage',
              're': 'RE target'}

    # Create new figure
    if way_2 == None:
        new_plots_folder = os.path.join(casePath, 'Capacity.png')
    elif way_2 == 're' or way_2 == 'i':
        assert (s_2 != None), "Missing second sensitivity value"
        new_plots_folder = os.path.join(casePath, 
                                        f'Capacity {str(int(s_2 * 100))}.png')
    elif way_2 == 'budget':
        assert (s_2 != None), "Missing second sensitivity value"
        new_plots_folder = os.path.join(casePath, 
                                        f'Capacity {str(s_2)}.png')
    fig, ax = plt.subplots(figsize=(8,6))
    
    tot_dg_add = []
    tot_pv_add = []
    tot_bat_add = []
    tot_dg_ret = []
    tot_pv_ret = []
    tot_bat_ret = []
    
    for s in s_range_1:        
        # find summary of evaluation metrics
        if way_2 != None:
            summary_df = pd.read_excel(os.path.join(casePath, 
                                                    'Sensitivity',
                                                    str(s),
                                                    'Evaluation Metrics.xlsx'))
            
            summary_df.set_index(labels[way_2], inplace=True)
            
            if way_2 == 're' or way_2 == 'i':
                fit_b = summary_df['FiT'].loc[int(s_2 * 100)]
                fit_b = round(fit_b, 2)
                p_b = round(summary_df['Price'].loc[int(s_2 * 100)], 2)
            else:
                fit_b = summary_df['FiT'].loc[s_2]
                fit_b = round(fit_b, 2)
                p_b = round(summary_df['Price'].loc[s_2], 2)
                
            
            out = pd.read_excel(os.path.join(casePath,
                                         'Sensitivity',
                                         str(s),
                                         'Output Files',
                                         str(int(s_2 * 100)),
                                         f'Output_{int(fit_b*100)}_{int(p_b*100)}.xlsx'),
                            sheet_name=None)
            out_add = out['Added Capacities']
            out_ret = out['Retired Capacities']
            
        else:
            try:
                summary_df = pd.read_excel(os.path.join(casePath,
                                                        'Grid Search',
                                                    'Evaluation Metrics.xlsx'))
            except:
                summary_df = pd.read_excel(os.path.join(casePath,
                                                    'Evaluation Metrics.xlsx'))
            summary_df.set_index(labels[way_1], inplace=True)
            if way_1 == 're' or way_1 == 'i':
                fit_b = summary_df['FiT'].loc[int(s * 100)]
                fit_b = round(fit_b, 2)
                p_b = round(summary_df['Price'].loc[int(s * 100)], 2)
                s = int(s * 100)
            else:
                fit_b = summary_df['FiT'].loc[s]
                fit_b = round(fit_b, 2)
                p_b = round(summary_df['Price'].loc[s], 2)
                
            try:
                out = pd.read_excel(os.path.join(casePath,
                                             'Grid Search',
                                             'Output Files',
                                             str(s),
                                             f'Output_{int(fit_b*100)}_{int(p_b*100)}.xlsx'),
                                sheet_name=None)
            except:
                out = pd.read_excel(os.path.join(casePath,
                                             'Output Files',
                                             str(s),
                                             f'Output_{int(fit_b*100)}_{int(p_b*100)}.xlsx'),
                                sheet_name=None)
            out_add = out['Added Capacities']
            out_ret = out['Retired Capacities']
                
        out_add.set_index('Unnamed: 0', inplace=True)
        out_ret.set_index('Unnamed: 0', inplace=True)
        cap_dg_add = out_add.loc['Diesel Generator'].sum()
        cap_pv_add = out_add.loc['Owned PV'].sum()
        cap_bat_add = out_add.loc['Owned Batteries'].sum()
        cap_dg_ret = - out_ret.loc['Diesel Generator'].sum()
        cap_pv_ret = - out_ret.loc['Owned PV'].sum()
        cap_bat_ret = - out_ret.loc['Owned Batteries'].sum()
        
        tot_dg_add.append(cap_dg_add)
        tot_pv_add.append(cap_pv_add)
        tot_bat_add.append(cap_bat_add)
        tot_dg_ret.append(cap_dg_ret)
        tot_pv_ret.append(cap_pv_ret)
        tot_bat_ret.append(cap_bat_ret)
    
    if way_1 == 'budget':
        s_range_1 = [i / 1e6 for i in s_range_1]
        
    tot_dg = np.array(tot_dg_add)
    tot_pv = np.array(tot_pv_add)
    tot_bat = np.array(tot_bat_add)
    tot_dg_ret = np.array(tot_dg_ret)
    tot_pv_ret = np.array(tot_pv_ret)
    tot_bat_ret = np.array(tot_bat_ret)
    
    ax.bar(np.arange(len(s_range_1)),
           tot_dg,
           label = 'Diesel Generator',
           color = "#d14b4b", width = 0.5
           )
    
    ax.bar(np.arange(len(s_range_1)),
           tot_pv,
           label = 'DGC PV',
           color = "#f9e395", width = 0.5,
           bottom = tot_dg)
    

    ax.bar(np.arange(len(s_range_1)),
           tot_bat,
           label = 'DGC Batteries',
           color = "#b1b1b1", width = 0.5,
           bottom = tot_dg + tot_pv)
    
    ax.bar(np.arange(len(s_range_1)),
           tot_dg_ret,
           color = "#d14b4b", width = 0.5, hatch='//'
           )
    
    ax.bar(np.arange(len(s_range_1)),
           tot_pv_ret,
           color = "#f9e395", width = 0.5,
           bottom = tot_dg_ret, hatch='//')
    
    ax.bar(np.arange(len(s_range_1)),
           tot_bat_ret,
           color = "#b1b1b1", width = 0.5,
           bottom = tot_dg_ret + tot_pv_ret, hatch='//')
    
    hatch_proxy = mpatches.Patch(
        facecolor='none', edgecolor='grey', hatch='//',
        label='Retired capacity')
    
    plt.xticks(np.arange(len(s_range_1)), s_range_1)
    
    if way_1 == 'budget':
        ax.set_xlabel('Budget (USD M)')
    elif way_1 == 'pros':
        ax.set_xlabel('PV-owner %')
    elif way_1 == 're':
        ax.set_xlabel('RE target')
    
    ax.set_ylabel('Total Added Capacity (kW)')
    
    handles, legend_labels = ax.get_legend_handles_labels()
    handles = list(handles)
    legend_labels = list(legend_labels)
    handles.append(hatch_proxy)
    legend_labels.append('Retired capacity')
    ax.legend(handles, 
              legend_labels,
              loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2,
              bbox_transform=fig.transFigure,
              frameon=False)

    
    plt.subplots_adjust(top=.85)
    plt.tight_layout(pad=1) 
    
    plt.savefig(new_plots_folder)
    plt.close()
    
def re_sensitivity(casePath, re_levels):
    
    sns.set(font_scale=1.15)
    
    new_plots_folder_cap = os.path.join(casePath, 
                                        'Capacity.png')
    new_plots_folder_ene = os.path.join(casePath, 
                                        'Energy.png')
    fig_c, ax_c = plt.subplots()
    fig_e, ax_e = plt.subplots()
    
    tot_dg = []
    tot_d = []
    tot_pv = []
    tot_bat = []
    tot_fi = []
    tot_ud = []
    
    tot_dg_cap = []
    tot_pv_cap = []
    tot_bat_cap = []
    
    summary_df = pd.read_excel(os.path.join(casePath, 
                                            'Grid Search',
                                            'Evaluation Metrics.xlsx'))
    
    summary_df = summary_df.sort_values(by='RE target')
    for index, row in summary_df.iterrows():
        re_level = row['RE target']
        re_level_s = str(int(re_level))
        fit = row['FiT']
        fit_s = str(int(fit * 100))
        p = row['Price']
        p_s = str(int(p * 100))
        
        out = pd.read_excel(os.path.join(casePath, 
                                         'Grid Search',
                                         'Output Files',
                                         re_level_s,
                                         f'Output_{fit_s}_{p_s}.xlsx'),
                            sheet_name=None)
        
        demand = out['Yearly demand'].set_index('Unnamed: 0')
        disp_dg = out['DG Dispatch'].set_index('Unnamed: 0')
        disp_pv = out['PV Dispatch'].set_index('Unnamed: 0')
        feed_in = out['Fed-in Capacity'].set_index('Unnamed: 0')
        unmet = out['Unmet Demand'].set_index('Unnamed: 0')
        bat = out['Battery Output'].set_index('Unnamed: 0')
        
        weights_df = out['Summary'].set_index('Unnamed: 0')
        weights = weights_df[0].loc['Day Weights'].split(',')
        weights = [int(w) for w in weights]
        
        cap = out['Installed Capacities']
        cap.set_index('Unnamed: 0', inplace=True)
        
        cap_dg = cap.loc['Diesel Generator'].mean()
        cap_pv = cap.loc['Owned PV'].mean()
        cap_bat = cap.loc['Owned Batteries'].mean()
        
        tot_dg_cap.append(cap_dg)
        tot_pv_cap.append(cap_pv)
        tot_bat_cap.append(cap_bat)
        
        year_d = []
        for y in range(15):
            tot_d_y = 0
            for d in range(3):
                tot_d_y += sum(demand.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_d.append(tot_d_y)
        tot_d.append(sum(year_d) / 1e6)
        
        year_dg = []
        for y in range(15):
            tot_dg_y = 0
            for d in range(3):
                tot_dg_y += sum(disp_dg.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_dg.append(tot_dg_y)
        tot_dg.append(sum(year_dg) / 1e6)
        
        year_pv = []
        for y in range(15):
            tot_pv_y = 0
            for d in range(3):
                tot_pv_y += sum(disp_pv.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_pv.append(tot_pv_y)
        tot_pv.append(sum(year_pv) / 1e6)
        
        year_fed_in = []
        for y in range(15):
            tot_fed_in_y = 0
            for d in range(3):
                tot_fed_in_y += sum(feed_in.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_fed_in.append(tot_fed_in_y)
        tot_fi.append(sum(year_fed_in) / 1e6)
    
        year_ud = []
        for y in range(15):
            tot_ud_y = 0
            for d in range(3):
                tot_ud_y += sum(unmet.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_ud.append(tot_ud_y)
        tot_ud.append(sum(year_ud) / 1e6) 
        
        year_bat_out = []
        for y in range(15):
            tot_bat_out_y = 0
            for d in range(3):
                tot_bat_out_y += sum(bat.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
            year_bat_out.append(tot_bat_out_y)
        tot_bat.append(sum(year_bat_out) / 1e6)
     
    tot_dg_cap = np.array(tot_dg_cap)
    tot_pv_cap = np.array(tot_pv_cap)
    tot_bat_cap = np.array(tot_bat_cap)
    
    ax_c.bar(re_levels,
            tot_dg_cap,
            label = 'Diesel Generator',
            color = "#d14b4b", width = 0.02
            )
    
    ax_c.bar(re_levels,
           tot_pv_cap,
           label = 'DGC PV',
           color = "#f9e395", width = 0.02,
           bottom = tot_dg_cap)
    

    ax_c.bar(re_levels,
           tot_bat_cap,
           label = 'DGC Batteries',
           color = "#b1b1b1", width = 0.02,
           bottom = tot_dg_cap + tot_pv_cap)
    

    ax_c.set_xlabel('RE target')
    ax_c.set_ylabel('Average Installed Capacity (kW)')
    ax_c.legend(loc='upper center', bbox_to_anchor=(0.5, .97), ncol=3,
              bbox_transform=fig_c.transFigure,
              frameon=False)

    fig_c.subplots_adjust(top=.85)
    fig_c.savefig(new_plots_folder_cap)
    plt.close(fig_c)

    tot_dg = np.array(tot_dg)
    tot_pv = np.array(tot_pv)
    tot_fi = np.array(tot_fi)
    tot_ud = np.array(tot_ud)
    tot_bat = np.array(tot_bat)
    tot_d = np.array(tot_d)
    
    ax_e.bar(re_levels,
           tot_dg,
           label = 'Diesel Generator',
           color = "#d14b4b", width = 0.04
           )
    
    ax_e.bar(re_levels,
           tot_pv,
           label = 'DGC PV',
           color = "#f9e395", width = 0.04,
           bottom = tot_dg)
    
    ax_e.bar(re_levels,
           tot_fi,
           label = 'Fed-in Capacity',
           color = "#c2deaf",  width = 0.04,
           bottom = tot_dg + tot_pv)
    
    ax_e.bar(re_levels,
           tot_ud,
           label = 'Unmet Demand',
           color = "#f2b382", width = 0.04,
           bottom = tot_dg + tot_pv + tot_fi)
    
    ax_e.bar(re_levels,
           tot_bat,
           label = 'DGC Batteries',
           color = "#b1b1b1", width = 0.04,
           bottom = tot_dg + tot_pv + tot_fi + tot_ud)
    
    ax_e.plot(re_levels, tot_d * -1, color="#595755", label='Total Demand')
    
    ax_e.set_xlabel('RE target')
    ax_e.set_ylabel('Energy (GWh)')
    ax_e.legend(loc='upper center', bbox_to_anchor=(0.5, .97), ncol=3,
              bbox_transform=fig_e.transFigure,
              frameon=False)

    
    fig_e.subplots_adjust(top=.85)
    fig_e.savefig(new_plots_folder_ene)
    plt.close(fig_e)
    
def min_v_act_RE(casePaths, labels=None):
    
    sns.set(font_scale=1.2)
    new_plots_folder = os.path.join(casePaths[-1], 
                                        'RE target.png')
    
    colors = ["#2f847e", "#40dbe6"]
    fig, ax = plt.subplots(figsize=(8,5))
    
    for j in range(len(casePaths)):
        
        act_re = []
        tot_pv = []
        tot_bat_out = []
        tot_bat_in = []
        tot_fi = []
        tot_d = []
        tot_dg = []
        tot_ud = []
        
        re_levels = []
        
        try:
            eval_sum = pd.read_excel(os.path.join(casePaths[j], 
                                                  'Grid Search',
                                                  'Evaluation Metrics.xlsx'))
        except:
            eval_sum = pd.read_excel(os.path.join(casePaths[j],
                                                  'Evaluation Metrics.xlsx'))
        eval_sum = eval_sum.sort_values(by='RE target')
        for index, row in eval_sum.iterrows():
            re_level = row['RE target']
            re_levels.append(re_level)
            re_level_s = str(int(re_level))
            fit = row['FiT']
            fit_s = str(int(fit * 100))
            p = row['Price']
            p_s = str(int(p * 100))
            try:
                out = pd.read_excel(os.path.join(casePaths[j], 
                                                 'Grid Search',
                                                 'Output Files',
                                                 re_level_s,
                                                 f'Output_{fit_s}_{p_s}.xlsx'),
                                    sheet_name=None)
            except:
                out = pd.read_excel(os.path.join(casePaths[j], 
                                                 'Output Files',
                                                 re_level_s,
                                                 f'Output_{fit_s}_{p_s}.xlsx'),
                                    sheet_name=None)
    
            demand = out['Yearly demand'].set_index('Unnamed: 0')
            disp_pv = out['PV Dispatch'].set_index('Unnamed: 0')
            feed_in = out['Fed-in Capacity'].set_index('Unnamed: 0')
            bat_out = out['Battery Output'].set_index('Unnamed: 0')
            bat_in = out['Battery Input '].set_index('Unnamed: 0')
            ud = out['Unmet Demand'].set_index('Unnamed: 0')
            disp_dg = out['DG Dispatch'].set_index('Unnamed: 0')
            
            weights_df = out['Summary'].set_index('Unnamed: 0')
            weights = weights_df[0].loc['Day Weights'].split(',')
            weights = [int(w) for w in weights]
    
            year_d = []
            for y in range(15):
                tot_d_y = 0
                for d in range(3):
                    tot_d_y += sum(demand.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
                year_d.append(tot_d_y)
            tot_d.append(sum(year_d))
            
            year_dg = []
            for y in range(15):
                tot_dg_y = 0
                for d in range(3):
                    tot_dg_y += sum(disp_dg.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
                year_dg.append(tot_dg_y)
            tot_dg.append(sum(year_dg))
            
            year_ud = []
            for y in range(15):
                tot_ud_y = 0
                for d in range(3):
                    tot_ud_y += sum(ud.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
                year_ud.append(tot_ud_y)
            tot_ud.append(sum(year_ud))
            
            year_pv = []
            for y in range(15):
                tot_pv_y = 0
                for d in range(3):
                    tot_pv_y += sum(disp_pv.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
                year_pv.append(tot_pv_y)
            tot_pv.append(sum(year_pv))
            
            year_fed_in = []
            for y in range(15):
                tot_fed_in_y = 0
                for d in range(3):
                    tot_fed_in_y += sum(feed_in.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
                year_fed_in.append(tot_fed_in_y)
            tot_fi.append(sum(year_fed_in))
            
            year_bat_out = []
            for y in range(15):
                tot_bat_out_y = 0
                for d in range(3):
                    tot_bat_out_y += sum(bat_out.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
                year_bat_out.append(tot_bat_out_y)
            tot_bat_out.append(sum(year_bat_out))
            
            year_bat_in = []
            for y in range(15):
                tot_bat_in_y = 0
                for d in range(3):
                    tot_bat_in_y += sum(bat_in.loc[float(f'{y}'+'.'+f'{d}')]) * weights[d]
                year_bat_in.append(tot_bat_in_y)
            tot_bat_in.append(sum(year_bat_in))
            
            act_re.append((sum(year_fed_in) + sum(year_pv)) / 
                          (- sum(year_d) + sum(bat_in) - sum(bat_out)))
            
        ax.plot(re_levels,
        np.array(act_re) * 100,
        color=colors[j],
        label = (labels[j] if labels != None else None),
        linewidth=3)

    ax.set_xlabel('RE target (%)')
    ax.set_ylabel('Effective RE penetration (%)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, .97), ncol=2,
              bbox_transform=fig.transFigure,
              frameon=False)

    fig.subplots_adjust(top=.85)
    fig.savefig(new_plots_folder)
    plt.close(fig)
    
def hs_comp(casePath):
    
    sns.set(font_scale=1.3)
    new_plots_folder = os.path.join(casePath,'HS comp.png')
    new_plots_folder_p = os.path.join(casePath, 'HS comp perc.png')
    
    fig, ax = plt.subplots()
    fig_p, ax_p = plt.subplots()
    
    base_hss = []
    base_files = os.listdir(os.path.join(casePath, "Base Cases"))
    bases = [int(i) for i in base_files]
    bases.sort()
    base_files = [str(i) for i in bases]
    
    for file in base_files:
        base_df = pd.read_excel(os.path.join(casePath,
                                             "Base Cases",
                                             file,
                                             "Output_0_40.xlsx"),
                                sheet_name="Summary")
        base_df.set_index("Unnamed: 0", inplace=True)
        base_hs = base_df.loc["Household Surplus"][0]
        base_hss.append(base_hs)
        
    
    new_hss_df = pd.read_excel(os.path.join(casePath,
                                            "Grid Search",
                                            "Evaluation Metrics.xlsx"))
    new_hss_df = new_hss_df.sort_values(by='Prosumer percentage')
    pros_percs = new_hss_df['Prosumer percentage']
    new_hss = new_hss_df['Household Surplus']
    
    new_hss = np.array(new_hss)
    base_hss = np.array(base_hss)
    
    perc_change = new_hss - base_hss
    perc_change_p = (new_hss - base_hss) / base_hss
    
    ax.plot(np.array(pros_percs),
            perc_change / 1e6,
            color= "#64b985",
            linewidth=3,
            linestyle="--")
    
    ax.set_xlabel('PV-owners (%)')
    ax.set_ylabel('Change in Household\n Economic Surplus (M USD) ')
    
    ax_p.plot(np.array(pros_percs),
            perc_change_p / 1e6,
            color= "#64b985",
            linewidth=3,
            linestyle="--")
    
    ax_p.set_xlabel('PV-owners (%)')
    ax_p.set_ylabel('Change in Household\n Economic Surplus (%) ')
    
    fig.tight_layout()
    fig.subplots_adjust(top=.85)
    fig.savefig(new_plots_folder)
    plt.close(fig)
    
    fig_p.tight_layout()
    fig_p.subplots_adjust(top=.85)
    fig_p.savefig(new_plots_folder_p)
    plt.close(fig_p)

def hs_constrained(casePath1, casePath2):
    sns.set(font_scale=1.1)
    
    new_plots_folder = os.path.join(casePath2, 'HES loss.png')
    fig, ax = plt.subplots()
    
    hes_uncs = {}
    hes_cs = {}
    loss = {}
    
    emPath1 = os.path.join(casePath1, 'Grid Search', "Evaluation Metrics.xlsx")
    emPath2 = os.path.join(casePath2, 'Grid Search', 'Evaluation Metrics.xlsx')
    
    emFile1 = pd.read_excel(emPath1).set_index('Prosumer percentage')
    emFile2 = pd.read_excel(emPath2).set_index('Prosumer percentage')
    
    for _, row in emFile1.iterrows():
        hes_uncs[_] = row['Household Surplus']
    for _, row in emFile2.iterrows():
        hes_cs[_] = row['Household Surplus']
        loss[_] = hes_cs[_] - hes_uncs[_]
    
    loss_df = pd.Series(loss)
    
    ax.plot(loss_df)
    
    fig.tight_layout()
    fig.subplots_adjust(top=.85)
    fig.savefig(new_plots_folder)
    plt.close(fig)
    
def inv_comp(casePath, outPaths):
    
    sns.set(font_scale=1.1)
    
    new_plots_folder = os.path.join(casePath, 'Investment comp.png')
    fig, ax = plt.subplots()
    
    pv_invs = {}
    
    for outPath in outPaths:
        re_level = os.path.basename(os.path.dirname(outPath))
        df = pd.read_excel(outPath, sheet_name='Added Capacities')
        df.set_index('Unnamed: 0', inplace=True)
        pv_inv = df.loc['Owned PV'].to_list()[0:4]
        pv_invs[f'{re_level}%'] = pv_inv
        df.reset_index()
    
    x_labels = [0, 1, 2, 3]                
    n_groups = len(x_labels)
    n_bars = len(pv_invs)
    bar_width = 0.8 / n_bars                
    
    x = np.arange(n_groups)            
    colors = ["#FFE797", "#FCB53B", "#B45253"]
    for i, (label, values) in enumerate(pv_invs.items()):
        for j, val in enumerate(values):
            xpos = x[j] + i * bar_width
    
            if val == 0:
                ax.scatter(
                    xpos, 
                    0.03 * max(map(max, pv_invs.values())),
                    marker='x',
                    color=colors[i],
                    s=80,
                    linewidths=2,
                    zorder=3
                )
            else:
                ax.bar(
                    xpos, val,
                    width=bar_width,
                    label=label, # if j == 0 else "",
                    color=colors[i]
                )
                
    for sep in x[:-1] + 0.75:
        ax.axvline(
            sep,
            linestyle='--',
            linewidth=1,
            color='gray',
            alpha=0.5,
            zorder=0
        )
    
    ax.set_xticks(x + bar_width * (n_bars - 1) / 2)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Added owned PV (kW)')
    ax.set_xlabel('Year')
    ax.legend(title='Renewable level', loc='upper center', 
              bbox_to_anchor=(0.5, 1), ncol=3,
              bbox_transform=fig.transFigure,
              frameon=False)
    plt.tight_layout()
    
    plt.savefig(new_plots_folder, dpi=300)
    plt.close()
    
    
def plot_lcoe(inFile, FiT = None):
    
    data = pd.read_excel(inFile, sheet_name=None)
    new_plots_folder = os.path.join(inFile, "..",
                                    'LCOEs.png')
    sns.set(font_scale=1.2)
    
    def compute_lcoe(capex,
                     fixed_opex,
                     variable_opex,
                     fuel_cost,
                     capacity_factor,
                     lifetime,
                     interest_rate):
        r = interest_rate
        N = lifetime
        crf = r * (1 + r)**N / ((1 + r)**N - 1)
        annualized_capex = capex * crf
        total_annual_cost = annualized_capex + fixed_opex
        lcoe = total_annual_cost / (8760 * capacity_factor) + variable_opex + fuel_cost
    
        return lcoe
    
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#d14b4b", "#f4d35e", "#779662"]
    
    interest = 0.1
    capex_all = {}
    vopex_all = {}
    fopex_all = {}
    cap_fact = {}
    lifetime_all = {}
    fuel_costs = {}
    
    tech_df = data['tech'].set_index('Unnamed: 0')
    for _, row in tech_df.iterrows():
        vopex_all[_] = row['UOVC']
        fopex_all[_] = row['UOFC']
        lifetime_all[_] = row['Lifetime']
    
    heat_rate = data['heat_rate'].loc[1, 'HR']
    diesel_costs = vopex_all['Diesel'] * heat_rate
    fuel_costs['Owned PV'] = 0
    fuel_costs['Owned Batteries'] = 0
    fuel_costs['Diesel Generator'] = diesel_costs
    capex_df = data['capex'].set_index('Unnamed: 0')
    
    for col in capex_df.columns:
        capex_all[col] = capex_df[col]
        
    cap_fact['Diesel Generator'] = 0.8
    cap_fact['Owned Batteries'] = 0.96
    cap_fact['Owned PV'] = 0.174
    
    lcoe_all = defaultdict(list)
    for j, col in enumerate(capex_df.columns):
        for i in range(15):
            lcoe_all[col].append(compute_lcoe(capex_all[col][i],
                                              fopex_all[col],
                                              vopex_all[col],
                                              fuel_costs[col],
                                              cap_fact[col],
                                              lifetime_all[col],
                                              interest
                                              )
                                 )
    
        ax.plot(lcoe_all[col], '-', label = f'{col} LCOE', color = colors[j])
        ax.plot([vopex_all[col] + fuel_costs[col]] * 15, '--', 
                label = f'{col} MC', color = colors[j])
    
    if FiT != None:
        ax.plot([FiT] * 15, '--', label = 'FiT', color="#85a4c4")
    ax.set_ylabel('USD/kWh')
    ax.set_xlabel('Year of installation')
    ax.legend(loc='upper center', 
              bbox_to_anchor=(0.5, 1), ncol=3,
              bbox_transform=fig.transFigure,
              frameon=False)
    plt.tight_layout()
    
    plt.savefig(new_plots_folder)
    plt.close()

def hs_comp_cons(casePath1, casePath2, index='pros', addCurrent=None):
    
    sns.set(font_scale=1.3)
    new_plots_folder_hs = os.path.join(casePath2, 'HS+constrained comparison.png')
    
    # Get summary from evaluation metrics files
    eval_sum = pd.read_excel(os.path.join(casePath1, 
                                          'Grid Search',
                                          'Evaluation Metrics.xlsx'))
    eval_sum_cons = pd.read_excel(os.path.join(casePath2,
                                               'Grid Search',
                                               "Evaluation Metrics.xlsx"))
    
    indices = {'re': 'RE target',
               'budget': 'Budget',
               'pros': 'Prosumer percentage'}
    eval_sum = eval_sum.sort_values(by=indices[index])
    eval_sum_cons = eval_sum_cons.sort_values(by=indices[index])
    
    # Plot
    x = np.array(eval_sum[indices[index]])
    y_hs = np.array(eval_sum['Household Surplus']) / 1e6
    y_hs_cons = np.array(eval_sum_cons['Household Surplus']) / 1e6
    
    fig_hs, ax_hs = plt.subplots(figsize=(7,5))
    
    label_hs = 'Unconstrained household economic surplus'
    label_hs_cons = 'Household economic surplus under unmet demand constraint'
    ncol = 1
    
    if addCurrent != None:
        if type(addCurrent) == str:
            curr_df = pd.read_excel(addCurrent, sheet_name='Summary')
            curr_df.set_index("Unnamed: 0", inplace=True)
            curr_hs = curr_df.loc["Household Surplus"][0]
            
            ax_hs.plot(x,
                       [curr_hs / 1e6] * len(x),
                       linewidth = 2,
                       linestyle= '--',
                       color = "#595755",
                       label='Status-quo household economic surplus')
        
        elif type(addCurrent) == list:
            curr_hss = []
            for curr in addCurrent:
                curr_df = pd.read_excel(curr, sheet_name='Summary')
                curr_df.set_index("Unnamed: 0", inplace=True)
                curr_hs = curr_df.loc["Household Surplus"][0]
                curr_hss.append(curr_hs)
            
            curr_hss = np.array(curr_hss)
            ax_hs.plot(x,
                       curr_hss /1e6,
                       linewidth = 2,
                       linestyle= '--',
                       color = "#595755",
                       label='Status-quo household economic surplus')
        
    ax_hs.plot(x,
               y_hs,
               linewidth = 3,
               color = '#64b985',
               label=label_hs)
    ax_hs.plot(x,
               y_hs_cons,
               linewidth = 3,
               color = '#36656B',
               label=label_hs_cons)
    
    if index == 'budget':
        ax_hs.set_xlabel(f'{indices[index]} (M USD)')
    elif index == 're':
        ax_hs.set_xlabel(f'{indices[index]} (%)')
    else:
        ax_hs.set_xlabel('PV-owners (%)')
    ax_hs.set_ylabel('Household economic surplus (M USD)')
    ax_hs.legend(loc='upper center',
                 bbox_to_anchor=(0.5, 1.25),
                 ncol=ncol,
                 frameon=False)
    
    fig_hs.subplots_adjust(top=.85)
    fig_hs.savefig(new_plots_folder_hs)
    plt.close(fig_hs)
    
    
#------------------------------------------------------------------------------#    
# Run the functions for the different cases                                    #
#------------------------------------------------------------------------------#

cwd = os.getcwd()
outFile = os.path.join(cwd, "Outputs")
outFile_0 = os.path.join(outFile, '0. Current Case', 'Output_0_40.xlsx')
inFile = os.path.join(cwd, "Inputs", 'inputs_RE.xlsx')

# plot_lcoe(inFile, FiT=0.11)
'''
# Current Case
add_ret(outFile_0, multi=0)
gen_year(outFile_0, multi=0)
rep_day(outFile_0, multi=0, year=10, day=1)
inst_cap(outFile_0, multi=0)
'''
# Budget~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
outFile_1 = os.path.join(outFile, '5. Budget constrained')

# Economic Analysis
keys = [250000, 400000, 750000, 1750000]
colors = ["#595755", "#6d597a", "#DA4167" ,
                  "#f2b382", "#f4d35e", "#85a4c4", 
                  "#c2deaf" ]

outFile_1_1 = os.path.join(outFile_1, 'Grid Search')
summary_path_1 = os.path.join(outFile_1, 'Summary.xlsx')
'''
em_path_1 = os.path.join(outFile_1, 'Grid Search', 'Evaluation Metrics.xlsx')
files = ["Output_12_36.xlsx", "Output_13_36.xlsx", 
         "Output_14_36.xlsx", "Output_12_35.xlsx"]

for file in files:
    file_path = os.path.join(outFile_1, "Grid Search", "Output Files",
                             "400000", file)
    gen_year(file_path)
    rep_day(file_path, 5, 0)
    rep_day(file_path, 5, 1)
    rep_day(file_path, 5, 2)
    inst_cap(file_path)

fit_v_price(outFile_1, search='budget', keys=keys, 
            colors = ["#6d597a", "#DA4167" , "#f2b382", "#c2deaf"])

# keys = [420000]
# fit_v_price(outFile_1, search='budget', keys=keys, 
#             colors = ["#595755"])

# keys = [420000]

keys = [100000, 250000, 400000, 750000, 1000000, 1250000, 1500000, 
        1750000, 2000000]

keys = [1750000]
for budget in keys:
    surp_heatmap(outFile_1_1, index='budget', key=budget, 
                  max_fits=summary_path_1, p_lb=0.26, fit_ub=0.22)
    # fi_gradient(outFile_1, budget)

# keys = [400000]
# surp_heatmap(outFile_1_1, index='budget', key=budget, 
#               max_fits=summary_path_1)

re_comp(outFile_1, index='budget', addCurrent=outFile_0)

# Technical Analysis
capacity_sensitivity(outFile_1, 'budget', s_range_1=keys)
energy_sensitivity(outFile_1, 'budget', s_range_1=keys)

# Daily generation 
#   Find summary
emFile_1 = pd.read_excel(os.path.join(outFile_1,
                                      'Grid Search (extra)',
                                      'Evaluation Metrics.xlsx'))
emFile_1.set_index('Budget', inplace=True)
for _, row in emFile_1.iterrows():
    budget = _
    price = int(row['Price'] * 100)
    fit = int(row['FiT'] * 100)
    outPath_day = os.path.join(outFile_1,
                                'Grid Search (extra)',
                                'Output Files',
                                str(_),
                                f'Output_{fit}_{price}.xlsx')
    # rep_day(outPath_day, 10, 1, 1)
    # add_ret(outPath_day, multi=1)
    gen_year(outPath_day)
    inst_cap(outPath_day)

# RE sensitivity ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
outFile_2 = os.path.join(outFile, "2. RE sensitivity")

fit_v_price(outFile_2)

current_budget = 400000
re_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

summaryPath_2_1 = os.path.join(outFile_2, 'Summary.xlsx')

outFile_2_1 = os.path.join(outFile_2, 'Grid Search')

for re_level in re_levels:
    surp_heatmap(outFile_2_1, re_level, max_fits=summaryPath_2_1, lb = 0.25)
   
# energy_sensitivity(outFile_2_1, 're', re_levels)
# capacity_sensitivity(outFile_2_1, 're', re_levels)
re_comp(outFile_2_1, index='re', addCurrent=outFile_0)
# min_v_act_RE([outFile_2_1])

endOutPaths = [os.path.join(outFile_2_1, "Output Files", re) 
               for re in ['40', '50', '60']]
outNames = ["Output_11_35.xlsx", "Output_10_35.xlsx", "Output_18_40.xlsx"]
outPaths = []
for i in range(len(outNames)):
    outPaths.append(os.path.join(endOutPaths[i], outNames[i]))
inv_comp(outFile_2, outPaths)
    

for budget in keys:
  outFile_2_b = os.path.join(outFile_2, 'Feasible Region', str(budget))
  fit_v_price(outFile_2_b, search='re')

summary_df = pd.read_excel(os.path.join(outFile_2, 
                                        'Sensitivity',
                                        '400000',
                                        'Evaluation Metrics.xlsx'))

summary_df = summary_df.sort_values(by='RE target')
for index, row in summary_df.iterrows():
    re_level = row['RE target']
    re_level_s = str(int(re_level))
    fit = row['FiT']
    fit_s = str(int(fit * 100))
    p = row['Price']
    p_s = str(int(p * 100))
    
    outName = f'Output_{fit_s}_{p_s}.xlsx'
    outPath = os.path.join(outFile_2, 'Sensitivity', '400000', 
                           'Output Files', re_level_s, outName)
    add_ret(outPath, 1)
    # inst_cap(outPath, 1)
    # gen_year(outPath, 1)

# Prosumers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
outFile_5 = os.path.join(outFile, '3. Prosumer percentage')
outFile_6 = os.path.join(outFile, '4. Prosumer percentage constrained')

hs_comp(outFile_5)

base_casePaths = []
percs_str = os.listdir(os.path.join(outFile_5, "Base Cases"))
percs = [int(i) for i in percs_str]
percs.sort()
percs_str = [str(i) for i in percs]

for perc in percs_str:
    base_p = os.path.join(outFile_5,
                          "Base Cases",
                          perc,
                          "Output_0_40.xlsx")
    base_casePaths.append(base_p)
    inst_cap(base_p, 1)
    gen_year(base_p, 1)
    rep_day(base_p, 5, 0, 1)
    rep_day(base_p, 5, 1, 1)
    rep_day(base_p, 5, 2, 1)

re_comp(outFile_5, index='pros', addCurrent = base_casePaths)

#   Find summary
emFile_5 = pd.read_excel(os.path.join(outFile_5,
                                      'Grid Search',
                                      'Evaluation Metrics.xlsx'))
emFile_5.set_index('Prosumer percentage', inplace=True)
for _, row in emFile_5.iterrows():
    budget = _
    price = int(row['Price'] * 100)
    fit = int(row['FiT'] * 100)
    outPath_day = os.path.join(outFile_5,
                                'Grid Search',
                                'Output Files',
                                str(_),
                                f'Output_{fit}_{price}.xlsx')
    # rep_day(outPath_day, 10, 1, 1)
    # add_ret(outPath_day, multi=1)
    gen_year(outPath_day)
    inst_cap(outPath_day)

# Technical Analysis
keys = [0, 10, 25, 40, 50, 60, 75, 90, 100]
capacity_sensitivity(outFile_5, 'pros', s_range_1=keys)
energy_sensitivity(outFile_5, 'pros', s_range_1=keys, show_demand=1)

#   Find summary
emFile_5 = pd.read_excel(os.path.join(outFile_5,
                                      'Grid Search',
                                      'Evaluation Metrics.xlsx'))
emFile_5.set_index('Prosumer percentage', inplace=True)
for _, row in emFile_5.iterrows():
    budget = _
    price = int(row['Price'] * 100)
    fit = int(row['FiT'] * 100)
    outPath_day = os.path.join(outFile_5,
                                'Grid Search',
                                'Output Files',
                                str(_),
                                f'Output_{fit}_{price}.xlsx')
    # rep_day(outPath_day, 10, 1, 1)
    # add_ret(outPath_day, multi=1)
    gen_year(outPath_day)
    inst_cap(outPath_day)

keys = [0, 10, 25, 40, 50, 60, 75, 90, 100]
hs_constrained(outFile_5, outFile_6)
capacity_sensitivity(outFile_6, 'pros', s_range_1=keys)
energy_sensitivity(outFile_6, 'pros', s_range_1=keys, show_demand=1)
# re_comp(outFile_6, index='pros', addCurrent = base_casePaths)
hs_comp_cons(outFile_5, outFile_6, index='pros', addCurrent=base_casePaths)
''';