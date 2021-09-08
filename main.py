import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns


def get_data():
    df = pd.read_excel('data.xlsx', sheet_name='Step 2')
    to_rename = {'Customer order cases':'COC', 'Forecast for 10th to 12th Jan':'forecast',
                 'Current Inventory level':'CIL', 'Category':'category'}
    df.rename(columns=to_rename, inplace=True)
    data = df[['SKU', 'category', 'COC', 'forecast', 'CIL', 'EPQ', 'SS']]
    return data


def get_msa(data):
    #get the earliest pickup date

    def get_epd(data):
        df = pd.read_excel('data.xlsx', sheet_name='Step 4')[['SKU', 'Earliest Pickup Date']]
        #parse SKUs and retrive information about bottle type, water, case and client
        skus = pd.read_excel('data.xlsx', sheet_name='Step 4')['SKU']
        client = []
        bottle = []
        water = []
        pack = []
        for i in range(len(skus)):
            name = str(skus.iloc[i])
            client.append(name[0:3])
            bottle.append(name[3:6])
            pack.append(name[6:9])
            water.append(name[9:11])

        df['client'] = client
        df['bottle'] = bottle
        df['water'] = water
        df['pack'] = pack
        #print(df.head(25))

        if df.duplicated().sum() > 0:
            warnings.warn('Duplicated jobs have been discovered and dropped during the retrival of the earliest pickup dates')
            df.drop_duplicates(inplace=True)
        df = df.set_index('SKU')
        EPD = []
        client = []
        bottle = []
        water = []
        pack = []
        for i in range(len(data['SKU'])):
            Id = data['SKU'].iloc[i]
            EPD.append(df.loc[Id]['Earliest Pickup Date'])
            client.append(df.loc[Id]['client'])
            bottle.append(df.loc[Id]['bottle'])
            water.append(df.loc[Id]['water'])
            pack.append(df.loc[Id]['pack'])
        data['EPD'] = EPD
        data['client'] = client
        data['bottle'] = bottle
        data['water'] = water
        data['pack'] = pack
        return data

    data = get_epd(data)

    def get_info(data):
        df = pd.read_excel('data.xlsx', sheet_name='Step 4')['SKU']
        client = []
        bottle = []
        water = []
        pack = []
        for i in range(len(df)):
            name = str(df.iloc[i])
            client.append(name[0:3])
            bottle.append(name[3:6])
            pack.append(name[6:9])
            water.append(name[9:11])
        return data

    # Total Capacity Bottles per day (-3)	Total Capacity Cases per day (-2)	Total Capacity Pallets per day (-1)
    df = pd.read_excel('MSA.xlsx', index_col='ITEM')
    dummy = [pd.NA for i in range(len(data))]
    data['TCB L1'] = dummy
    data['TCB L2'] = dummy
    data['TCB L3'] = dummy

    data['TCC L1'] = dummy
    data['TCC L2'] = dummy
    data['TCC L3'] = dummy

    data['TCP L1'] = dummy
    data['TCP L2'] = dummy
    data['TCP L3'] = dummy

    SKUs = data['SKU']
    for i in range(len(data)):
        sku = SKUs.iloc[i]
        if len(df.loc[sku].values) == 2:
            warnings.warn('An SKU can be produced on two production lines')
        if len(df.loc[sku].values) != 3:
            if df.loc[sku]['Line']=='L3':
                data['TCB L3'].iloc[i] = df.loc[sku].values[-3]
                data['TCC L3'].iloc[i] = df.loc[sku].values[-2]
                data['TCP L3'].iloc[i] = df.loc[sku].values[-1]
            elif df.loc[sku]['Line']=='L1':
                data['TCB L1'].iloc[i] = df.loc[sku].values[-3]
                data['TCC L1'].iloc[i] = df.loc[sku].values[-2]
                data['TCP L1'].iloc[i] = df.loc[sku].values[-1]
            elif df.loc[sku]['Line']=='L2':
                data['TCB L2'].iloc[i] = df.loc[sku].values[-3]
                data['TCB L2'].iloc[i] = df.loc[sku].values[-2]
            warnings.warn('Production capacity has to be specified for all 3 lines (L1, L2, L3)')
        else:
            data['TCB L1'].iloc[i] = df.loc[sku].values[0, -3]
            data['TCB L2'].iloc[i] = df.loc[sku].values[1, -3]
            data['TCB L3'].iloc[i] = df.loc[sku].values[2, -3]

            data['TCC L1'].iloc[i] = df.loc[sku].values[0, -2]
            data['TCC L2'].iloc[i] = df.loc[sku].values[1, -2]
            data['TCC L3'].iloc[i] = df.loc[sku].values[2, -2]

            data['TCP L1'].iloc[i] = df.loc[sku].values[0, -1]
            data['TCP L2'].iloc[i] = df.loc[sku].values[1, -1]
            data['TCP L3'].iloc[i] = df.loc[sku].values[2, -1]


    mask = (data['TCC L1']*1.5 < data['PQ']) #extend for L2 and L3
    to_split = data[mask].index
    print(to_split)
    for i in to_split:
        new = data.iloc[i].copy()
        data['SKU'].iloc[i] += '_1'
        data['PQ'].iloc[i] /= 2
        new['SKU'] = new['SKU'] + '_2'
        new['PQ'] = new['PQ'] / 2
        data = data.append(new)

    return data


def engineer(data):
    data['TD'] = data['COC'] + data['forecast']
    shortlisted = data['TD'] > data['CIL']
    data['shortlisted'] = shortlisted
    return data


def quantify(data):
    scen_names = []
    for i in range(1, 7):
        scen_names.append('scenario {}'.format(i))
    #drop shortlisted
    mask = data['shortlisted'] == True
    data = data[mask]
    #assign scenarios
    scenarious = []
    pq = [] #production quantity
    n, m = data.shape
    for i in range(n):
        if data['category'].iloc[i] == 'C':
            scenarious.append(scen_names[-1])
            pq.append(data['COC'].iloc[i] - data['CIL'].iloc[i]) #(Customer Demand - Inventory)
        else:
            if data['TD'].iloc[i] > data['EPQ'].iloc[i] and data['CIL'].iloc[i] >= data['SS'].iloc[i]:   # If Total Demand > EPQ and Inventory >= Safety Stock
                scenarious.append(scen_names[0])
                pq.append(data['TD'].iloc[i] - data['CIL'].iloc[i]) #(Total Demand – Inventory + SS)
            elif data['TD'].iloc[i] > data['EPQ'].iloc[i] and data['CIL'].iloc[i] < data['SS'].iloc[i]: # If Total Demand > EPQ and Inventory < Safety Stock
                scenarious.append(scen_names[1])
                pq.append(data['TD'].iloc[i] - (data['CIL'].iloc[i] - data['SS'].iloc[i]))  # (Total Demand + (SS - Inventory))
            elif data['TD'].iloc[i] < data['EPQ'].iloc[i] and data['CIL'].iloc[i] >= data['SS'].iloc[i]: #If Total Demand < EPQ and Inventory >= Safety Stock
                scenarious.append(scen_names[2])
                pq.append(data['EPQ'].iloc[i]) #(EPQ)
            elif data['TD'].iloc[i] < data['EPQ'].iloc[i] and data['CIL'].iloc[i] < data['SS'].iloc[i]: # If Total Demand < EPQ and Inventory < Safety Stock
                scenarious.append(scen_names[3])
                pq.append(data['EPQ'].iloc[i] + (data['SS'].iloc[i] + data['CIL'].iloc[i])) #(EPQ + (SS-Inventory)):
            elif data['TD'].iloc[i] == 0 and data['CIL'].iloc[i] < data['SS'].iloc[i]: # If Total Demand = 0 (for the next 7 days) and Inventory < Safety Stock
                scenarious.append(scen_names[4])
                pq.append(data['EPQ'].iloc[i]) #(EPQ) or produce nothing in case the capacity is required by more priority items
            else:
                scenarious.append('TBD')
                pq.append(pd.NA)
                print('something went wrong!')

    data['scenario'] = scenarious
    data['PQ'] = pq
    return data

def plotter(data):
    to_plot = ['water', 'pack', 'bottle', 'category', 'client']
    for i in to_plot:
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x=i, y="PQ", data=data, capsize=.2, ci=None)
        plt.ylabel('Production quantity')
        plt.show()
    #sns.histplot(data['PQ'])
    plt.ylabel('SKUs')
    plt.show()

def epd_sort(data):
    #data = data.sort_values(by=['category', 'EPD'], ascending=['False', 'False'])
    data = data.sort_values(by=['category', 'EPD'], ascending=[False, False])
    return data


def allocate_line(data):
    data['PT'] = 24 * data['PQ'] / data['TCC L1']  # time to get the job done[h]
    #filtering out picky jobs
    mask1 = (data['TCC L1'].isna()) & (data['TCC L2'].isna())
    mask2 = (data['TCC L2'].isna()) & (data['TCC L3'].isna())
    L3DF = data[mask1].reset_index()
    L1DF = data[mask2].reset_index()
    new = data[~mask1]
    new_mask2 = (new['TCC L2'].isna()) & (new['TCC L3'].isna())
    remain = new[~new_mask2]

    cap_req = data['PQ'].sum()/3

    #will work correctly only if len(L1DF) == L1DF (have to extend later)
    remain.sort_values(by=['bottle', 'pack', 'water'], inplace=True)
    df = remain.reset_index()
    remain = remain.append(L3DF)
    remain = remain.append(L1DF)
    print(remain[['SKU', 'PQ', 'PT', 'bottle', 'pack', 'water']])

    # --------------------------------------- Line 1

    task1 = [L1DF.iloc[0]['SKU']]
    start1 = [0]
    finish1 = [L1DF.iloc[0]['PT']]
    duration1 = []
    duration1.append(L1DF.iloc[0]['PT'])
    #print(remain.columns)
    print('Capacity to allocate ', cap_req)
    used = 0.0
    i=0
    while used < 1.3*cap_req and i < len(df['SKU']):
        job = df.iloc[i]
        #print(df.iloc[i]['bottle'])
        if i>1 and df.iloc[i]['bottle'] != df.iloc[i-1]['bottle']:
            #used add time converted to PQ
            task1.append('Bottle changeover')
            start1.append(finish1[-1])
            duration1.append(4.0)
            finish1.append(start1[-1] + duration1[-1])
        if i>1 and df.iloc[i]['pack'] != df.iloc[i-1]['pack']:
            #used add time converted to PQ
            task1.append('Pack changeover')
            start1.append(finish1[-1])
            duration1.append(0.5)
            finish1.append(start1[-1] + duration1[-1])
        if i>1 and df.iloc[i]['water'] != df.iloc[i-1]['water']:
            #used add time converted to PQ
            task1.append('Water changeover')
            start1.append(finish1[-1])
            duration1.append(0.1667)
            finish1.append(start1[-1] + duration1[-1])
        used += df.iloc[i]['PQ']
        task1.append(job['SKU'])
        start1.append(finish1[-1])
        finish1.append(start1[-1] + job['PT'])
        duration1.append(job['PT'])
        i +=1


    # --------------------------------------- Line 2

    task2 = [L1DF.iloc[0]['SKU']]
    start2 = [0]
    finish2 = [L1DF.iloc[0]['PT']]
    duration2 = []
    duration2.append(L1DF.iloc[0]['PT'])
    #print(remain.columns)
    print('Capacity to allocate ', cap_req)
    used = 0.0
    while used < 0.8*cap_req and i < len(df['SKU']):
        job = df.iloc[i]
        #print(df.iloc[i]['bottle'])
        if i>1 and df.iloc[i]['bottle'] != df.iloc[i-1]['bottle']:
            #used add time converted to PQ
            task2.append('Bottle changeover')
            start2.append(finish2[-1])
            duration2.append(4.0)
            finish2.append(start2[-1] + duration2[-1])
        if i>1 and df.iloc[i]['pack'] != df.iloc[i-1]['pack']:
            #used add time converted to PQ
            task2.append('Pack changeover')
            start2.append(finish2[-1])
            duration2.append(0.5)
            finish2.append(start2[-1] + duration2[-1])
        if i>1 and df.iloc[i]['water'] != df.iloc[i-1]['water']:
            #used add time converted to PQ
            task2.append('Water changeover')
            start2.append(finish2[-1])
            duration2.append(0.1667)
            finish2.append(start2[-1] + duration2[-1])
        used += df.iloc[i]['PQ']
        task2.append(job['SKU'])
        start2.append(finish2[-1])
        finish2.append(start2[-1] + job['PT'])
        duration2.append(job['PT'])
        i +=1


    #--------------------------------------- Line 3

    task3 = [L1DF.iloc[0]['SKU']]
    start3 = [0]
    finish3 = [L1DF.iloc[0]['PT']]
    duration3 = []
    duration3.append(L1DF.iloc[0]['PT'])
    used3 = 0.0
    L3DF['PT'] = 24 * L3DF['PQ'] / L3DF['TCC L3']  # time to get the job done[h]
    df = df.append(L3DF)
    while i < len(df['SKU']):
        job = df.iloc[i]
        #print(df.iloc[i]['bottle'])
        if i>1 and df.iloc[i]['bottle'] != df.iloc[i-1]['bottle']:
            #used add time converted to PQ
            task3.append('Bottle changeover')
            start3.append(finish3[-1])
            duration3.append(4.0)
            finish3.append(start3[-1] + duration3[-1])
        if i>1 and df.iloc[i]['pack'] != df.iloc[i-1]['pack']:
            #used add time converted to PQ
            task3.append('Pack changeover')
            start3.append(finish3[-1])
            duration3.append(0.5)
            finish3.append(start3[-1] + duration3[-1])
        if i>1 and df.iloc[i]['water'] != df.iloc[i-1]['water']:
            #used add time converted to PQ
            task3.append('Water changeover')
            start3.append(finish3[-1])
            duration3.append(0.1667)
            finish3.append(start3[-1] + duration3[-1])
        used3 += df.iloc[i]['PQ']
        task3.append(job['SKU'])
        start3.append(finish3[-1])
        finish3.append(start3[-1] + job['PT'])
        duration3.append(job['PT'])
        i +=1


    # --------------------------------------- Line 1
    print(i)
        #print(i, ' {} capacity is used'.format(used))
    #counters
    bco = 0
    pco = 0
    wco = 0
    j = 0
    for i in range(len(task1)):
        if task1[i] == 'Pack changeover':
            if pco ==0:
                plt.barh(y=task1[i], left=start1[i], width=duration1[i], color='red', label='Pack changeover')
                pco += 1
            else:
                plt.barh(y=task1[i], left=start1[i], width=duration1[i], color='red')
        elif task1[i] == 'Water changeover':
            if wco ==0:
                plt.barh(y=task1[i], left=start1[i], width=duration1[i], color='teal', label='Water changeover')
                wco += 1
            else:
                plt.barh(y=task1[i], left=start1[i], width=duration1[i], color='teal')
        elif task1[i] == 'Bottle changeover':
            if bco ==0:
                plt.barh(y=task1[i], left=start1[i], width=duration1[i], color='pink', label='Bottle changeover')
                bco += 1
            else:
                plt.barh(y=task1[i], left=start1[i], width=duration1[i], color='pink')
        else:
            if j == 0:
                plt.barh(y=task1[i], left=start1[i], width=duration1[i], color='blue', label='Jobs')
                j += 1
            else:
                plt.barh(y=task1[i], left=start1[i], width=duration1[i], color='blue')

    my_lines = [24*i for i in range(2,7)]
    plt.axvline(x=24, linestyle='dashed', color='green', label='days')
    for l in my_lines:
        plt.axvline(x=l, linestyle='dashed', color='green')
    plt.xlabel('Time [h]')
    plt.title('Line 1')
    plt.legend()
    plt.grid()
    plt.show()


    #--------------------------------------- Line 2
    bco = 0
    pco = 0
    wco = 0
    j = 0
    for i in range(len(task2)):
        if task2[i] == 'Pack changeover':
            if pco == 0:
                plt.barh(y=task2[i], left=start2[i], width=duration2[i], color='red', label='Pack changeover')
                pco += 1
            else:
                plt.barh(y=task2[i], left=start2[i], width=duration2[i], color='red')
        elif task2[i] == 'Water changeover':
            if wco == 0:
                plt.barh(y=task2[i], left=start2[i], width=duration2[i], color='teal', label='Water changeover')
                wco += 1
            else:
                plt.barh(y=task2[i], left=start2[i], width=duration2[i], color='teal')
        elif task2[i] == 'Bottle changeover':
            if bco == 0:
                plt.barh(y=task2[i], left=start2[i], width=duration2[i], color='pink', label='Bottle changeover')
                bco += 1
            else:
                plt.barh(y=task2[i], left=start2[i], width=duration2[i], color='pink')
        else:
            if j == 0:
                plt.barh(y=task2[i], left=start2[i], width=duration2[i], color='blue', label='Jobs')
                j += 1
            else:
                plt.barh(y=task2[i], left=start2[i], width=duration2[i], color='blue')

    my_lines = [24 * i for i in range(2, 7)]
    plt.axvline(x=24, linestyle='dashed', color='green', label='days')
    for l in my_lines:
        plt.axvline(x=l, linestyle='dashed', color='green')
    plt.xlabel('Time [h]')
    plt.title('Line 2')
    plt.legend()
    plt.grid()
    plt.show()


    #--------------------------------------- Line 3
    bco = 0
    pco = 0
    wco = 0
    i = 0
    for i in range(len(task3)):
        if task3[i] == 'Pack changeover':
            if pco == 0:
                plt.barh(y=task3[i], left=start3[i], width=duration3[i], color='red', label='Pack changeover')
                pco += 1
            else:
                plt.barh(y=task3[i], left=start3[i], width=duration3[i], color='red')
        elif task3[i] == 'Water changeover':
            if wco == 0:
                plt.barh(y=task3[i], left=start3[i], width=duration3[i], color='teal', label='Water changeover')
                wco += 1
            else:
                plt.barh(y=task3[i], left=start3[i], width=duration3[i], color='teal')
        elif task3[i] == 'Bottle changeover':
            if bco == 0:
                plt.barh(y=task3[i], left=start3[i], width=duration3[i], color='pink', label='Bottle changeover')
                bco += 1
            else:
                plt.barh(y=task3[i], left=start3[i], width=duration3[i], color='pink')
        else:
            if j == 0:
                plt.barh(y=task3[i], left=start3[i], width=duration3[i], color='blue', label='Jobs')
                j += 1
            else:
                plt.barh(y=task3[i], left=start3[i], width=duration3[i], color='blue')

    my_lines = [24 * i for i in range(2, 7)]
    plt.axvline(x=24, linestyle='dashed', color='green', label='days')
    for l in my_lines:
        plt.axvline(x=l, linestyle='dashed', color='green')
    plt.xlabel('Time [h]')
    plt.title('Line 3')
    plt.legend()
    plt.grid()
    plt.show()

    def plot_production():
        bpd = 2243592
        bph = bpd/24.0
        changeovers = {'Bottle changeover', 'Pack changeover', 'Water changeover'}
        lines = ['Line 1', 'Line 2', 'Line 3']
        produced = [0, 0, 0]
        total_changeover = [0, 0, 0]
        for i in range(len(task1)):
            if task1[i] not in changeovers:
                produced[0] += bph*duration1[i]
            else:
                total_changeover[0] += duration1[i]
        for i in range(len(task2)):
            if task2[i] not in changeovers:
                produced[1] += bph*duration2[i]
            else:
                total_changeover[1] += duration2[i]
        for i in range(len(task3)):
            if task3[i] not in changeovers:
                produced[2] += bph*duration3[i]
            else:
                total_changeover[2] += duration3[i]
        sns.set_theme(style="whitegrid")
        sns.barplot(x=lines, y=produced, capsize=.2, ci=None)
        plt.ylabel('Bottles produced (millions)')
        plt.show()
        sns.set_theme(style="whitegrid")
        sns.barplot(x=lines, y=total_changeover, capsize=.2, ci=None)
        plt.ylabel('Total changeover time (hours)')
        plt.show()

    plot_production()
if __name__ == '__main__':
    data = epd_sort(get_msa(quantify(engineer(get_data()))))
    #plotter(data)
    allocate_line(data)

    #print(data[['SKU', 'EPD', 'category', 'PQ', 'TCC L1', 'TCC L2', 'TCC L3']].head(25))