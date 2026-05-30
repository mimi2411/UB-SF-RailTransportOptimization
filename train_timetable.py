#Model - švedske železnice
from pulp import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from datetime import datetime, timedelta



# Inicijalizujemo model
model = LpProblem("Railway_Timetable_Optimization", LpMinimize)

# Parameters
nrT = 5  # broj vozova
nrS = 8  # broj deonica
M = 24*60*60  # velika konstanta 
Delta = 1  # interval sleđenja 


# Sets
Trains = range(1, nrT+1)
Sections = range(1, nrS+1)

# Input Data - broj vozova
tNr = {
    1: 322,
    2: 522,
    3: 523,
    4: 7315,
    5: 40661
}

# matrica vremena vožnje vozova po deonicama (u minutima)
d = {
    (1, 1): 15, (1, 2): 12, (1, 3): 5, (1, 4): 5, (1, 5): 10, (1, 6): 3, (1, 7): 3, (1, 8): 5,
    (2, 1): 14, (2, 2): 11, (2, 3): 3, (2, 4): 3, (2, 5): 6, (2, 6): 3, (2, 7): 3, (2, 8): 5,
    (3, 1): 12, (3, 2): 11, (3, 3): 3, (3, 4): 3, (3, 5): 6, (3, 6): 11, (3, 7): 11, (3, 8): 12,
    (4, 1): 5, (4, 2): 5, (4, 3): 10, (4, 4): 13, (4, 5): 25, (4, 6): 12, (4, 7): 14, (4, 8): 15,
    (5, 1): 13, (5, 2): 10, (5, 3): 5, (5, 4): 5, (5, 5): 10, (5, 6): 25, (5, 7): 30, (5, 8): 30
}

# vremena polaska vozova (u minutima)
b = {
    1: 313,
    2: 330,
    3: 366,
    4: 361,
    5: 340
}
#željeno vreme dolaska u kranju stanicu
e = {
    1: 391,
    2: 392,
    3: 428,
    4: 449,
    5: 482
}

#indeks voza za događaj k na deonici j 
index_t = {
    (1, 1): 1, (1, 2): 2, (1, 3): 3, (1, 4): 4, (1, 5): 5,
    (2, 1): 1, (2, 2): 2, (2, 3): 3, (2, 4): 4, (2, 5): 5,
    (3, 1): 1, (3, 2): 2, (3, 3): 3, (3, 4): 4, (3, 5): 5,
    (4, 1): 1, (4, 2): 2, (4, 3): 3, (4, 4): 4, (4, 5): 5,
    (5, 1): 1, (5, 2): 2, (5, 3): 3, (5, 4): 4, (5, 5): 5,
    (6, 1): 1, (6, 2): 2, (6, 3): 3, (6, 4): 4, (6, 5): 5,
    (7, 1): 1, (7, 2): 2, (7, 3): 3, (7, 4): 4, (7, 5): 5,
    (8, 1): 1, (8, 2): 2, (8, 3): 3, (8, 4): 4, (8, 5): 5
}
#indeks voznog dođagaja za događaj k na listi događaja deonice j - Lj
index_e = {
    (1, 1): 8, (1, 2): 8, (1, 3): 1, (1, 4): 1, (1, 5): 1,
    (2, 1): 7, (2, 2): 7, (2, 3): 2, (2, 4): 2, (2, 5): 2,
    (3, 1): 6, (3, 2): 6, (3, 3): 3, (3, 4): 3, (3, 5): 3,
    (4, 1): 5, (4, 2): 5, (4, 3): 4, (4, 4): 4, (4, 5): 4,
    (5, 1): 4, (5, 2): 4, (5, 3): 5, (5, 4): 5, (5, 5): 5,
    (6, 1): 3, (6, 2): 3, (6, 3): 6, (6, 4): 6, (6, 5): 6,
    (7, 1): 2, (7, 2): 2, (7, 3): 7, (7, 4): 7, (7, 5): 7,
    (8, 1): 1, (8, 2): 1, (8, 3): 8, (8, 4): 8, (8, 5): 8
}

# Decision Variables
x_start = LpVariable.dicts("x_start", 
                         [(i, k) for i in Trains for k in Sections], 
                         lowBound=0)
x_end = LpVariable.dicts("x_end", 
                       [(i, k) for i in Trains for k in Sections], 
                       lowBound=0)

# Sequencing variables (binary)
lambda_vars = LpVariable.dicts("lambda", 
                              [(j, k, v) for j in Sections 
                               for k in range(1, nrT) 
                               for v in range(1, nrT - k + 1)], 
                              cat='Binary')

# Objective Function
model += lpSum(x_end[(i, nrS)] - x_start[(i, 1)] for i in Trains), "Total_Travel_Time"

# Constraints
# Train constraints
for i in Trains:
    for k in range(1, nrS):
        model += x_end[(i, k)] <= x_start[(i, k+1)], f"ExitAndEnter_{i}_{k}"
    
    for k in Sections:
        model += x_end[(i, k)] >= x_start[(i, k)] + d[(i, k)], f"MinDuration_{i}_{k}"

# Departure time windows
for i in range(1, 5):
    model += x_start[(i, 1)] >= b[i] - 15, f"StartTimeLower_{i}"
    model += x_start[(i, 1)] <= b[i] + 15, f"StartTimeUpper_{i}"

model += x_start[(5, 1)] >= b[5] - 30, "StartTimeLower_5"
model += x_start[(5, 1)] <= b[5] + 30, "StartTimeUpper_5"

# Passenger stop constraint
model += x_start[(1, 6)] - x_end[(1, 5)] <= 2, "PassengerStop_Train322"

# Infrastructure constraints
for j in Sections:
    for k in range(1, nrT):
        for v in range(1, nrT - k + 1):
            train_k = index_t[(j, k)]
            event_k = index_e[(j, k)]
            train_kv = index_t[(j, k+v)]
            event_kv = index_e[(j, k+v)]
            
            model += ((x_start[(train_k, event_k)] - x_end[(train_kv, event_kv)] >= 
            Delta * lambda_vars[(j, k, v)] - M * (1 - lambda_vars[(j, k, v)])), 
            f"OrderBefore_{j}_{k}_{v}")
            
            model += ((x_start[(train_kv, event_kv)] - x_end[(train_k, event_k)] >= 
            Delta * (1 - lambda_vars[(j, k, v)]) - M * lambda_vars[(j, k, v)]), 
            f"OrderAfter_{j}_{k}_{v}")

# Solve the model
model.solve()

# Print results
print("Status:", LpStatus[model.status])
if model.status == 1:
    print("Optimal Total Travel Time:", value(model.objective))
    
    print("\nTrain Schedules:")
    for i in Trains:
        print(f"\nTrain {tNr[i]} (ID {i}):")
        for k in Sections:
            print(f"  Section {k}: Start={value(x_start[(i, k)]):.1f}, End={value(x_end[(i, k)]):.1f} (Min duration: {d[(i, k)]})")

    print("\nSequencing Decisions:")
    for j in Sections:
        print(f"\nSection {j}:")
        for k in range(1, nrT):
            for v in range(1, nrT - k + 1):
                val = value(lambda_vars[(j, k, v)])
                if val > 0.5:
                    train1 = tNr[index_t[(j, k)]]
                    train2 = tNr[index_t[(j, k+v)]]
                    print(f"  {train1} before {train2}")
else:
    print("No optimal solution found")

# Prikupljanje podataka iz rešenja
train_data = {}
for i in Trains:
    train_num = tNr[i]
    schedule = []
    for k in Sections:
        start = value(x_start[(i, k)])
        end = value(x_end[(i, k)])
        duration = d[(i, k)]
        schedule.append((k, start, end, duration))
    train_data[train_num] = schedule

# Kreiranje grafikonova
plt.figure(figsize=(15, 10))

# Boje za različite vozove
colors = {
    322: '#1f77b4',
    522: '#ff7f0e',
    523: '#2ca02c',
    7315: '#d62728',
    40661: '#9467bd'
}

# Y-osne pozicije za svaki voz
y_positions = {train: idx*3 for idx, train in enumerate(tNr.values())}

# Crtanje svakog segmenta za svaki voz
for train_num, schedule in train_data.items():
    for section, start, end, duration in schedule:
        plt.barh(y_positions[train_num], 
                width=end-start, 
                left=start, 
                height=2, 
                color=colors[train_num],
                edgecolor='black',
                alpha=0.7)
        
        # Dodavanje teksta sa informacijama
        plt.text(start + (end-start)/2, y_positions[train_num], 
                f'S{section}\n{duration}min', 
                ha='center', va='center',
                color='black', fontsize=8)

# Dodavanje vertikalnih linija za sekcije
for section in Sections:
    plt.axvline(x=section*50, color='gray', linestyle='--', alpha=0.3)

# Formatiranje grafika
plt.yticks([y_positions[t] for t in tNr.values()], [f'Train {t}' for t in tNr.values()])
plt.xlabel('Vreme (minute)')
plt.title('Švedski železnički raspored - Grafikoni vozova')
plt.grid(True, axis='x', alpha=0.3)

# Legenda za sekcije
handles = [Rectangle((0,0),1,1, color=colors[t]) for t in tNr.values()]
plt.legend(handles, [f'Train {t}' for t in tNr.values()], loc='upper right')

plt.tight_layout()
plt.show()

# Alternativni prikaz - Gantogram po sekcijama
plt.figure(figsize=(15, 8))

# Y-osne pozicije za svaku sekciju
section_positions = {section: idx*3 for idx, section in enumerate(Sections)}

for train_num, schedule in train_data.items():
    for section, start, end, duration in schedule:
        plt.barh(section_positions[section], 
                width=end-start, 
                left=start, 
                height=1.5, 
                color=colors[train_num],
                edgecolor='black',
                alpha=0.7,
                label=f'Train {train_num}')
        
        # Dodavanje teksta
        plt.text(start + (end-start)/2, section_positions[section], 
                f'T{train_num}\n{duration}min', 
                ha='center', va='center',
                color='black', fontsize=8)

# Formatiranje grafika
plt.yticks([section_positions[s] for s in Sections], [f'Section {s}' for s in Sections])
plt.xlabel('Vreme (minute)')
plt.title('Švedski železnički raspored - Gantogram po sekcijama')
plt.grid(True, axis='x', alpha=0.3)

# Legenda bez duplikata
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.tight_layout()
plt.show()


#Kao grafikon 

# Pretpostavljamo da su vremena u minutama od 00:00
base_time = datetime(2023, 1, 1)  # Proizvoljan datum kao referenca

# Definišemo "stanice" (sekcije pruge)
stations = [f'Sekcija {i}' for i in range(1, nrS+1)]
#stations.reverse()  # Da bismo imali prvu sekciju na vrhu

# Priprema podataka za crtanje
plt.figure(figsize=(15, 8))

# Boje za različite vozove
colors = {
    322: '#1f77b4',
    522: '#ff7f0e',
    523: '#2ca02c',
    7315: '#d62728',
    40661: '#9467bd'
}

# Za svaki voz
for i in Trains:
    train_num = tNr[i]
    x_vals = []
    y_vals = []
    
    # Prikupljanje podataka za svaku sekciju
    for k in Sections:
        start_time = base_time + timedelta(minutes=value(x_start[(i, k)]))
        end_time = base_time + timedelta(minutes=value(x_end[(i, k)]))
        
        # Dodajemo tačke za početak i kraj sekcije
        x_vals.extend([start_time, end_time])
        y_vals.extend([stations[k-1]]*2)
    
    # Crtanje linije za voz
    plt.plot(x_vals, y_vals, 
             color=colors[train_num], 
             marker='o',
             markersize=6,
             linewidth=2,
             label=f'Voz {train_num}')
    
    # Dodavanje oznaka za vreme na početku i kraju svake sekcije
    for k in Sections:
        start_time = base_time + timedelta(minutes=value(x_start[(i, k)]))
        end_time = base_time + timedelta(minutes=value(x_end[(i, k)]))
        
        plt.text(start_time, stations[k-1], 
                 f"{start_time.strftime('%H:%M')}", 
                 ha='right', va='center', fontsize=8)
        plt.text(end_time, stations[k-1], 
                 f"{end_time.strftime('%H:%M')}", 
                 ha='left', va='center', fontsize=8)

# Formatiranje x-ose (vreme)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
plt.gcf().autofmt_xdate()

# Dodatna podešavanja
plt.title('Dijagram kretanja vozova - Švedske železnice')
plt.xlabel('Vreme')
plt.ylabel('Pružne sekcije')
plt.grid(True, axis='x', alpha=0.3)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()