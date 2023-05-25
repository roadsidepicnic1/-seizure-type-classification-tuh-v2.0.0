import pickle

# Dateiname der .pkl-Datei
file_name = 'szr_0_pid_1_type_fnsz.pkl'

# Datei öffnen und Daten laden
with open(file_name, 'rb') as file:
    data = pickle.load(file)

# Daten anzeigen oder weiter verarbeiten
print(data)
