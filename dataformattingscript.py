import re
import re
import os
import csv

raw_data = """Rank	Player	Club	Nationality	Stat	

Rank	Player	Club	Nationality	Stat	
1.	
João Palhinha
-	

Portugal
13	 
1.	
Marcos Senesi

Bournemouth

Argentina
13	 
3.	
Douglas Luiz
-	

Brazil
12	 
4.	
Edson Álvarez

West Ham United

Mexico
11	 
4.	
Nélson Semedo

Wolverhampton Wanderers

Portugal
11	 
4.	
Moisés Caicedo

Chelsea

Ecuador
11	 
4.	
João Gomes

Wolverhampton Wanderers

Brazil
11	 
4.	
Kai Havertz

Arsenal

Germany
11	 
4.	
James Tarkowski

Everton

England
11	 
10.	
Marc Cucurella

Chelsea

Spain
10	 
10.	
Wataru Endo

Liverpool

Japan
10	 
10.	
Anthony Gordon

Newcastle United

England
10	 
10.	
Nicolas Jackson

Chelsea

Senegal
10	 
10.	
Mario Lemina

Wolverhampton Wanderers

Gabon
10	 
10.	
Emerson

West Ham United

Italy
10	 
10.	
Jack Robinson
-	

England
10	 
10.	
Lucas Paquetá

West Ham United

Brazil
10	 
18.	
Anel Ahmedhodzic
-	

Bosnia & Herzegovina
9	 
18.	
Yves Bissouma

Tottenham Hotspur

Mali
9	 
18.	
Jayden Bogle
-	

England
9	 
"""

def parse_raw_data(raw, year="2023/2024", statName="Yellow Cards"):
    entries = re.split(r'\d+\.\s+', raw)[1:]  # Split by rank numbers
    data = []
    for entry in entries:
        lines = [line.strip() for line in entry.strip().split("\n") if line.strip()]  # Remove empty lines
        if len(lines) >= 4:  # Ensure enough data is present
            player = lines[0]
            club = lines[1] if lines[1] != "-" else "Unknown"
            nationality = lines[2]
            try:
                stat = int(lines[3].replace(",", ""))  # Convert stat to integer
            except ValueError:
                stat = 0  # Default to 0 if stat is invalid
            data.append({"Rank": None, "Player": player, "Club": club, "Year": year, statName: stat})
    return data

# Write data to CSV
def write_to_csv(data, filename="output.csv", statName="Yellow Cards"):
    for rank, item in enumerate(data, start=1):
        item["Rank"] = rank

    fieldnames = ["Rank", "Player", "Club", "Year", statName]

    file_exists = os.path.exists(filename)

    with open(filename, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:  # Write header only if file does not exist
            writer.writeheader()
        writer.writerows(data)

structured_data = parse_raw_data(raw_data, year="2023/2024")
write_to_csv(structured_data, filename="yellow_cards_stats.csv")
print("Data written to minutes_played_stats.csv")