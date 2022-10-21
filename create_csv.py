import re
import csv

date = re.compile(r'([a-zA-Z]+,\s\d+,\s\d+)')
winningNumbers = re.compile(r'((?m)^\d+)')
jackpot = re.compile('(\$\d*.?\d+\s\w+)')
index = []
dates = []
drawings = []
jackpots = []
count = 0
max = 12
draw = []
doublePlay = True
x = 0
i = 0

with open('powerball.txt', 'r') as f:
    data = f.read()
    dateMatches = date.finditer(data)
    numbersMatches = winningNumbers.finditer(data)
    jackpotMatches = jackpot.finditer(data)
    
    for match in dateMatches:
        dates.append(match.group(0))
        index.append(x)
        x = x + 1
        
    for match in numbersMatches:
        drawings.append(match.group(0))

    for match in jackpotMatches:
        jackpots.append(match.group(0))


with open('Powerball.csv', 'w', newline='') as f:
    header = ['INDEX','DATE','FIRST DRAW','SECOND DRAW','THIRD DRAW','FOURTH DRAW','FIFTH DRAW','POWERBALL','DP FIRST DRAW','DP SECOND DRAW','DP THIRD DRAW','DP FOURTH DRAW','DP FIFTH DRAW','DP POWERBALL','JACKPOT']
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    
    for d in drawings:
        if dates[i] == "August, 21, 2021":
            doublePlay = False
            max = 6

        draw.append(d)
        count = count + 1

        if count == max:
            if doublePlay == False:
                writer.writerow({'INDEX' : index[i],'DATE' : dates[i],'FIRST DRAW' : draw[0],'SECOND DRAW' : draw[1],'THIRD DRAW' : draw[2],'FOURTH DRAW' : draw[3],'FIFTH DRAW' : draw[4],'POWERBALL' : draw[5],'JACKPOT' : jackpots[i]})
                draw.clear()
                count = 0
                i = i + 1
            else:
                writer.writerow({'INDEX' : index[i],'DATE' : dates[i],'FIRST DRAW' : draw[0],'SECOND DRAW' : draw[1],'THIRD DRAW' : draw[2],'FOURTH DRAW' : draw[3],'FIFTH DRAW' : draw[4],'POWERBALL' : draw[5],'DP FIRST DRAW' : draw[6],'DP SECOND DRAW' : draw[7],'DP THIRD DRAW' : draw[8],'DP FOURTH DRAW' : draw[9],'DP FIFTH DRAW' : draw[10],'DP POWERBALL' : draw[11],'JACKPOT' : jackpots[i]})
                draw.clear()
                count = 0
                i = i + 1

print("CSV file written successfully!")