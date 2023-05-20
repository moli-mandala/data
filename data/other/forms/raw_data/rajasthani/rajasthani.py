import re
import csv

lects = {
    'hadothi': {
        'F': 'patera',
        'B': 'gothda',
        'C': 'kawai',
        'E': 'pakkarana',
        'H': 'shyampura',
        'A': 'silori',
        'I': 'devpura',
        'G': 'piplia',
        'D': 'kelwada',
        'h': None
    },
    'mewati': {
        'G': 'goyla',
        'T': 'akera',
        'H': 'hathiya',
        'U': 'udaka',
        'S': 'gulpeda',
        'J': 'jakhopur',
        'Q': 'chirkana',
        'A': 'jhambaus',
        'h': None
    },
    'dhundari': {
        's': 'Pathalvas',
        'D': 'Deladi',
        'N': 'Bamore',
        'J': 'Jorpura',
        'B': 'Bhagatpur',
        'A': 'Badagaon',
        'P': 'Chalkoi',
        'h': None
    },
    'marwari': {
        'w': 'Mukheri',
        'g': 'Gomat',
        'D': 'Fatehgarh',
        'E': 'Husangsar',
        'd': 'Degana',
        'k': 'Kherwa',
        'N': 'Bagra',
        'F': 'Falna',
        'B': 'Bhagatpura',
        'A': 'Badagaon',
        'P': 'Chalkoi',
        'h': None
    },
    'mewari': {
        'Y': 'Gorana',
        'w': 'Kannouj',
        'y': 'Sangad',
        'k': 'Padarada',
        'l': 'Dindoli',
        'z': 'Kalnsas',
        'd': 'Dholpura',
        'b': 'Eklingpura',
        'c': 'Kishanji ka Kheda',
        'i': 'Bannoda',
        'j': 'Hurda',
        'G': 'Khor',
        'M': 'Ajmer',
        'f': 'Kalgav',
        'm': 'Basad',
        'J': 'Jesingpura',
        'n': 'Bhunyakhedi',
        'e': 'Pathera',
        'X': 'Godra',
        'h': None,
        'g': None
    },
    'bagri': {
        'A': 'Pallu',
        'B': 'Loonkansar',
        'C': 'Sardarsahar',
        'D': 'Old_Abaddi',
        'E': 'Makkasar',
        'F': 'Mirzawala',
        'J': 'Jamal',
        'K': 'Karnigedda',
        'L': 'Lakjikirani',
        'P': 'Panjkosi',
        'T': 'Fatehabad',
        'V': 'Mannaksar',
        'h': None
    }
}

with open(f'rajasthani.csv', 'w') as fout:
    fout.write('')

match_str = r'^(\*)?\d+\. ?'

rows = []
for file in lects:
    lines = []

    with open(file, 'r') as fin:
        for line in fin:
            line = line.strip('\n').lstrip()
            if line.strip().isdigit():
                continue
            elif re.match(match_str, line):
                if line[-1] in ['h', 'ɦ']:
                    toks = line.split(' ')
                    line = ' '.join(toks[:-1])
                    lines.append(re.sub(match_str, '', line) + ' | ')
                    lines.append(toks[-1])
                else:
                    lines.append(re.sub(match_str, '', line) + ' | ')
                print(line)
            else:
                lines[-1] = lines[-1] + line

    cur_gloss = None
    for line in lines:
        line = line.strip()
        print(f'line: "{line}"')
        if not line: continue
        gloss, text = line.split('|')
        gloss = gloss.strip()
        words = text.split(']')
        print(words)
        for word in words:
            word = word.strip()
            if not word: continue
            lemma, dialects = word.split('[')
            lemma = lemma.strip()
            dialects = dialects.strip()
            
            for dialect in dialects:
                if lects[file][dialect] == None: continue
                rows.append((file + '_' + lects[file][dialect].lower(), '', lemma, gloss, '', lemma, '', file))

rows = list(set(rows))
rows.sort(key=lambda x: (x[3], x[2]))
with open(f'rajasthani.csv', 'a') as fout:
    writer = csv.writer(fout)
    for row in rows:
        writer.writerow(row)
