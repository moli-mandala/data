import csv

with open('rau_2019.csv', 'r') as f, open('params.csv', 'w') as p, open('forms.csv', 'w') as fr:
    reader = csv.DictReader(f)
    params = csv.writer(p)
    forms = csv.writer(fr)

    for row in reader:
        num = str(int(row['ID'][2:]))
        desc = f"Pinnow: {row['pinnow']}.<br>MKCD: "
        row['mkcd_form'] = row['mkcd_form'].replace('*', '\\*')
        if row['mkcd_form'] != '—':
            desc += f"*{row['mkcd_form']}* '{row['mkcd_meaning']}' ({row['mkcd_no']}). Attested in {row['mkcd_attested']}."
        else: desc += '—.'

        params.writerow([f'm{num}', row['pmunda'], '', row['gloss'], desc])
        forms.writerow(['PMu', f'm{num}', row['pmunda'], row['gloss'], '', '', '', 'rau'])
        
        for entry in row:
            if not entry.endswith('_form') or 'mkcd' in entry: continue
            if row[entry] == '—': continue
            lang = entry.split('_')[0]
            source = row[f'{lang}_source'].replace('.', ':') if '.' in row[f'{lang}_source'] else row[f'{lang}_source']
            forms.writerow([lang, f'm{num}', row[entry], row['gloss'], '', '', '', source])