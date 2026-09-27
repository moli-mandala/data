"""Exact page-image transcription, printed pp.176–177; | denotes printed alternates."""
import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PAGES={
(176,'left'):'''nangū|nāngā	foot
nāngar	plow
nēkkī	cut
nū,ē	drink
nīrī|nīrū	flee
oraso	year
oro	grain
pachna	blood
pagrī	turban
pāk	holy
pandarī	calf of leg
papalātī,ē	feed
pār-ī-pār	twelve o'clock
pārkom	cot
pārsī	language
pārūmkī	take across (a river)
pātā|pātoman	morning
perākū	abide; dwell; remain
pījīto	trouble
pinī	comb
pinjrā	cage
pītar	brass
poharā	boy
popā	hole
pūchī	rat
pūjaī	worship
pūlā|pūl	bridge
rabang	cold; cool
rāgo	anger
rān	medicine
rangī	hunger; hungry
rango	colour
rantī	tent
rātā	red
ringū,ē	forget
rū,ā	fever
rūtū	blackwood
sā,artē|sā,ā	take away''',
(176,'right'):'''sabkī	worship
sāgē	bring
sajā	place
sākom	leaf
sanain	old
sandūko	box
sangū,ī	blessing
sanī|senī	small
sanko	chain
sarka	line; straight
sarkār	the government; governor; ruler
saurūbē	run
shēnā	dung of cow, bull, or buffalo
shīdū	liquor
shim	hen
shingel|shingal	fire
shingrūp	evening
shokrā	bread; chapātī
shūkarī	hog
simil	sweet
similbā	is sweet
sing	tree
sipnā	teak
sirdī	ladder
sirī	she-goat
sītom	thread
sīsā	bottle
sobā	show; sight
sojū	fall (used when one falls down)
sūbangē	sit
sūnūm	oil
sūsīring	song
sūtrī	twine
sutū	first
sirī,ē	sing
sūlīkin	before
tālā	in the midst''',
(177,'left'):'''talwār	sword
tamāncha	pistol
tambā	copper
tambū	tent
tambūr	drum
tan	for
tapai,ē	fight
tāptī	trowel
tārē	girl
tarpai,kē	cast forth
tarsā	hyena
tārū|tārē	abide; dwell; remain
tautē	afterward
tawen	behind
te|ten	from
teng	to-day
tengin	there
tengenē	stand; wait
tharī	forest
tī	hand
tī,ā	brother-in-law by wife
tī,ār	ready
tī kālī	walking stick
tin	for
tingkē	cast forth
tīrīn	loan
tītīt	kind of bird
tobai	stumble
to,en	behind
tokatē	stumble
tolkē	bind
tongan	where
topā	cannon''',
(177,'right'):'''toprē	knee
totrā	meek
tsing	tree
tsūrī	knife
tūlē	raise
tūpo	ghee
tūr	squirrel
tū,ība	is floating
ūbdai	fall (in running)
ūbrā	sweat
ūjrī	sore
ūlā	vomit
ūmnā	example
ūnē	new
ūmkī	another
ūt	camel
ūthai|uthai,ē	seize
ūtū	dāl
ūrā	house
ūrī,ē	dress; put on ornaments
walen	arrived
watē	ground
wo,ī	yes
yādokī	remember
yam	cry
yaten	able
yebā	is giving
yih	who
yih	whose
yīñ	whose''',
}
def main():
 out=[]
 for (page,col),text in PAGES.items():
  for n,line in enumerate(text.splitlines(),1):
   form,gloss=line.split('\t')
   out.append(dict(printed_page=page,pdf_page=page+19,column=col,ordinal=n,reviewed_source_form=form,reviewed_english_gloss=gloss,review_note='Visually transcribed original page image 2026-09-26; alternates separated by |; source comma retained within forms.'))
 assert len(out)==138
 with (ROOT/'reverse_final_pages_review.tsv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t');w.writeheader();w.writerows(out)
if __name__=='__main__': main()
